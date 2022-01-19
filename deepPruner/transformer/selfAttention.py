import numpy as np
import torch
import time
import torch.nn as nn
from einops import rearrange
from torchsummary import summary
import torch.nn.functional as f
from PIL import Image

def relative_position_codeMat(C,H,W,device):
    # desgin a relative position code matrix to reflect the Manhattan Distance
    # It need 2 matrix to form the codeMat
    mat_x=torch.zeros((C,W)).to(device)
    mat_y=torch.zeros((C,H)).to(device)
    # form the raw mat
    for c in range(C//2):
        for x in range(W):
            theta=torch.tensor(x/(10000**(2*c/C)))
            mat_x[c*2][x]=torch.sin(theta)
            mat_x[c*2+1][x]=torch.cos(theta)
        for y in range(H):
            theta=torch.tensor(y/(10000**(2*c/C)))
            mat_y[c*2][y]=torch.sin(theta)
            mat_y[c*2+1][y]=torch.cos(theta)
    # now the mat is [C,H,W]
    mat_x=mat_x.expand(H,-1,-1).permute(1,0,2)
    mat_y=mat_y.expand(W,-1,-1).permute(1,2,0)
    mat_x=rearrange(mat_x,'c h w->(h w) c')
    mat_y=rearrange(mat_y,'c h w->(h w) c')
    relative_pos_code_x=torch.matmul(mat_x,torch.t(mat_x))
    relative_pos_code_y=torch.matmul(mat_y,torch.t(mat_y))
    return relative_pos_code_x+relative_pos_code_y

class embedding(nn.Module):
    def __init__(self,patch_size,patch_channels,vec_dim):
        super(embedding, self).__init__()
        self.embedding_layer=nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(patch_size*patch_size*patch_channels,vec_dim)
        )

    def forward(self,inputs,h,w):
        '''
        :param inputs: shape like [N,patch_num,C,P,P], here P is the size of patch
        :return: embedding vectors with dimension=vec_dim(or D), shape=[N,D,patch_num]
        '''
        embedding_vectors=torch.transpose(self.embedding_layer(inputs),dim0=1,dim1=2)
        return embedding_vectors

class SA(nn.Module):
    def __init__(self,vec_dim):
        super(SA, self).__init__()
        self.key_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=vec_dim,kernel_size=1)
        self.query_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=vec_dim,kernel_size=1)
        self.value_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=vec_dim,kernel_size=1)
        self.key_embedding=None
        self.query_embedding=None
        self.value_embedding=None
        self.softmax=nn.Softmax(dim=1)
        self.D=vec_dim

    def forward(self,inputs,pos_mat=None):
        '''
        :param inputs:shape=[N,D,patch_num]
        :return:outputs:shape=[N,D,patch_num]
        '''
        self.key_embedding=self.key_embedding_layer(inputs)
        self.query_embedding=self.query_embedding_layer(inputs)
        self.value_embedding=self.value_embedding_layer(inputs)
        self.key_embedding=torch.transpose(self.key_embedding,dim0=1,dim1=2)
        attention_map=torch.matmul(self.key_embedding,self.query_embedding)
        if pos_mat is not None:
            attention_map+=pos_mat
        attention_map/=(self.D)**0.5
        weight_map=self.softmax(attention_map)
        outputs=torch.matmul(self.value_embedding,weight_map)
        return outputs

class MultiHeadSA(nn.Module):
    def __init__(self,vec_dim,head_num):
        super(MultiHeadSA, self).__init__()
        self.key_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.query_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.value_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.multiHead2SingleHead_layer=nn.Conv1d(in_channels=head_num*vec_dim,out_channels=vec_dim,kernel_size=1)
        self.key_embedding=None
        self.query_embedding=None
        self.value_embedding=None
        self.softmax=nn.Softmax(dim=2)
        self.N=None
        self.D=vec_dim
        self.patch_num=None

    def forward(self,inputs,pos_mat=None):
        '''
        :param inputs:shape=[N,D,patch_num]
        :return:outputs:shape=[N,D,patch_num]
        '''
        self.N,_,self.patch_num=inputs.shape
        self.key_embedding=self.key_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.query_embedding=self.query_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.value_embedding=self.value_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.key_embedding=torch.transpose(self.key_embedding,dim0=2,dim1=3)
        attention_map=torch.matmul(self.key_embedding,self.query_embedding)
        attention_map/=(self.D)**0.5
        if pos_mat is not None:
            attention_map+=pos_mat
        weight_map=self.softmax(attention_map)
        outputs=torch.matmul(self.value_embedding,weight_map).view(self.N,-1,self.patch_num)
        outputs=self.multiHead2SingleHead_layer(outputs)
        return outputs

class Translayer(nn.Module):
    def __init__(self,vec_dim,head_num):
        super(Translayer, self).__init__()
        self.LN1=nn.LayerNorm(vec_dim)
        self.SA=MultiHeadSA(vec_dim,head_num)
        self.LN2=nn.LayerNorm(vec_dim)
        self.MLP=nn.Sequential(
            nn.Conv1d(vec_dim,vec_dim,1),
            nn.ReLU(),
            nn.Conv1d(vec_dim,vec_dim,1)
        )


    def forward(self,inputs,pos_mat=None):
        '''
        :param inputs:shape=[N,D,patch_num]
        :return: outputs:shape=[N,D,patch_num]
        '''
        norm1_output=self.LN1(inputs.permute([0,2,1])).permute([0,2,1])
        if pos_mat is not None:
            sa_output=self.SA(norm1_output,pos_mat)
        else:
            sa_output=self.SA(norm1_output)
        res_output=sa_output+inputs
        norm2_output=self.LN2(res_output.permute([0,2,1])).permute([0,2,1])
        mlp_output=self.MLP(norm2_output)
        outputs=mlp_output+res_output
        return outputs

class TransEncoder(nn.Module):
    def __init__(self,num_layer,patch_size,patch_channel,vec_dim,head_num):
        super(TransEncoder, self).__init__()
        self.embedding_layer=embedding(patch_size,patch_channel,vec_dim)
        self.translayers_with_pos_mat=Translayer(vec_dim,head_num)
        self.translayers=nn.Sequential(*([Translayer(vec_dim,head_num)]*(num_layer-1)))
        self.patchsize=patch_size

    def features2patches(self,features):
        return rearrange(features,'b c (h p1) (w p2) ->b (h w) c p1 p2',p1=self.patchsize,p2=self.patchsize)

    def forward(self,inputs):
        '''
        :param inputs: shape=[N,C,H,W]
        :return: outputs: shape=[N,D,patch_num]
        '''
        N,C,H,W=inputs.shape
        device=inputs.get_device()
        patches=self.features2patches(inputs)
        embeddings=self.embedding_layer(patches)
        pos_mat=relative_position_codeMat(C,H//self.patchsize,W//self.patchsize,device)
        pos_mat=f.layer_norm(pos_mat,pos_mat.shape)
        out=self.translayers_with_pos_mat(embeddings,pos_mat)
        output=self.translayers(out)
        return output

class decoder(nn.Module):
    def __init__(self,H,W,patch_size,in_channel,out_channel):
        super(decoder, self).__init__()
        self.h_scale=H//patch_size
        self.w_scale=W//patch_size
        self.P=patch_size
        self.mlp=nn.Conv1d(in_channel,out_channel*patch_size*patch_size,1)

    def patches2features(self,patches):
        '''
        :param patches:shape=[N,C,P,P,patch_num]
        :return: features: shape=[N,C,H,W]
        '''
        return rearrange(patches,'b c p1 p2 (h w)-> b c (h p1) (w p2)',h=self.h_scale,w=self.w_scale)

    def forward(self,inputs):
        '''
        :param inputs:shape =[N,D,patch_num]
        :return: outputs:shape =[N,C,H,W]
        '''
        N,D,patch_num=inputs.shape
        outputs=self.mlp(inputs)  # [N,out_channel,patch_num]
        outputs=outputs.view(N,-1,self.P,self.P,patch_num)
        outputs=self.patches2features(outputs)
        return outputs

if __name__=='__main__':
    img=Image.open('/data/kitti_scene/testing/image_2/000000_10.png')
    img=img.crop((0,0,512,256))
    img.save('img.png')
    img=np.array(img)
    raw=torch.from_numpy(img).permute([2,0,1]).unsqueeze(0).to('cuda').float()
    raw=f.interpolate(raw,scale_factor=(0.25,0.25))
    print(raw.shape)
    pos_mat=relative_position_codeMat(8,32,64,'cuda')
    TRE=TransEncoder(num_layer=4,patch_size=2,patch_channel=3,vec_dim=8,head_num=8,pos_mat=pos_mat).to('cuda')
    DE=decoder(W=64,H=128,patch_size=2,in_channel=32,out_channel=32).to('cuda')
    start=time.time()
    out=TRE(raw)
    end= time.time()
    print("time cost:",end-start,"s")
    print(torch.cuda.max_memory_allocated()/1024/1024,'MB')
    # out=DE(out)
    print(out.shape)
