from deepPruner.transformer.selfAttention import *
from timm.models.layers import trunc_normal_
import torch.nn as nn
class patchMerging(nn.Module):
    def __init__(self,patch_size,vec_dim):
        super(patchMerging, self).__init__()
        self.new_patch_size=patch_size
        self.current_patch_size=(patch_size//2)
        self.vec_dim=vec_dim
        self.current_h=0
        self.current_w=0
        # module
        self.embedding_layer=embedding(self.new_patch_size,patch_channels=vec_dim//((self.current_patch_size)**2),vec_dim=vec_dim*2)

    def features2patches(self,features):
        '''
        :param features:shape=[N,D,H,W]
        :return: output:shape=[N,patch_num/4,D,new_patch_size,new_patch_size]
        '''
        return rearrange(features,'b c (h p1) (w p2) ->b (h w) c p1 p2',p1=self.new_patch_size,p2=self.new_patch_size)

    def patches2features(self,patches):
        '''
        :param patches:shape=[N,C,P,P,patch_num]
        :return: features: shape=[N,C,H,W]
        '''
        return rearrange(patches,'b c p1 p2 (h w)-> b c (h p1) (w p2)',h=self.current_h,w=self.current_w)

    def forward(self,inputs,h,w):
        '''
        :param inputs: shape=[N,D,patch_num]
        :return: outputs: shape=[N,D*2,patch_num/4]
        '''
        self.current_h=h//self.current_patch_size
        self.current_w=w//self.current_patch_size
        N,D,patch_num=inputs.shape
        # [N,C,P,P,patch_num]
        out=inputs.view(N,-1,self.current_patch_size,self.current_patch_size,patch_num)
        out=self.patches2features(out)  # [N,C,H,W]
        out=self.features2patches(out)  # [N,patch_num/4,D,new_patch_size,new_patch_size]
        out=self.embedding_layer(out)
        return out

class SwinMultiHeadSA(nn.Module):
    def __init__(self,vec_dim,head_num,patch_size,window_size):
        super(SwinMultiHeadSA, self).__init__()
        self.D=vec_dim
        self.head_num=head_num
        self.patch_size=patch_size
        self.window_size=window_size

        # this h/w is represent the number of patch in h/w
        self.current_h=0
        self.current_w=0

        # module
        self.key_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.query_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.value_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.multiHead2SingleHead_layer=nn.Conv1d(in_channels=head_num*vec_dim,out_channels=vec_dim,kernel_size=1)
        self.key_embedding=None
        self.query_embedding=None
        self.value_embedding=None
        self.softmax=nn.Softmax(dim=3)
        self.N=None
        self.patch_num=None

        self.mask=None
        # param: relative position code
        self.relative_pos_code=nn.Parameter(torch.zeros(((self.window_size*2-1)**2,self.head_num)))
        trunc_normal_(self.relative_pos_code, std=.02)
        self.coord_h=torch.arange(self.window_size)
        self.coord_w=torch.arange(self.window_size)
        self.coord=torch.stack(torch.meshgrid([self.coord_h,self.coord_w]))
        coord_flatten=torch.flatten(self.coord,1)
        relative_coord=(coord_flatten[:,:,None]-coord_flatten[:,None,:]).permute([1,2,0])
        relative_coord[:,:,0]+=self.window_size-1
        relative_coord[:,:,1]+=self.window_size-1
        relative_coord[:,:,1]*=2 * self.window_size - 1
        self.relative_index=relative_coord.sum(-1)

    def window_partition(self,embeddings):
        '''
        group patches by window
        :param embeddings: shape=[N,head_num,D,patch_num]
        :return:shape= [N,head_num,window_num,d,patch_num_in_a_window]
        '''
        N,head_num,D,patch_num=embeddings.shape
        embeddings=embeddings.view(N,self.head_num,D,self.current_h,self.current_w)
        return rearrange(embeddings,'n h d (num_h win1) (num_w win2) -> n h (num_h num_w) d (win1 win2)',win1=self.window_size,win2=self.window_size)

    def window_merge(self,embeddings):
        '''
        merge patches by window
        :param embeddings:shape=[N,head_num,window_num,D,patch_num_in_a_window]
        :return: shape=[N,head_num,D,patch_num]
        '''
        embeddings=rearrange(embeddings,'n h (num_h num_w) d (win1 win2) -> n h d (num_h win1) (num_w win2)',num_h=self.current_h//(self.window_size),win1=self.window_size)
        return embeddings.view(self.N,self.head_num,self.D,-1)

    def patches2features(self,patches):
        '''
        :param patches:shape=[N,C,P,P,patch_num]
        :return: features: shape=[N,C,H,W]
        '''
        return rearrange(patches,'b c p1 p2 (h w)-> b c (h p1) (w p2)',h=self.current_h,w=self.current_w)

    def features2patches(self,features):
        return rearrange(features,'b c (h p1) (w p2) ->b (h w) c p1 p2',p1=self.patch_size,p2=self.patch_size)

    def forward(self,inputs,h,w):
        '''
        :param inputs:shape=[N,D,patch_num]
        :return: shape=[N,D,patch_num]
        '''
        self.current_h=h//self.patch_size
        self.current_w=w//self.patch_size
        self.N,_,self.patch_num=inputs.shape
        device=inputs.get_device()
        # cycle shift
        inputs=inputs.view(self.N,-1,self.patch_size,self.patch_size,self.patch_num)
        inputs=self.patches2features(inputs)
        inputs=torch.roll(inputs,(self.window_size//2,self.window_size//2),(-2,-1))
        inputs=self.features2patches(inputs).view(self.N,self.patch_num,self.D).permute([0,2,1])
        # embedding
        self.key_embedding=self.key_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)  # [N,head_num,D,patch_num]
        self.query_embedding=self.query_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.value_embedding=self.value_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        # window partition
        self.key_embedding=self.window_partition(self.key_embedding)
        self.query_embedding=self.window_partition(self.query_embedding)
        self.value_embedding=self.window_partition(self.value_embedding)
        self.key_embedding=torch.transpose(self.key_embedding,dim0=3,dim1=4)
        # attention
        attention_map=torch.matmul(self.key_embedding,self.query_embedding)  # [N,head_num,window_num,patch_num_in_a_window,patch_num_in_a_window]
        attention_map/=(self.D)**0.5
        # mask
        # N,head_num,window_num,patch_num_win,patch_num_win
        index1=(slice(0,-self.window_size),slice(-self.window_size,-(self.window_size//2)),slice(-(self.window_size//2),None))
        feature_mask=torch.zeros((self.current_h,self.current_w))
        cnt=0
        for y in index1:
            for x in index1:
                feature_mask[y,x]=cnt
                cnt+=1
        feature_mask=rearrange(feature_mask,'(h wins1) (w wins2) ->(h w) (wins1 wins2)',wins1=self.window_size,wins2=self.window_size)
        self.mask=(feature_mask.unsqueeze(1)-feature_mask.unsqueeze(2))
        self.mask=self.mask.masked_fill(self.mask!=0,float(-100.0)).masked_fill(self.mask==0,float(0.0))

        attention_map+=self.mask.to(device)
        # add the position code
        relative_map=self.relative_pos_code[self.relative_index.view(-1)].view(self.window_size*self.window_size,self.window_size*self.window_size,-1).permute([2,0,1])
        attention_map+=relative_map.unsqueeze(1).unsqueeze(0).contiguous()
        # softmax
        weight_map=self.softmax(attention_map)
        outputs=torch.matmul(self.value_embedding,weight_map)  # [N,head_num,window_num,D,patch_num_in_a_window]
        # window mergeF
        outputs=self.window_merge(outputs).view(self.N,-1,self.patch_num)
        # multi-head to single head
        outputs=self.multiHead2SingleHead_layer(outputs)
        return outputs

class WinMultiHeadSA(nn.Module):
    def __init__(self,vec_dim,head_num,patch_size,window_size):
        super(WinMultiHeadSA, self).__init__()
        self.window_size=window_size
        self.head_num=head_num
        self.patch_size=patch_size
        self.current_h=0
        self.current_w=0
        # module
        self.key_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.query_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.value_embedding_layer=nn.Conv1d(in_channels=vec_dim,out_channels=head_num*vec_dim,kernel_size=1)
        self.multiHead2SingleHead_layer=nn.Conv1d(in_channels=head_num*vec_dim,out_channels=vec_dim,kernel_size=1)
        self.key_embedding=None
        self.query_embedding=None
        self.value_embedding=None
        self.softmax=nn.Softmax(dim=3)
        self.N=None
        self.D=vec_dim
        self.patch_num=None
        # param: relative position code
        self.relative_pos_code=nn.Parameter(torch.zeros(((self.window_size*2-1)**2,self.head_num)))
        trunc_normal_(self.relative_pos_code, std=.02)
        self.coord_h=torch.arange(self.window_size)
        self.coord_w=torch.arange(self.window_size)
        self.coord=torch.stack(torch.meshgrid([self.coord_h,self.coord_w]))
        coord_flatten=torch.flatten(self.coord,1)
        relative_coord=(coord_flatten[:,:,None]-coord_flatten[:,None,:]).permute([1,2,0])
        relative_coord[:,:,0]+=self.window_size-1
        relative_coord[:,:,1]+=self.window_size-1
        relative_coord[:,:,1]*=2 * self.window_size - 1
        self.relative_index=relative_coord.sum(-1)

    def window_partition(self,embeddings):
        '''
        group patches by window
        :param embeddings: shape=[N,head_num,D,patch_num]
        :return:shape= [N,head_num,window_num,d,patch_num_in_a_window]
        '''
        N,head_num,D,patch_num=embeddings.shape
        embeddings=embeddings.view(N,self.head_num,D,self.current_h,self.current_w)
        return rearrange(embeddings,'n h d (num_h win1) (num_w win2) -> n h (num_h num_w) d (win1 win2)',win1=self.window_size,win2=self.window_size)

    def window_merge(self,embeddings):
        '''
        merge patches by window
        :param embeddings:shape=[N,head_num,window_num,D,patch_num_in_a_window]
        :return: shape=[N,head_num,D,patch_num]
        '''
        embeddings=rearrange(embeddings,'n h (num_h num_w) d (win1 win2) -> n h d (num_h win1) (num_w win2)',num_h=self.current_h//(self.window_size),win1=self.window_size)
        return embeddings.view(self.N,self.head_num,self.D,-1)

    def forward(self,inputs,h,w):
        '''
        :param inputs:shape=[N,D,patch_num]
        :return: shape=[N,D,patch_num]
        '''
        self.current_h=h//self.patch_size
        self.current_w=w//self.patch_size
        self.N,_,self.patch_num=inputs.shape
        self.key_embedding=self.key_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)  # [N,head_num,D,patch_num]
        self.query_embedding=self.query_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.value_embedding=self.value_embedding_layer(inputs).view(self.N,-1,self.D,self.patch_num)
        self.key_embedding=self.window_partition(self.key_embedding)
        self.query_embedding=self.window_partition(self.query_embedding)
        self.value_embedding=self.window_partition(self.value_embedding)
        self.key_embedding=torch.transpose(self.key_embedding,dim0=3,dim1=4)
        attention_map=torch.matmul(self.key_embedding,self.query_embedding)  # [N,head_num,window_num,patch_num_in_a_window,patch_num_in_a_window]
        attention_map/=(self.D)**0.5
        # add the position code
        relative_map=self.relative_pos_code[self.relative_index.view(-1)].view(self.window_size*self.window_size,self.window_size*self.window_size,-1).permute([2,0,1])
        attention_map+=relative_map.unsqueeze(1).unsqueeze(0).contiguous()
        # softmax
        weight_map=self.softmax(attention_map)
        outputs=torch.matmul(self.value_embedding,weight_map)  # [N,head_num,window_num,D,patch_num_in_a_window]
        outputs=self.window_merge(outputs).view(self.N,-1,self.patch_num)
        outputs=self.multiHead2SingleHead_layer(outputs)
        return outputs

class swinTransformerBlock(nn.Module):
    '''
    patch_size: the size of patch
    vec_dims: the dimensions of the patch vector
    head_dims: the number of multi-heads
    window_size: the number of the patches in a window
    '''
    def __init__(self,patch_size,vec_dims,head_dims,window_size):
        super(swinTransformerBlock, self).__init__()
        # parameters
        self.patch_size=patch_size
        self.vec_dims=vec_dims
        self.head_dims=head_dims
        self.window_size=window_size
        self.h=0
        self.w=0
        # modules
        # W-MSA
        self.LN1=nn.LayerNorm(self.vec_dims)
        self.W_MSA=WinMultiHeadSA(self.vec_dims,self.head_dims,self.patch_size,self.window_size)
        self.LN2=nn.LayerNorm(self.vec_dims)
        self.MLP1=nn.Sequential(
            nn.Conv1d(self.vec_dims,self.vec_dims,1),
            nn.GELU(),
            nn.Conv1d(self.vec_dims,self.vec_dims,1)
        )
        # SW-MSA
        self.LN3=nn.LayerNorm(self.vec_dims)
        self.SW_MSA=SwinMultiHeadSA(self.vec_dims,self.head_dims,self.patch_size,self.window_size)
        self.LN4=nn.LayerNorm(self.vec_dims)
        self.MLP2=nn.Sequential(
            nn.Conv1d(self.vec_dims,self.vec_dims,1),
            nn.GELU(),
            nn.Conv1d(self.vec_dims,self.vec_dims,1)
        )

    def forward(self,inputs,h,w):
        self.h=h
        self.w=w
        # W-MSA
        out=self.LN1(inputs.permute([0,2,1])).permute([0,2,1])
        out=self.W_MSA(out,self.h,self.w)
        out1=out+inputs
        out=self.LN2(out1.permute([0,2,1])).permute([0,2,1])
        out=self.MLP1(out)
        out2=out+out1
        # SW-MSA
        out=self.LN3(out2.permute([0,2,1])).permute([0,2,1])
        out=self.SW_MSA(out,self.h,self.w)
        out3=out+out2
        out3=self.LN4(out3.permute([0,2,1])).permute([0,2,1])
        out=self.MLP2(out3)
        out4=out+out3
        return out4

class swinTransformer(nn.Module):
    '''
    stage_num: the number of stage
    module_num_list: the number of swin transformer block in each stage
    in_channels: the input features' channel number
    vec_dim: the dimension of the patch embedding
    head_dims:the number of multi-head
    window_size: the size of the window (unit：number of the patch)
    '''
    def __init__(self,stage_num,module_num_list,patch_size,in_channels,vec_dims,head_dims,window_size):
        super(swinTransformer, self).__init__()
        # parameters
        self.stage_num=stage_num
        self.module_num_list=module_num_list
        assert self.stage_num==len(module_num_list)
        self.vec_dims=vec_dims
        self.patch_size=patch_size
        self.in_channels=in_channels
        self.head_dims=head_dims
        self.window_size=window_size
        self.h=0
        self.w=0
        self.scale=2**(self.stage_num-1)
        # modules
        # self.embedding_layer=embedding(self.patch_size,self.in_channels,self.vec_dims)
        self.stage_module_list=nn.ModuleList()
        for i in range(self.stage_num):
            self.stage_module_list.append(nn.ModuleList())
        for i,module in enumerate(self.stage_module_list):
            if i==0:
                self.stage_module_list[i].append(embedding(self.patch_size,self.in_channels,self.vec_dims))
            else:
                self.stage_module_list[i].append(patchMerging(self.patch_size*(2**i),self.vec_dims*(2**(i-1))))
            for j in range(self.module_num_list[i]):
                self.stage_module_list[i].append(swinTransformerBlock(self.patch_size*(2**i),self.vec_dims*(2**i),self.head_dims,self.window_size))

    def features2patches(self,features):
        return rearrange(features,'b c (h p1) (w p2) ->b (h w) c p1 p2',p1=self.patch_size,p2=self.patch_size)

    def forward(self,inputs,h,w):
        '''
        :param inputs: tensor with shape[N,C,H,W]
        :return:outputs: tensor with shape [N,C,H/(2^(stage_num-1)),W/(2^(stage_num-1))]
        '''
        self.h=h
        self.w=w
        patches=self.features2patches(inputs)
        out=patches
        for stage,module in enumerate(self.stage_module_list):
            for i in range(self.module_num_list[stage]):
                out=self.stage_module_list[stage][i](out,self.h,self.w)
        # keep the map size:
        # out=rearrange(out,'n (c ps1 ps2) (h w) -> n c (h ps1) (w ps2)',ps1=self.patch_size*self.scale,
        #               ps2=self.patch_size*self.scale,h=config.defualt_h//(self.patch_size*self.scale),w=config.defualt_w//(self.patch_size*self.scale))
        # reduce the map size:
        out=rearrange(out,'n (c ps1 ps2) (h w) -> n c (h ps1) (w ps2)',ps1=self.patch_size,ps2=self.patch_size,h=self.h//(self.patch_size*self.scale))
        return out


if __name__=="__main__":
    module_num_list=[4]
    swinT=swinTransformer(stage_num=1,module_num_list=module_num_list,patch_size=4,in_channels=3,vec_dims=64,
                          head_dims=8,window_size=8)
    inputs=torch.rand((3,256,512)).unsqueeze(0)
    out=swinT(inputs,h=256,w=512)
    print(out.shape)