from torchvision.transforms import transforms

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}

class div255(object):
    def __init__(self):
        pass

    def __call__(self,tensor):
        tensor1=tensor/255
        return tensor1

def process():  # 仅针对np array输入的transforms
    normalize = __imagenet_stats
    t_list = [
        transforms.ToTensor(),  # 对于PIL图片会做归一化，但是对于numpy不会做归一化
        div255(),
        transforms.Normalize(**normalize)
    ]
    return transforms.Compose(t_list)
