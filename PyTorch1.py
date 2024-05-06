import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import numpy as np
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assuming that we are on a CUDA machine, this should print a CUDA device:

print(device)
'''
Tensor1=torch.cuda.FloatTensor([[1,2,3],[4,5,6],[7,8,9]])
#print(Tensor1)
class Model_1(nn.Module):
    def __init__(self):
        super(Model_1, self).__init__()
        self.conv1=nn.Conv2d(1,20,5)#submodule: Conv2d:(二维)卷积层,特征提取层
        self.conv2=nn.Conv2d(20,20,5)
        self.pool1=nn.MaxPool2d(3,stride=2)#MaxPool2d:(二维最大)池化层,特征压缩层
        '''
        最大池化：最大池化从每个区域中选择最大值作为输出。它关注特征图中最显著的特征，提取出显著的信息，同时具有位置不变性的特点。
        平均池化：平均池化计算每个区域的平均值作为输出。它对特征的整体统计信息感兴趣，可以模糊细节并提取图像的整体特征。'''
        self.add_module("conv", nn.Conv2d(10, 20, 4))#于nn.Conv2d(10, 20,4) 解释:10是输入通道数input_channels，例如RGB图片是三原色，就是三个通道；而20是输出通道数output_channels；4是卷积核尺寸
        self.add_module("conv1", nn.Conv2d(20 ,10, 4))#4代表卷积核尺寸，此时为4*4，若为（3,5），则尺寸为3*5
    def forward(self,x):
        x=F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
#对于2d卷积层和3d卷积层区别的解释:处理的数据维度不同，二维处理二维图像居多，三维可能会处理三维影像,一维卷积层原理相同,Conv即卷积之意
model=Model_1()
model.cuda(device=0)#挂到0号GPU设备上
for sub_module in model.children():
    print(sub_module)
mm=nn.Conv1d(16,33,3,stride=2)
input = autograd.Variable(torch.randn(20, 16, 50))#20->batches;16->input_channels;50->length # autograd.Variable():type to autograd.Variable
m = nn.ReLU()#一种Non-Linear Activations ，非线性激活函数，即小于0的输入返回0，大于0的返回自身，非线性激活函数还有sigmond()等等
print(input)
print(m(input))