#卷积神经网络
#手写数字识别
#有点问题：RuntimeError: mat1 and mat2 shapes cannot be multiplied (512x500 and 2000x500)：已解决，是在全连接层出问题，权重矩阵尺寸不适配

import torch
import torch.nn as nn
import torch.optim as op
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
import pandas as pd
import openpyxl

batch_size=512
learning_rate=0.02
num_epoches=20
Device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(Device)

train_loader=DataLoader(
    datasets.MNIST('data',train=True,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)
test_loader=DataLoader(
    datasets.MNIST('data',train=False,download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,),(0.3081,))
                   ])),
    batch_size=batch_size,shuffle=True
)

data1=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\附件2_d.xlsx")
data2=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\品类-日变化.xlsx",sheet_name="Sheet3")
data3=pd.read_excel(r"C:\Users\D\Documents\Tencent Files\3185220995\FileRecv\品类日利润.xlsx",sheet_name="Sheet3")
data2.columns=[1,2,3,4,5,6,'all']
data3.columns=[1,2,3,4,5,6]

list_1=list(data1['单品编码'])
list_all=list(data2['all'])
list_0=[[j/(i-j) for i,j in zip(data2[k],data3[k])] for k in range(1,7)]
list_0=torch.tensor(list_0)
list_0=torch.transpose(list_0,0,1)
#print(list_0)


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #主要定义或实例化不同类型的layer
        self.conv1=nn.Conv1d(1,10,5)
        self.conv2=nn.Conv1d(10,20,3)
        self.fc1=nn.Linear(20*10*10,500)#输出500个神经元,这个全连接层，其实是一个权重矩阵的大小，即2000*500
        self.fc2=nn.Linear(500,10)#输出10个神经元，权重矩阵同理，500*10
    def forward(self,x):
        #实现数据从输入到输出的具体变化过程
        in_size=x.size(0)
        out=self.conv1(x)
        out=F.relu(out)
        out=F.max_pool1d(out,2,2)
        out=self.conv2(out)
        out=F.relu(out)
        out=F.max_pool1d(out,2,2)
        out=out.view(in_size,-1)
        out=self.fc2(out)
        out=F.log_softmax(out,dim=1)
        return out

model=ConvNet().to(Device)
optimizer=op.Adam(model.parameters())


def train(model,device,train_loader,optimizer,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_loader):
        data,target=data.to(device),target.to(device)
        optimizer.zero_grad()
        output=model(data)
        loss=F.nll_loss(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%30==0:
            print('Train Epoch:{}[{}/{}({:.0f}%)]\tLoss:{:.6f}'.format(
                epoch,batch_idx*len(data),len(train_loader.dataset),
                100.*batch_idx/len(train_loader),loss.item()))
def test(model,device,test_loader):
    model.eval()
    test_loss=0
    correct=0
    with torch.no_grad():
        for data,target in test_loader:
            data,target=data.to(device),target.to(device)
            output=model(data)
            test_loss+=F.nll_loss(output,target,reduction='sum').item()
            pred=output.max(1,keepdim=True)[1]
            correct+=pred.eq(target.view_as(pred)).sum().item()
    test_loss/=len(test_loader.dataset)
    print('\nTest set:Average loss:{:.4f},Accuracy:{}/{}({:.0f}%)\n'.format(
        test_loss,correct,len(test_loader.dataset),
        100.*correct/len(test_loader.dataset)))
for epoch in range(1,num_epoches+1):
    train(model,Device,train_loader,optimizer,epoch)
    test(model,Device,test_loader)
