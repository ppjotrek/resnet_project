import torch
from torch import nn

class SEBlock(torch.nn.Module):

  def __init__(self,channel,reduction=16):
    super(SEBlock,self).__init__()
    self.avg_pool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Sequential(
        nn.Linear(channel,channel//reduction,bias=False),
        nn.ReLU(inplace=True),
        nn.Linear(channel//reduction,channel,bias=False),
        nn.Sigmoid()
    )

  def forward(self,x):
    b,c,_= x.size()
    y = self.avg_pool(x).view(b,c)
    y = self.fc(y).view(b,c,1)
    return x*y.expand_as(x)
  
class ResSeBasicBlock(nn.Module):
  def __init__(self,in_channels,channels,stride=1,reduction=16,downsample = None) :
    super(ResSeBasicBlock,self).__init__()
    self.conv1 = nn.Conv1d(in_channels,channels,3,stride,padding=1,bias=False)
    self.bn1 = nn.BatchNorm1d(channels)
    self.elu = nn.ELU(inplace=True)
    self.conv2 = nn.Conv1d(channels,channels,3,1,padding=1,bias=False)
    self.bn2 = nn.BatchNorm1d(channels)
    self.se = SEBlock(channels,reduction)
    self.downsample = downsample

  def forward(self,x):
    residual = x
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.elu(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.se(out)

    if self.downsample is not None:
      residual = self.downsample(x)

    out+=residual
    out = self.elu(out)

    return out
  
class SERes1d(nn.Module):

  def __init__(self,in_channels,num_classes):

    super(SERes1d,self).__init__()

    self.conv1 = nn.Conv1d(in_channels,64,kernel_size=7, stride=2, padding=3, bias=False)
    self.bn1 = nn.BatchNorm1d(64)
    self.elu = nn.ELU()
    self.maxpool = nn.MaxPool1d(kernel_size=3,stride=2,padding=1)


    self.SE1_1 = ResSeBasicBlock(64,64)
    self.SE1_2 = ResSeBasicBlock(64,64)
    self.SE1_3 = ResSeBasicBlock(64,64)
    # downsample = None
    downsample = nn.Sequential(
        nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(128)
        )

    self.SE2_1 = ResSeBasicBlock(64,128,downsample=downsample,stride=2)
    self.SE2_2 = ResSeBasicBlock(128,128)
    self.SE2_3 = ResSeBasicBlock(128,128)
    self.SE2_4 = ResSeBasicBlock(128,128)

    downsample = nn.Sequential(
        nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(256)
        )
    self.SE3_1 = ResSeBasicBlock(128,256,downsample=downsample,stride=2)
    self.SE3_2 = ResSeBasicBlock(256,256)
    self.SE3_3 = ResSeBasicBlock(256,256)
    self.SE3_4 = ResSeBasicBlock(256,256)
    self.SE3_5 = ResSeBasicBlock(256,256)
    self.SE3_6 = ResSeBasicBlock(256,256)
    downsample = nn.Sequential(
        nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False),
        nn.BatchNorm1d(512)
        )
    self.SE4_1 = ResSeBasicBlock(256,512,downsample=downsample,stride=2)
    self.SE4_2 = ResSeBasicBlock(512,512)
    self.SE4_3 = ResSeBasicBlock(512,512)

    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512, num_classes)

  def forward(self,x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.elu(x)
    x = self.maxpool(x)

    x = self.SE1_1(x)
    x = self.SE1_2(x)
    x = self.SE1_3(x)

    x = self.SE2_1(x)
    x = self.SE2_2(x)
    x = self.SE2_3(x)
    x = self.SE2_4(x)

    x = self.SE3_1(x)
    x = self.SE3_2(x)
    x = self.SE3_3(x)
    x = self.SE3_4(x)
    x = self.SE3_5(x)
    x = self.SE3_6(x)

    x = self.SE4_1(x)
    x = self.SE4_2(x)
    x = self.SE4_3(x)

    x = self.avgpool(x)
    x = torch.flatten(x,1)
    x = self.fc(x)

    return x
