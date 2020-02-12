# -*- coding: utf-8 -*-
"""
Created on Sun Sep 16 10:01:14 2018

@author: carri
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn import init
affine_par = True
import sys
sys.path.append('./deeplab')
import units
import units.ConvGRU2 as ConvGRU
#区别于siamese_model_concat的地方就是采用的最标准的deeplab_v3的基础网络，然后加上了非对称的分支
from PAM import PAM_Module
import time
def conv3x3(in_planes, out_planes, stride=1):   # 3x3的卷积操作
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ASPP(nn.Module):
    def __init__(self, dilation_series, padding_series, depth):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1,1))
        self.conv= nn.Conv2d(2048, depth, 1,1)
        self.bn_x = nn.BatchNorm2d(depth)
        self.conv2d_0 = nn.Conv2d(2048, depth, kernel_size=1, stride=1)
        self.bn_0 = nn.BatchNorm2d(depth)
        self.conv2d_1 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[0], dilation=dilation_series[0])
        self.bn_1 = nn.BatchNorm2d(depth)
        self.conv2d_2 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[1], dilation=dilation_series[1])
        self.bn_2 = nn.BatchNorm2d(depth)
        self.conv2d_3 = nn.Conv2d(2048, depth, kernel_size=3, stride=1, padding=padding_series[2], dilation=dilation_series[2])
        self.bn_3 = nn.BatchNorm2d(depth)
        self.relu = nn.ReLU(inplace=True)
        self.bottleneck = nn.Conv2d( depth*5, 256, kernel_size=3, padding=1 )  #512 1x1Conv
        self.bn = nn.BatchNorm2d(256)
        self.prelu = nn.PReLU()
        self.sa = PAM_Module(256)
        self.conv51 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                    nn.BatchNorm2d(256, affine=affine_par),
                                    nn.ReLU())
        self.dropout = nn.Dropout2d(p=0.1)
        #for m in self.conv2d_list:
        #    m.weight.data.normal_(0, 0.01)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            
    def _make_stage_(self, dilation1, padding1):
        Conv = nn.Conv2d(2048, 256, kernel_size=3, stride=1, padding=padding1, dilation=dilation1, bias=True)#classes
        Bn = nn.BatchNorm2d(256)
        Relu = nn.ReLU(inplace=True)
        return nn.Sequential(Conv, Bn, Relu)
        

    def forward(self, x):
        #out = self.conv2d_list[0](x)
        #mulBranches = [conv2d_l(x) for conv2d_l in self.conv2d_list]
        size=x.shape[2:]
        image_features=self.mean(x)
        image_features=self.conv(image_features)
        image_features = self.bn_x(image_features)
        image_features = self.relu(image_features)
        image_features=F.upsample(image_features, size=size, mode='bilinear', align_corners=True)
        out_0 = self.conv2d_0(x)
        out_0 = self.bn_0(out_0) 
        out_0 = self.relu(out_0)
        out_1 = self.conv2d_1(x)
        out_1 = self.bn_1(out_1) 
        out_1 = self.relu(out_1)
        out_2 = self.conv2d_2(x)
        out_2 = self.bn_2(out_2) 
        out_2 = self.relu(out_2)
        out_3 = self.conv2d_3(x)
        out_3 = self.bn_3(out_3) 
        out_3 = self.relu(out_3)
        out = torch.cat([image_features, out_0, out_1, out_2, out_3], 1)
        out = self.bottleneck(out)
        out = self.bn(out)
        out = self.prelu(out)
        out = self.sa(out)
        out = self.conv51(out)
        out = self.dropout(out)
        #for i in range(len(self.conv2d_list) - 1):
        #    out += self.conv2d_list[i + 1](x)
        
        return out
  


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(ASPP, [ 6, 12, 18], [6, 12, 18], 512)
        self.main_classifier = nn.Conv2d(256, num_classes, kernel_size=1, bias= True)
        self.softmax = nn.Sigmoid()#nn.Softmax()
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, dilation_series, padding_series, num_classes):
        return block(dilation_series, padding_series, num_classes)

    def forward(self, x):
        input_size = x.size()[2:]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        fea = self.layer5(x)
        x = self.main_classifier(fea)
        #print("before upsample, tensor size:", x.size())
        x = F.upsample(x, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x = self.softmax(x)
        return fea, x

class CoattentionModel(nn.Module):
    def  __init__(self, block, layers, num_classes, all_channel=256, all_dim= 60):	#473./8=60
        super(CoattentionModel, self).__init__()
        self.encoder = ResNet(block, layers, num_classes) ##  使用ResNet来提取特征
        self.linear_e = nn.Linear(all_channel, all_channel,bias = False)
        self.channel = all_channel
        self.dim = all_dim*all_dim
        self.gate = nn.Conv2d(all_channel, 1, kernel_size  = 1, bias = False)
        self.gate_s = nn.Sigmoid()
        self.conv1 = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias = False)
        self.ConvGRU = ConvGRU.ConvGRUCell(all_channel, all_channel, all_dim, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(all_channel)
        self.conv_fusion = nn.Conv2d(all_channel*2, all_channel, kernel_size=3, padding=1, bias= True)
        self.relu_fusion = nn.ReLU(inplace=True)
        self.prelu = nn.ReLU(inplace=True)
        self.relu_m = nn.ReLU(inplace=True)
        self.main_classifier1 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        #self.main_classifier2 = nn.Conv2d(all_channel, num_classes, kernel_size=1, bias = True)
        self.softmax = nn.Sigmoid()
        self.propagate_layers = 5
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                #n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
                #init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                #init.xavier_normal(m.weight.data)
                #m.bias.data.fill_(0)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()    
    
		
    def forward(self, input1, input2, input3): #注意input2 可以是多帧图像
        
        #input1_att, input2_att = self.coattention(input1, input2) 
        input_size = input1.size()[2:]  # NCHW
        batch_num  = input1.size()[0]   # 这一批有多少数据
        exemplars, temp = self.encoder(input1)  # 前5层的输出
        querys, temp = self.encoder(input2)
        query1s, temp1 = self.encoder(input3)
        x1s = torch.zeros(batch_num,1,input_size[0], input_size[1]).cuda()
        x2s = torch.zeros(batch_num,1,input_size[0], input_size[1]).cuda()
        x3s = torch.zeros(batch_num,1,input_size[0], input_size[1]).cuda()
        start_time = time.time()
        for ii in range(batch_num):  # 对每一张图片
            exemplar = exemplars[ii,:,:,:][None].contiguous().clone()  # 调用contiguous()之后，PyTorch会开辟一块新的内存空间存放变换之后的数据，并会真正改变Tensor的内容，按照变换之后的顺序存放数据。
            query = querys[ii, :, :, :][None].contiguous().clone()
            query1 = query1s[ii,:,:,:][None].contiguous().clone()
            #print('size:', query.size())
            for passing_round in range(self.propagate_layers):  # 迭代5次

                attention1 = self.conv_fusion(torch.cat([self.generate_attention(exemplar, query),
                                         self.generate_attention(exemplar, query1)],1)) # message passing with concat operation
                attention2 = self.conv_fusion(torch.cat([self.generate_attention(query, exemplar),
                                        self.generate_attention(query, query1)],1))
                attention3 = self.conv_fusion(torch.cat([self.generate_attention(query1, exemplar),
                                        self.generate_attention(query1, query)],1))

                h_v1 = self.ConvGRU(attention1, exemplar)
                #h_v1 = self.relu_m(h_v1)
                h_v2 = self.ConvGRU(attention2, query)
                #h_v2 = self.relu_m(h_v2)
                h_v3 = self.ConvGRU(attention3, query1)
                #h_v3 = self.relu_m(h_v3)
                exemplar = h_v1.clone()
                query = h_v2.clone()
                query1 = h_v3.clone()


                print('attention size:', attention3[None].contiguous().size(), exemplar.size())
                if passing_round == self.propagate_layers -1:  # 迭代的最后一次
                    x1s[ii,:,:,:] = self.my_fcn(h_v1, exemplars[ii,:,:,:][None].contiguous(), input_size)
                    x2s[ii,:,:,:] = self.my_fcn(h_v2, querys[ii,:,:,:][None].contiguous(), input_size)
                    x3s[ii,:,:,:] = self.my_fcn(h_v3, query1s[ii,:,:,:][None].contiguous(), input_size)

        end_time = time.time()
        #print('network fedforward time:', end_time-start_time)
        return x1s, x2s, x3s, temp, temp1

    def message_fun(self,input):
        input1 = self.conv_fusion(input)
        input1 = self.relu_fusion(input1)
        return input1

    def generate_attention(self, exemplar, query):
        fea_size = query.size()[2:]	 
#		 #all_dim = exemplar.shape[1]*exemplar.shape[2]
        exemplar_flat = exemplar.view(-1, self.channel, fea_size[0]*fea_size[1]) #N,C,H*W
        query_flat = query.view(-1, self.channel, fea_size[0]*fea_size[1])
        exemplar_t = torch.transpose(exemplar_flat,1,2).contiguous()  #batch size x dim x num
        exemplar_corr = self.linear_e(exemplar_t) #
        A = torch.bmm(exemplar_corr, query_flat) # 计算两个tensor的矩阵乘法，torch.bmm(a,b),tensor a 的size为(b,h,w),tensor b的size为(b,w,h),注意两个tensor的维度必须为3.

        B = F.softmax(torch.transpose(A,1,2),dim=1)
        #query_att = torch.bmm(exemplar_flat, A).contiguous() #注意我们这个地方要不要用交互以及Residual的结构
        exemplar_att = torch.bmm(query_flat, B).contiguous()
        
        input1_att = exemplar_att.view(-1, self.channel, fea_size[0], fea_size[1])  
        #input2_att = query_att.view(-1, self.channel, fea_size[0], fea_size[1])
        input1_mask = self.gate(input1_att)
        #input2_mask = self.gate(input2_att)
        input1_mask = self.gate_s(input1_mask)
        #input2_mask = self.gate_s(input2_mask)
        input1_att = input1_att * input1_mask
        #input2_att = input2_att * input2_mask

        return input1_att

        #print('h_v size, h_v_org size:', torch.min(input1_att), torch.min(exemplar))

    def my_fcn(self, input1_att,  exemplar,  input_size): #exemplar,

        input1_att = torch.cat([input1_att, exemplar],1)
        input1_att  = self.conv1(input1_att )
        input1_att  = self.bn1(input1_att )
        input1_att  = self.prelu(input1_att )
        x1 = self.main_classifier1(input1_att)
        x1 = F.upsample(x1, input_size, mode='bilinear')  #upsample to the size of input image, scale=8
        #print("after upsample, tensor size:", x.size())
        x1 = self.softmax(x1)

        return x1 #, x2, temp  #shape: NxCx
    

def Res_Deeplab(num_classes=2):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes-1)
    return model

def GNNNet(num_classes=2):
    model = CoattentionModel(Bottleneck,[3, 4, 23, 3], num_classes-1)
	
    return model
