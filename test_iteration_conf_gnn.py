# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:53:20 2018

@author: carri
"""

import argparse
import torch
import torch.nn as nn
from torch.utils import data
import numpy as np
import pickle
import cv2
from torch.autograd import Variable  # variable默认是不需要求导的，即requires_grad属性默认为False
import torch.optim as optim
import scipy.misc
# uDNN使用的非确定性算法就会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题
'''
一般来讲，应该遵循以下准则：

如果网络的输入数据维度或类型上变化不大，设置  torch.backends.cudnn.benchmark = true  可以增加运行效率；
如果网络的输入数据在每次 iteration 都变化的话，会导致 cnDNN 每次都会去寻找一遍最优配置，这样反而会降低运行效率。
'''
import torch.backends.cudnn as cudnn  # cuDNN使用非确定性算法

import sys
import os
import os.path as osp
from dataloaders import PairwiseImg_test_new as db
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
import timeit # 计时
from PIL import Image
from collections import OrderedDict # 实现了对字典对象中元素的排序
import matplotlib.pyplot as plt
import torch.nn as nn

# from utils.colorize_mask import cityscapes_colorize_mask, VOCColorize
from scipy import ndimage
#from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
from deeplab.siamese_model_conf_gnn import GNNNet
from  torchvision.utils import save_image# 保存图片 结果
my_scales = [0.75, 1.0, 1.5]
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSPnet")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        help="voc12, cityscapes, or pascal-context")

    # GPU configuration
    parser.add_argument("--cuda", default=True, help="Run on CPU or GPU")
    parser.add_argument("--gpus", type=str, default="0",
                        help="choose gpu device.")
    parser.add_argument("--seq_name", default = 'bmx-bumps')
    parser.add_argument("--use_crf", default = 'True')
    parser.add_argument("--sample_range", default = 2)  # 一次样本的数量
    
    return parser.parse_args()

def configure_dataset_model(args):
    args.batch_size = 1# 1 card: 5, 2 cards: 10 Number of images sent to the network in one step, 16 on paper
    args.maxEpoches = 15 # 1 card: 15, 2 cards: 15 epoches, equal to 30k iterations, max iterations= maxEpoches*len(train_aug)/batch_size_per_gpu'),
    args.data_dir = 'F:\\data\\DAVIS-data\\DAVIS'   # 37572 image pairs
    args.data_list = 'F:\\data\\DAVIS-data\\DAVIS\\vallist.txt'  # Path to the file listing the images in the dataset
    args.ignore_label = 255     #The index of the label to ignore during the training
    args.input_size = '473, 473' #Comma-separated string with height and width of images
    args.num_classes = 2      #Number of classes to predict (including background)
    args.img_mean = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)       # saving model file and log record during the process of training
    args.restore_from = './snapshots/attention_agnn_51.pth'#'./snapshots/davis_iteration_conf_gnn3_sa/co_attention_davis_55.pth' #resnet50-19c8e357.pth''/home/xiankai/PSPNet_PyTorch/snapshots/davis/psp_davis_0.pth' #
    args.snapshot_dir = './snapshots/davis_iteration/'          #Where to save snapshots of the model
    args.save_segimage = True
    args.seg_save_dir = "./result/test/davis_iteration_conf_gnn3_sa_org_scale_batch"
    args.vis_save_dir = "./result/test/davis_vis"
    args.corp_size =(473, 473)
        


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal 
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
       You probably saved the model using nn.DataParallel, which stores the model in module, and now you are trying to load it 
       without DataParallel. You can either add a nn.DataParallel temporarily in your network for loading purposes, or you can 
       load the weights file, create a new ordered dict without the module prefix, and load it back 
    """
    state_dict_new = OrderedDict()
    #print(type(state_dict))
    for k, v in state_dict.items():
        #print(k)
        name = k[7:] # remove the prefix module.
        # My heart is broken, the pytorch have no ability to do with the problem.
        state_dict_new[name] = v
        if name == 'linear_e.weight':
            np.save('weight_matrix.npy',v.cpu().numpy())
    return state_dict_new

def sigmoid(inX): 
    return 1.0/(1+np.exp(-inX))#定义一个sigmoid方法，其本质就是1/(1+e^-x)

def main():
    args = get_arguments()
    print("=====> Configure dataset and model")
    configure_dataset_model(args)
    print(args)

    print("=====> Set GPU for training")
    if args.cuda:
        print("====> Use gpu id: '{}'".format(args.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")
    model = GNNNet(num_classes=args.num_classes)  # 加载网络结构
    for param in model.parameters():  # 在测试的时候，网络的参数并不需要再计算梯度
        param.requires_grad = False

    saved_state_dict = torch.load(args.restore_from, map_location=lambda storage, loc: storage) # torch.load 返回的是一个 OrderedDict.
    print(saved_state_dict.keys())
    # model.load_state_dict({k.replace('pspmodule.',''):v for k,v in torch.load(args.restore_from)['state_dict'].items()})
    model.load_state_dict( convert_state_dict(saved_state_dict["model"]) ) #convert_state_dict(saved_state_dict["model"])

    model.eval()  # 测试
    model.cuda()
    h, w = map(int, args.input_size.split(','))  # 将输入图片大小进行int类型转换
    input_size = (h, w) # 输入图片的大小
    # 加载测试集
    db_test = db.PairwiseImg(train=False, inputRes=input_size, db_root_dir=args.data_dir,  transform=None, seq_name = None, sample_range = args.sample_range, scales = my_scales) #db_root_dir() --> '/path/to/DAVIS-2016' train path
    testloader = data.DataLoader(db_test, batch_size= 1, shuffle=False, num_workers=0)
    # voc_colorize = VOCColorize()


    data_list = []

    if args.save_segimage:  # 保存测试结果
        if not os.path.exists(args.seg_save_dir) and not os.path.exists(args.vis_save_dir):
            os.makedirs(args.seg_save_dir)
            os.makedirs(args.vis_save_dir)
    print("======> test set size:", len(testloader))  # 测试集的大小
    my_index = 0
    old_temp=''
    for index, batch in enumerate(testloader): # 取出批次数据
        print('%d processd'%(index))
        targets = batch['target']  # 要分割的帧
        #search = batch['search']
        temp = batch['seq_name'] # 当前处理的视频名称
        args.seq_name=temp[0]    # 当前处理的视频名称
        print(args.seq_name)
        if old_temp==args.seq_name:  # 开始处理下一个视频序列
            my_index = my_index+1
        else:
            my_index = 0

        output_sum = []  # 输出
        search_im0s = batch['search_0']  # 参考帧1
        search_im1s = batch['search_1'] # 参考帧2
        first_image = np.array(Image.open(args.data_dir + '/JPEGImages/480p/blackswan/00000.jpg')) # 第一张图片
        original_shape = first_image.shape  # 原始的图片的大小
        for my_scale in range(0,len(my_scales)):
            # 获取当前scale的图片 （一共有三个scale）
            target = targets[my_scale]
            search_im0 = search_im0s[my_scale]
            search_im1 = search_im1s[my_scale]

            output = model(Variable(target, volatile=True).cuda(),Variable(search_im0, volatile=True).cuda(),Variable(search_im1, volatile=True).cuda()) # 计算输出
            pred1 = output[0].data.cpu()  # 预测的分割 1
            print('output size:', pred1.size())
            target_rs1 = F.interpolate(input=Variable(pred1), size=(original_shape[0], original_shape[1]), mode='bilinear', align_corners=True) # 上采样到原始图片的大小

            output_sum.append(target_rs1[0,0,:,:])  #
        
        output1 = torch.mean(torch.stack(output_sum, dim=0), 0)#output_sum/len(my_scales) #/2  取平均
        output1 = output1.cpu().numpy()
        mask = (output1*255).astype(np.uint8)  # 分割的mask
        print(mask.shape[0])  # 480
        mask = Image.fromarray(mask)  # 从一个数组中能生成一幅图像

        ##  保存分割结果
        save_dir_res = os.path.join(args.seg_save_dir, 'Results', args.seq_name)
        old_temp=args.seq_name
        if not os.path.exists(save_dir_res):
            os.makedirs(save_dir_res)
        if args.save_segimage:
            my_index1 = str(my_index).zfill(5)  #  前面添加0 直到有5位字符
            seg_filename = os.path.join(save_dir_res, '{}.png'.format(my_index1))
            #color_file = Image.fromarray(voc_colorize(output).transpose(1, 2, 0), 'RGB')
            mask.save(seg_filename)
            #np.concatenate((torch.zeros(1, 473, 473), mask, torch.zeros(1, 512, 512)),axis = 0)
            #save_image(output1 * 0.8 + target.data, args.vis_save_dir, normalize=True)

    

if __name__ == '__main__':
    main()
