# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 11:39:54 2018

@author: carri
"""
# for testing case
from __future__ import division

import os
import numpy as np
import cv2
from scipy.misc import imresize
import scipy.misc 
import random

# from dataloaders.helpers import *
from torch.utils.data import Dataset

def flip(I,flip_p):
    if flip_p>0.5:
        return np.fliplr(I)
    else:
        return I

def scale_im(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims).astype(float)

def scale_gt(img_temp,scale):
    new_dims = (  int(img_temp.shape[0]*scale),  int(img_temp.shape[1]*scale)   )
    return cv2.resize(img_temp,new_dims,interpolation = cv2.INTER_NEAREST).astype(float)

def my_crop(img,gt):
    H = int(0.9 * img.shape[0])
    W = int(0.9 * img.shape[1])
    H_offset = random.choice(range(img.shape[0] - H))
    W_offset = random.choice(range(img.shape[1] - W))
    H_slice = slice(H_offset, H_offset + H)
    W_slice = slice(W_offset, W_offset + W)
    img = img[H_slice, W_slice, :]
    gt = gt[H_slice, W_slice]
    
    return img, gt

class PairwiseImg(Dataset):
    """DAVIS 2016 dataset constructed using the PyTorch built-in functionalities"""

    def __init__(self, train=True,
                 inputRes=None,
                 db_root_dir='F:\\data\\DAVIS-data\\DAVIS',
                 transform=None,
                 meanval=(104.00699, 116.66877, 122.67892),
                 seq_name=None, sample_range=10, scales = None):
        """Loads image to label pairs for tool pose estimation
        db_root_dir: dataset directory with subfolders "JPEGImages" and "Annotations"
        """
        self.train = train
        self.range = sample_range
        self.inputRes = inputRes  #  图片的大小
        self.db_root_dir = db_root_dir
        self.transform = transform
        self.meanval = meanval
        self.seq_name = seq_name
        self.scales = scales
        if self.train:
            fname = 'train_seqs'
        else:
            fname = 'val_seqs'

        if self.seq_name is None: #所有的数据集都参与训练
            with open(os.path.join(db_root_dir, fname + '.txt')) as f:
                seqs = f.readlines() # 读取全部视频序列的名称
                img_list = [] # 保存当前视频序列每一帧的路径
                labels = [] # 保存当前视频序列每一帧标签的路径
                Index = {} # 保存当前视频序列的下标范围
                for seq in seqs:                    
                    images = np.sort(os.listdir(os.path.join(db_root_dir, 'JPEGImages/480p/', seq.strip('\n')))) # 所有视频序列帧的名称
                    images_path = list(map(lambda x: os.path.join('JPEGImages/480p/', seq.strip(), x), images)) # 所有视频序列每一帧的路径
                    images_path = list(map(lambda x: x.replace('/', '\\'), images_path))
                    start_num = len(img_list)  # 当前视频序列的开始下标  为当前imglist中图片的数量
                    img_list.extend(images_path)
                    end_num = len(img_list)  # 当前视频序列的结束下标  为添加当前视频序列之后的图片的数量
                    Index[seq.strip('\n')]= np.array([start_num, end_num]) # 保存当前视频序列的 的下标范围

                    #  标签
                    lab = np.sort(os.listdir(os.path.join(db_root_dir, 'Annotations/480p/', seq.strip('\n'))))
                    lab_path = list(map(lambda x: os.path.join('Annotations/480p/', seq.strip(), x), lab))
                    labels.extend(lab_path)
        else: #针对所有的训练样本， img_list存放的是图片的路径

            # Initialize the per sequence images for online training
            names_img = np.sort(os.listdir(os.path.join(db_root_dir, str(seq_name))))  # 所有视频序列帧的名称
            img_list = list(map(lambda x: os.path.join(( str(seq_name)), x), names_img)) # 获取所有视频序列帧的路径
            #name_label = np.sort(os.listdir(os.path.join(db_root_dir,  str(seq_name))))
            labels = [os.path.join( (str(seq_name)+'/saliencymaps'), names_img[0])]  #第一帧的anotation
            labels.extend([None]*(len(names_img)-1)) #在labels这个列表后面添加元素None
            if self.train:
                img_list = [img_list[0]]
                labels = [labels[0]]

        #assert (len(labels) == len(img_list))

        self.img_list = img_list
        self.labels = labels
        self.Index = Index
        #img_files = open('all_im.txt','w+')

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        target, sequence_name = self.make_img_gt_pair(idx) #测试时候要分割的帧
        target_id = idx
        seq_name1 = self.img_list[target_id].split('\\')[-2] #获取视频名称
        sample = {'target': target, 'seq_name': sequence_name, 'search_0': None}  # 样本 字典  { target ： 要分割的帧， seq_name： 视频名称， search_0： }
        if self.range >= 1:   #  default  = 2
            my_index = self.Index[seq_name1]  # 获取当前视频序列的起始下标
            search_num = list(range(my_index[0], my_index[1]))   # 视频序列的下标范围
            search_ids = random.sample(search_num, self.range)#min(len(self.img_list)-1, target_id+np.random.randint(1,self.range+1)) # 随机选取两帧的下标
        
            for i in range(0,self.range): #  （0 ~ 2）
                search_id = search_ids[i] # 获取第 i 帧的下标
                search, sequence_name = self.make_img_gt_pair(search_id)  # 获取这个帧和视频序列名称
                if sample['search_0'] is None:
                    sample['search_0'] = search  # search帧也有3个图片
                else:
                    sample['search'+'_'+str(i)] = search
            #np.save('search1.npy',search)
            #np.save('search_gt.npy',search_gt)
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)  # 这一帧的路径
                sample['fname'] = fname
       
        else:
            img, gt = self.make_img_gt_pair(idx)
            sample = {'image': img, 'gt': gt}
            if self.seq_name is not None:
                fname = os.path.join(self.seq_name, "%05d" % idx)
                sample['fname'] = fname

        return sample  #这个类最后的输出

    def make_img_gt_pair(self, idx): #这个函数存在的意义是为了getitem函数服务的
        """
        Make the image-ground-truth pair
        """
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[idx]), cv2.IMREAD_COLOR)   #  加载当前帧

        imgs = []
         ## 已经读取了image以及对应的ground truth可以进行data augmentation了
        for scale in self.scales:  # 对图片进行数据增强   my_scales = [0.75, 1.0, 1.5]

            if self.inputRes is not None:
                input_res = (int(self.inputRes[0]*scale),int(self.inputRes[0]*scale))
                img1 = cv2.resize(img.copy(), input_res)  # 对图片裁剪到指定大小

            img1 = np.array(img1, dtype=np.float32)
            #img = img[:, :, ::-1]
            img1 = np.subtract(img1, np.array(self.meanval, dtype=np.float32))  # 图片的每一个通道减去均值
            img1 = img1.transpose((2, 0, 1))  # NHWC -> NCHW  转换通道
            imgs.append(img1)

                #gt = gt/np.max([gt.max(), 1e-8])
        #np.save('gt.npy')
        sequence_name = self.img_list[idx].split('\\')[2] # 当前视频的名称
        return imgs, sequence_name  # 返回的三张增强后的图片和该视频的名称

    def get_img_size(self):
        img = cv2.imread(os.path.join(self.db_root_dir, self.img_list[0]))
        
        return list(img.shape[:2])


if __name__ == '__main__':
    import custom_transforms as tr
    import torch
    from torchvision import transforms
    from matplotlib import pyplot as plt

    transforms = transforms.Compose([tr.RandomHorizontalFlip(), tr.Resize(scales=[0.5, 0.8, 1]), tr.ToTensor()])

    #dataset = DAVIS2016(db_root_dir='/media/eec/external/Databases/Segmentation/DAVIS-2016',
                       # train=True, transform=transforms)
    #dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
#
#    for i, data in enumerate(dataloader):
#        plt.figure()
#        plt.imshow(overlay_mask(im_normalize(tens2image(data['image'])), tens2image(data['gt'])))
#        if i == 10:
#            break
#
#    plt.show(block=True)
