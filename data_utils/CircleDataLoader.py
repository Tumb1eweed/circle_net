# *_*coding:utf-8 *_*
import os
import json
import warnings
import numpy as np
import h5py
from torch.utils.data import Dataset
warnings.filterwarnings('ignore')

def pc_normalize(pc,pc2):
    centroid = np.mean(pc, axis=1,keepdims = True)
    pc = pc - centroid
    pc2 = pc2 - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=2)),axis = 1, keepdims=True)
    pc = pc / np.expand_dims(m, axis=-1)
    pc2 = pc2 / np.expand_dims(m, axis=-1)
    return pc,pc2

class PartNormalDataset(Dataset):
    def __init__(self,root = './data/circle_data', npoints=500, split='train', class_choice=None, normal_channel=False):
        self.npoints = npoints
        self.root = root
        self.catfile = os.path.join(self.root, 'label.txt')
        self.cat = {}
        self.normal_channel = normal_channel

        self.catfile = os.path.join(self.root, 'data.hdf5')
        f = h5py.File(self.catfile,'r')

        self.data_cls0 = f["data_l0"][:]
        self.data_cls1 = f["data_l1"][:]

        size_cls = 50
        idx_c = np.array([81*j + i  for j in range(12) for i in range(size_cls)])

        #cad scan data
        self.data = self.data_cls0[idx_c,:,:3]    # Sub-point clouds cut from the complete point cloud. each sub-point cloud contains one circle hole. 500 points
        self.label = self.data_cls0[idx_c,:,3]    # label of circle boundary: 0 or 1.
        self.data2 = self.data_cls1[idx_c,:,:3]   # Point clouds provider complete neighborhood for sub-point. It is a bit larger than sub-point cloud. 2000 points
        self.orgdata = self.data_cls0[idx_c,:,:]
        print(self.data.shape, self.data2.shape)  
        self.data,self.data2= pc_normalize(self.data,self.data2)

        idx = np.arange(0,self.data.shape[0])
        np.random.seed(1111)
        np.random.shuffle(idx)
        self.data = self.data[idx]
        self.label = self.label[idx]
        self.data2 = self.data2[idx]
        self.orgdata = self.orgdata[idx]

        sample_size = int(self.data.shape[0] * 0.9+1)
        if(split == 'train'):
            self.data = self.data[:sample_size]
            self.label = self.label[:sample_size]
            self.data2 = self.data2[:sample_size]
            self.orgdata = self.orgdata[:sample_size]
        elif(split == 'test'):
            self.data = self.data[sample_size:]
            self.label = self.label[sample_size:]
            self.data2 = self.data2[sample_size:]
            self.orgdata = self.orgdata[sample_size:]
        print('The size of %s data is %d'%(split,self.data.shape[0]))

        # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
        self.seg_classes = {'circle': [0,1]}

        self.cache = {}  # from index to (point_set, cls, seg) tuple
        self.cache_size = 1000


    def __getitem__(self, index):
        if index in self.cache:
            point_set, seg, point_set2, radlabel ,rawdata, norm= self.cache[index]
        else:
            point_set = self.data[index].astype(np.float32)
            seg = self.label[index].astype(np.int32)
            point_set2 = self.data2[index].astype(np.float32)
            rawdata = self.orgdata[index].astype(np.float32)

            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, seg, point_set2 ,rawdata)
            choice = np.random.choice(seg.shape[0], self.npoints, replace=True)
            # resample
            point_set = point_set[choice, :]
            seg = seg[choice]
            rawdata = rawdata[choice, :]
        return point_set, seg, point_set2 ,rawdata, rawdata

    def __len__(self):
        return self.data.shape[0]



