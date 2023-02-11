import os
import numpy as np
from torch.utils.data import Dataset
import scipy.io as scio

class loader_1d(Dataset):
    def __init__(self, data_dir, attr, split):
        self.split = split
        self.attr = attr
        input_dir = os.path.join(data_dir, self.attr+'_'+self.split+'.mat')
        label_dir = os.path.join(data_dir, 'Y_'+self.split+'.mat')
        self.scene = scio.loadmat(input_dir)[self.split]
        self.label = scio.loadmat(label_dir)[self.split]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        scene = self.scene[index]
        label = self.label[index]
        return np.array(scene, dtype=np.float32), np.array(label, dtype=np.float32)

class loader_2d(Dataset):
    def __init__(self, data_dir, attr_1, attr_2, split):
        self.split = split
        self.attr_1 = attr_1
        self.attr_2 = attr_2

        dir_1 = os.path.join(data_dir, self.attr_1+'_'+self.split+'.mat')
        dir_2 = os.path.join(data_dir, self.attr_2+'_'+self.split+'.mat')
        # label_dir = 'D:/2020_linjiaxin/UCF101/label/Y.mat'
        label_dir = os.path.join(data_dir, 'Y_'+self.split+'.mat')
        # self.data_1 = scio.loadmat(dir_1)[self.attr_1+'_'+self.split]
        # self.data_2 = scio.loadmat(dir_2)[self.attr_2+'_'+self.split]
        # self.label = scio.loadmat(label_dir)['Y_'+self.split]
        self.data_1 = scio.loadmat(dir_1)[self.split]
        self.data_2 = scio.loadmat(dir_2)[self.split]
        self.label = scio.loadmat(label_dir)[self.split]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        data_1 = self.data_1[item]
        data_2 = self.data_2[item]
        label = self.label[item]
        return np.array(data_1, dtype=np.float32), np.array(data_2, dtype=np.float32), np.array(label, dtype=np.float32)

class loader_3d(Dataset):
    def __init__(self, data_dir, attr_1, attr_2, attr_3, split):
        self.split = split
        self.attr_1 = attr_1
        self.attr_2 = attr_2
        self.attr_3 = attr_3

        dir_1 = os.path.join(data_dir, self.attr_1+'_'+self.split+'.mat')
        dir_2 = os.path.join(data_dir, self.attr_2+'_'+self.split+'.mat')
        dir_3 = os.path.join(data_dir, self.attr_3+'_'+self.split+'.mat')
        label_dir = os.path.join(data_dir, 'Y_'+self.split+'.mat')
        self.data_1 = scio.loadmat(dir_1)[self.split]
        self.data_2 = scio.loadmat(dir_2)[self.split]
        self.data_3 = scio.loadmat(dir_3)[self.split]
        self.label = scio.loadmat(label_dir)[self.split]

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, item):
        data_1 = self.data_1[item]
        data_2 = self.data_2[item]
        data_3 = self.data_3[item]
        label = self.label[item]
        return np.array(data_1, dtype=np.float32), np.array(data_2, dtype=np.float32), np.array(data_3, dtype=np.float32), np.array(label, dtype=np.float32)

