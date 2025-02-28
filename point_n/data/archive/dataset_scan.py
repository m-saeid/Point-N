"""
ScanObjectNN download: http://103.24.77.34/scanobjectnn/h5_files.zip
"""

import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def load_scanobjectnn_data(split, partition):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    if split == 1:
        h5_name = BASE_DIR + '/../datasets/h5_files/main_split/' + partition + '_objectdataset.h5'
    elif split == 2:
        h5_name = BASE_DIR + '/../datasets/h5_files/main_split_nobg/' + partition + '_objectdataset.h5'
    elif split == 3:
        h5_name = BASE_DIR + '/../datasets/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'

    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label





class ScanObjectNN(Dataset):
    def __init__(self, num_points, split=3, partition='training'):
        self.data, self.label = load_scanobjectnn_data(split, partition)
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            # pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN(1024)
    test = ScanObjectNN(1024, partition = 'test')
    
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ScanObjectNN(partition='training', num_points=1024)
    test_set = ScanObjectNN(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
    
    
    