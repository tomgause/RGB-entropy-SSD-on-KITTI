import os
import collections
import json
import torch
import torchvision
import numpy as np
import scipy.misc as m
import scipy.io as io
from glob import glob
import os.path
import re
import random
import cv2
from torch.utils import data
import pathlib

KITTI_CLASSES= [
    'BG','Car','Van','Truck',
    'Pedestrian','Person_sitting',
    'Cyclist','Tram','Misc','DontCare'
    ]

class Class_to_ind(object):
    def __init__(self,binary,binary_item):
        self.binary=binary
        self.binary_item=binary_item
        self.classes=KITTI_CLASSES

    def __call__(self, name):
        if not name in self.classes:
            raise ValueError('No such class name : {}'.format(name))
        else:
            if self.binary:
                if name==self.binary_item:
                    return True
                else:
                    return False
            else:
                return self.classes.index(name)
# def get_data_path(name):
#     js = open('config.json').read()
#     data = json.loads(js)
#     return data[name]['data_path']

class AnnotationTransform_kitti(object):
    '''
    Transform Kitti detection labeling type to norm type:
    source: Car 0.00 0 1.55 614.24 181.78 727.31 284.77 1.57 1.73 4.15 1.00 1.75 13.22 1.62
    target: [xmin,ymin,xmax,ymax,label_ind]

    levels=['easy','medium']
    '''
    def __init__(self,class_to_ind=Class_to_ind(True,'Car'),levels=['easy','medium','hard']):
        self.class_to_ind=class_to_ind
        self.levels=levels if isinstance(levels,list) else [levels]

    def __call__(self,target_lines,width,height):
        res=list()
        for line in target_lines:
            xmin,ymin,xmax,ymax=tuple(line.strip().split(' ')[4:8])
            bnd_box=[xmin,ymin,xmax,ymax]
            new_bnd_box=list()
            for i,pt in enumerate(range(4)):
                cur_pt=float(bnd_box[i])
                cur_pt = cur_pt / width if i % 2 == 0 else cur_pt / height
                new_bnd_box.append(cur_pt)
            label_idx=self.class_to_ind(line.split(' ')[0])
            new_bnd_box.append(label_idx)
            res.append(new_bnd_box)
        return res

class KittiLoader(data.Dataset):
    def __init__(self, root, split="training",
                 img_size= (384, 1280)
                 , transforms=None,target_transform=None, train_split=(-1,-1)):
        self.root = root
        self.split = split
        self.target_transform = target_transform
        self.train_split = train_split
        self.n_classes = 11
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([123,117,104])
        self.files = collections.defaultdict(list)
        self.labels = collections.defaultdict(list)
        self.ids = collections.defaultdict(list)
        self.transforms = transforms
        self.name='kitti'


        print("Root: ",root)

        for split in ["training", "testing"]:

            # Create list of image files
            file_list = glob(os.path.join(root, split, 'image_2', '*.png'))
            if train_split[0]==-1:
                self.files[split] = file_list
            else:
                self.files[split] = file_list[train_split[0]:train_split[1]]

            # Create list of label files
            label_list = glob(os.path.join(root, split, 'label_2', '*.txt'))
            if train_split[0]==-1:
                self.labels[split] = label_list
            else:
                self.labels[split] = label_list[train_split[0]:train_split[1]]

            # Create list of image ids
            id_list = []
            p = pathlib.Path(root, split, 'label_2')
            for x in p.glob('*.txt'):
                id_list.append(x)
            if train_split[0] == -1:
                self.ids[split] = id_list
            else:
                self.ids[split] = id_list[train_split[0]:train_split[1]]

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        img_name = self.files[self.split][index]
        img_path = img_name

        #img = m.imread(img_path)
        img = cv2.imread(img_path)
        height, width, channels = img.shape
        #img = np.array(img, dtype=np.uint8)

        if self.split != "testing":
            lbl_path = self.labels[self.split][index]
            lbl_lines=open(lbl_path,'r').readlines()
            if self.target_transform is not None:
                target = self.target_transform(lbl_lines, width, height)
        else:
            lbl = None

        # if self.is_transform:
        #     img, lbl = self.transform(img, lbl)

        if self.transforms is not None:
            target = np.array(target)
            img, boxes, labels = self.transforms(img, target[:, :4], target[:, 4])
            #img, lbl = self.transforms(img, lbl)
            img = img[:, :, (2, 1, 0)]
            target = np.hstack((boxes, np.expand_dims(labels, axis=1)))            

        if self.split != "testing":
            #return img, lbl
            return torch.from_numpy(img).permute(2, 0, 1), target, height, width
        else:
            return img

    def pull_label(self, index):
        return self.labels['training'][index]

    def pull_id(self, index):
        return self.ids['training'][index]

    def detection_collate(batch):
        """Custom collate fn for dealing with batches of images that have a different
        number of associated object annotations (bounding boxes).

        Arguments:
            batch: (tuple) A tuple of tensor images and lists of annotations

        Return:
            A tuple containing:
                1) (tensor) batch of images stacked on their 0 dim
                2) (list of tensors) annotations for a given image are stacked on 0 dim
        """
        targets = []
        imgs = []
        for sample in batch:
            imgs.append(sample[0])
            targets.append(torch.FloatTensor(sample[1]))
        return torch.stack(imgs, 0), targets

