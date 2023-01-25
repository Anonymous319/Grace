import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms
from sklearn.utils import shuffle
import torch.nn.functional as F

########################################################################################################################

class Pad(object):

    def __init__(self, size, fill=0, padding_mode='constant'):
        self.size = size
        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        img_size = img.size()[1]
        assert ((self.size - img_size) % 2 == 0)
        padding = (self.size - img_size) // 2
        padding = (padding, padding, padding, padding)
        return F.pad(img, padding, self.padding_mode, self.fill)


class Convert2RGB(object):

    def __init__(self, num_channel):
        self.num_channel = num_channel

    def __call__(self, img):
        img_channel = img.size()[0]
        img = torch.cat([img] * (self.num_channel - img_channel + 1), 0)
        return img

def get_transform():
    transform = transforms.Compose([transforms.ToTensor(),
                                    Pad(32),
                                    Convert2RGB(3),
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    return transform

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 3):
    path = '../data/mnist_usps_svhn/'

    MNIST_train = datasets.MNIST(root=os.path.join(path, "MNIST"),
                                       transform=transform, train=True, download=False)

    MNIST_test = datasets.MNIST(root=os.path.join(path, "MNIST"),
                                 transform=transform,
                                 train=False,
                                 download=False)

    SVHN_train = datasets.SVHN(root=os.path.join(path, "SVHN"),
                                    transform=transform,
                                      split='train',
                                      download=False)
    SVHN_test = datasets.SVHN(root=os.path.join(path, "SVHN"),
                               transform=transform,
                               split='test',
                               download=False)

    USPS_train = datasets.USPS(root=os.path.join(path, "USPS"),
                                      transform=transform,
                                      train=True,
                                      download=False)
    USPS_test = datasets.USPS(root=os.path.join(path, "USPS"),
                               transform=transform,
                               train=False,
                               download=False)
    data = {}
    taskname = ["SVHN","MNIST","USPS"]
    for i in range(tasknum):
        print(i, end=',')
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = taskname[i]
        data[i]['ncla'] = 10
        for s in ['train', 'test']:
            if i==0:
                if s == 'train':
                    arr = transform(SVHN_train.data,i)
                    label = torch.LongTensor(SVHN_train.labels)
                else:
                    arr = transform(SVHN_test.data,i)
                    label = torch.LongTensor(SVHN_test.labels)
            elif i==1:
                if s == 'train':
                    arr = transform(MNIST_train.train_data,i)
                    label = torch.LongTensor(MNIST_train.train_labels)
                else:
                    arr = transform(MNIST_test.test_data, i)
                    label = torch.LongTensor(MNIST_test.test_labels)
            else:
                if s == 'train':
                    arr = transform(USPS_train.data,i)
                    label = torch.LongTensor(USPS_train.targets)
                else:
                    arr = transform(USPS_test.data, i)
                    label = torch.LongTensor(USPS_test.targets)
            data[i][s]={}
            data[i][s]['x'] = arr
            data[i][s]['y'] = label


    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    taskcla = []
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n

    size = [3, 32, 32]
    return data, taskcla,size

def transform(d,t):
    numdata=d.shape[0]
    result = torch.FloatTensor(numdata, 3,32,32)
    transform = get_transform()
    for i in range(numdata):
        if t ==0:
            img = Image.fromarray(np.transpose(d[i], (1, 2, 0)))
        elif t ==1:
            img = Image.fromarray(d[i].numpy(), mode='L')
        else:
            img = Image.fromarray(d[i], mode='L')
        result[i] = transform(img)
    return result

########################################################################################################################
