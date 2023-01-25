import os, sys
import random
from PIL import Image
import numpy as np
import torch
from torchvision import datasets, transforms

def get(seed=0, fixed_order=False, pc_valid=0, tasknum = 10):
    dataloadfile = "savedata/rmnistrot_seed{}.npz".format(seed)
    data = {}
    taskcla = []
    size = [1, 28, 28]
    mean = torch.Tensor([0.1307])
    std = torch.Tensor([0.3081])
    datadir = "../data/mnist_usps_svhn/MNIST/MNIST/"
    dat = {}
    dat['train'] = datasets.MNIST(datadir, train=True, download=False)
    dat['test'] = datasets.MNIST(datadir, train=False, download=False)

    if os.path.exists(dataloadfile):
        rotload = np.load(dataloadfile)
        allrot = rotload["allrot"]
    else:
        allrot = []
    print(allrot)

    for i in range(tasknum):
        sys.stdout.flush()
        data[i] = {}
        data[i]['name'] = 'rotate_mnist-{:d}'.format(i)
        data[i]['ncla'] = 10

        if os.path.exists(dataloadfile):
            rot = allrot[i]
        else:
            min_rot = 1.0 * i / tasknum * 180
            max_rot = 1.0 * (i + 1) / tasknum * 180
            rot = random.random() * (max_rot - min_rot) + min_rot
            allrot.append(rot)
        print(i,rot, end=',')



        for s in ['train', 'test']:
            if s == 'train':
                arr =rotate_dataset(dat[s].train_data, rot)
                label =torch.LongTensor(dat[s].train_labels)
            else:
                arr = rotate_dataset(dat[s].test_data,rot)
                label = torch.LongTensor(dat[s].test_labels)

            data[i][s]={}
            data[i][s]['x'] = arr
            data[i][s]['y'] = label

    for t in range(tasknum):
        data[t]['valid'] = {}
        data[t]['valid']['x'] = data[t]['train']['x'].clone()
        data[t]['valid']['y'] = data[t]['train']['y'].clone()

    # Others
    n = 0
    for t in range(tasknum):
        taskcla.append((t, data[t]['ncla']))
        n += data[t]['ncla']
    data['ncla'] = n
    if not os.path.exists(dataloadfile):
        np.savez(dataloadfile, allrot=allrot)
    return data, taskcla, size

def rotate_dataset(d, rotation):
    result = torch.FloatTensor(d.size(0), 1,28,28)
    mean = torch.Tensor([0.5])
    std = torch.Tensor([0.5])
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)])
    #tensor = transforms.ToTensor()

    for i in range(d.size(0)):
        img = Image.fromarray(d[i].numpy(), mode='L')
        result[i] = transform(img.rotate(rotation)).view(1,28,28)
        #result[i] = tensor(img.rotate(rotation)).view(784)
    return result
########################################################################################################################
