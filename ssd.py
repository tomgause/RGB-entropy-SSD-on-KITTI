import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v as cfg
import os
import math
from IPython import embed
from skimage.filters.rank import entropy
from skimage.morphology import disk
from sys import getsizeof
import numpy as np

class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        base: VGG16 layers for input
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = cfg
        self.priorbox = PriorBox(self.cfg)
        with torch.no_grad():
            self.priors = self.priorbox.forward()

        # SSD network
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if self.phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            #self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)
            self.detect = Detect()

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,192,624]

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch,num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors,4]
                    3: priorbox layers, Shape: [num_priors,4]
        """
        sources = list()
        loc = list()
        conf = list()
        # x=x.view(3,-1,192)
        # apply vgg up to conv4_3 relu
        # entropy_layers=[9,22,38,54,67]
        # x0 = x.reshape(3,1280,384).numpy().astype('float64')
        # x0=(x0-x0.min())/(x0.max()-x0.min())
        # iterable = iter(range(58))
        # for k in iterable:#range(39):#
        #     if k in entropy_layers:#in entropy_layers:
        #         # sources.append(x)
        #         xent = self.vgg[k](x0).astype('double')
        #         xconv = self.vgg[k + 1](torch.tensor(xent).reshape(1,1,1280,384).float())
        #         # sources.append(xconv)
        #         sig=self.vgg[k+2](xconv)
        #         x = self.vgg[k + 3](sig, xent, x).float()
        #
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else: x = self.vgg[k](x)
        for k in range(39):
            x = self.vgg[k](x)
        sources.append(x)
        # apply vgg up to conv5_3 relu
        # iterable = iter(range(58,71))
        # for k in iterable:#(39, 50):#
        #     if k in entropy_layers:
        #         # sources.append(x)
        #         xent = self.vgg[k](x0).astype('double')
        #         xconv = self.vgg[k + 1](torch.tensor(xent).reshape(1,1,1280,384).float())
        #         sig=self.vgg[k+2](xconv)
        #         x = self.vgg[k + 3](sig, xent, x).float()
        #         if(k>66): break
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else: x = self.vgg[k](x)
        for k in range(39, 51):
            x = self.vgg[k](x)
        sources.append(x)
        # apply extra layers up to conv7relu
        for k in range(6):#(15)
            x = self.extras[k](x)
        sources.append(x)
        # entropy_layers=[3,10,20,30,40]
        # iterable = iter(range(14))
        # for k in iterable:
        #     if k in entropy_layers:
        #         xent = self.extras[k](x0).astype('double')
        #         xconv = self.extras[k + 1](torch.tensor(xent).reshape(1,1,624,192).float())
        #         sig=self.extras[k+2](xconv)
        #         x = self.extras[k + 3](sig, xent, x).float()
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else: x = self.extras[k](x)
        # sources.append(x)
        # apply extra layers up to conv8_2 relu
        for k in range(6, 12):
            x = self.extras[k](x)
        sources.append(x)
        # iterable = iter(range(14,24))
        # for k in iterable:
        #     if k in entropy_layers:
        #         xent = self.extras[k](x0).astype('double')
        #         xconv = self.extras[k + 1](torch.tensor(xent).reshape(1, 1, 624, 192).float())
        #         sig = self.extras[k + 2](xconv)
        #         x = self.extras[k + 3](sig, xent, x).float()
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else: x = self.extras[k](x)
        # sources.append(x)
        # apply extra layers up to conv9_2 relu
        for k in range(12, 18):
            x = self.extras[k](x)
        sources.append(x)
        # iterable = iter(range(24,34))
        # for k in iterable:
        #     if k in entropy_layers:
        #         xent = self.extras[k](x0).astype('double')
        #         xconv = self.extras[k + 1](torch.tensor(xent).reshape(1, 1, 624, 192).float())
        #         sig = self.extras[k + 2](xconv)
        #         x = self.extras[k + 3](sig, xent, x).float()
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else: x = self.extras[k](x)
        # sources.append(x)
        # apply extra layers up to conv10_2 relu
        # iterable = iter(range(34,44))
        # for k in iterable:
        #     if k in entropy_layers:
        #         xent = self.extras[k](x0).astype('double')
        #         xconv = self.extras[k + 1](torch.tensor(xent).reshape(1, 1, 624, 192).float())
        #         sig = self.extras[k + 2](xconv)
        #         x = self.extras[k + 3](sig, xent, x).float()
        #         [iterable.__next__() for x in range(3)]
        #         continue
        #     else:
        #         x = self.extras[k](x)
        # sources.append(x)
        for k in range(18, 24):
            x = self.extras[k](x)
        sources.append(x)

        # apply multibox head to source layers         ????????????????????????????????????????????????????????????
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)

        print(loc.size())
        print(conf.size())

        if self.phase == "test":
            output = self.detect.apply(self.num_classes, 0, 200, 0.01, 0.45,
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(-1, self.num_classes)),  # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )

        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file, map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Only .pth and .pkl files supported.')

# #Entropy function
# class entropy_func(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # size_in=1
#         # size_out=4
#         # self.size_in, self.size_out = size_in, size_out
#         # weights = torch.Tensor(size_out, size_in)
#         # self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
#         # bias = torch.Tensor(size_out)
#         # self.bias = nn.Parameter(bias)
#         #
#         # # initialize weights and biases
#         # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
#         # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
#         # bound = 1 / math.sqrt(fan_in)
#         # nn.init.uniform_(self.bias, -bound, bound)  # bias init
#     def forward(self, x0):
#         return (entropy(x0[0], disk(5))+entropy(x0[1], disk(5))+entropy(x0[2], disk(5)))/3
# class entropy_layer(nn.Module):
#     """ Custom Linear layer but mimics a standard linear layer """

    #def __init__(self):
        #super().__init__()
        # size_in=1
        # size_out=4
        # self.size_in, self.size_out = size_in, size_out
        # weights = torch.Tensor(size_out, size_in)
        # self.weights = nn.Parameter(weights)  # nn.Parameter is a Tensor that's a module parameter.
        # bias = torch.Tensor(size_out)
        # self.bias = nn.Parameter(bias)
        #
        # # initialize weights and biases
        # nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))  # weight init
        # fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weights)
        # bound = 1 / math.sqrt(fan_in)
        # nn.init.uniform_(self.bias, -bound, bound)  # bias init

    # def forward(self, sig,xent,x):
    #     self.weights=sig
    #     self.bias=xent
    #     weight=self.weights.squeeze().detach().numpy()
    #     xbackup=x
    #     x=x.squeeze().reshape(-1,self.bias.shape[0],self.bias.shape[1]).detach().numpy()
    #     res=[]
    #     for i in range(x.shape[0]):
    #         # w_times_x = torch.mm(x[i], torch.tensor(weight).t())
    #         w_times_x=np.multiply(x[i],weight)
    #         sumation=w_times_x+ self.bias
    #         res.append(sumation)
    #     # xsize=x.squeeze().reshape(1,-1).size()[1]
    #     # wsize=weight.reshape(1,-1).size()[1]
    #     # gcdv=math.gcd(xsize,wsize)
    #     # weight=weight.reshape(-1,gcdv)
    #     # x=x.reshape(-1,gcdv)
    #     return torch.tensor(res).resize_as(xbackup)  # w times x + b
# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(i):
    in_channels = i
    # x0=None
    # ent=entropy()
    layers = [

        # TODO: Adjust network architecture (obviously)

        # Conv1_1, Conv1_2, Pool1
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        # EXPERIMENT WITH ADAPTIVE MAX POOL
        #nn.AdaptiveMaxPool2d((1, 64, 192, 640)), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(),
        #entropy_func(),nn.Conv2d(1,1,kernel_size=3,padding=1),nn.Sigmoid(),
        #entropy_layer(),
        # Conv2_1, Conv2_2, Pool2
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(128), nn.ReLU(),
        #entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        #entropy_layer(),
        # Conv3_1, Conv3_2, Conv3_3, Pool3
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(256), nn.ReLU(),
        #entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        #entropy_layer(),
        # Conv4_1, Conv4_2, Conv4_3, Conv4_4 FIRST SOURCE, Pool4
        nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.MaxPool2d(kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(),
        #entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        #entropy_layer(),
        # Conv5_1, Conv5_2, Conv5_3 SECOND SOURCE
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        #entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        #entropy_layer(),
    ]
    return layers

def add_extras():
    layers = [
        nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),
        # entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        # entropy_layer(),
        # Conv7, THIRD SOURCE
        nn.Conv2d(1024, 1024, kernel_size=1), nn.BatchNorm2d(1024), nn.ReLU(),
        # entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        # entropy_layer(),
        # Conv8_1, Conv8_2 FOURTH SOURCE
        nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        # entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        # entropy_layer(),
        # Conv9_1, Conv9_2 FIFTH SOURCE
        nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        # entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        # entropy_layer(),
        # Conv10_1, Conv10_2 SIXTH SOURCE
        nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(256), nn.ReLU()
        # ,entropy_func(), nn.Conv2d(1, 1, kernel_size=3,padding=1),nn.Sigmoid(),
        # entropy_layer()
    ]
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [38, -1]
    #vgg_source = [57,-1]#[38, -1]#
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v-2].out_channels,#-9
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v-2].out_channels, #-9
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[3::6], 2):#[7::10]
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

# TODO: determine ideal # of boxes for kitti
mbox = [6, 6, 6, 6, 6, 6] # of boxes per feature map location

def build_ssd(phase, size=384, num_classes=11):
    #print("Building SSD...")
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 384:
        print("Error: Only supports SSD1280x384.")
        return

    return SSD(phase, *multibox(vgg(3), add_extras(), mbox, num_classes), num_classes)
