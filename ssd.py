import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import v as cfg
import os
from IPython import embed


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
        self.priors = Variable(self.priorbox.forward(), volatile=True)

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
            x: input image or batch of images. Shape: [batch,3,300,300]. or [batch,3,512,512]

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

        # apply vgg up to conv4_3 relu
        for k in range(39):
            x = self.vgg[k](x)
        sources.append(x)
        # apply vgg up to conv5_3 relu
        for k in range(39, 51):
            x = self.vgg[k](x)
        sources.append(x)
        # apply extra layers up to conv7relu
        for k in range(6):
            x = self.extras[k](x)
        sources.append(x)
        # apply extra layers up to conv8_2 relu
        for k in range(6, 12):
            x = self.extras[k](x)
        sources.append(x)
        # apply extra layers up to conv9_2 relu
        for k in range(12, 18):
            x = self.extras[k](x)
        sources.append(x)
        # apply extra layers up to conv10_2 relu
        for k in range(18, 24):
            x = self.extras[k](x)
        sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
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


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(i):
    in_channels = i
    layers = [

        # TODO: Adjust network architecture (obviously)

        # Conv1_1, Conv1_2, Pool1
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(64), nn.ReLU(),

        # Conv2_1, Conv2_2, Pool2
        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(128), nn.ReLU(),

        # Conv3_1, Conv3_2, Conv3_3, Pool3
        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.MaxPool2d(kernel_size=2), nn.BatchNorm2d(256), nn.ReLU(),

        # Conv4_1, Conv4_2, Conv4_3, Conv4_4 FIRST SOURCE, Pool4
        nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.MaxPool2d(kernel_size=1), nn.BatchNorm2d(512), nn.ReLU(),

        # Conv5_1, Conv5_2, Conv5_3 SECOND SOURCE
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU()
    ]
    return layers

def add_extras():
    layers = [
        nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(1024), nn.ReLU(),

        # Conv7, THIRD SOURCE
        nn.Conv2d(1024, 1024, kernel_size=1), nn.BatchNorm2d(1024), nn.ReLU(),

        # Conv8_1, Conv8_2 FOURTH SOURCE
        nn.Conv2d(1024, 256, kernel_size=1), nn.BatchNorm2d(256), nn.ReLU(),
        nn.Conv2d(256, 512, kernel_size=3, padding=0), nn.BatchNorm2d(512), nn.ReLU(),

        # Conv9_1, Conv9_2 FIFTH SOURCE
        nn.Conv2d(512, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(256), nn.ReLU(),

        # Conv10_1, Conv10_2 SIXTH SOURCE
        nn.Conv2d(256, 128, kernel_size=1), nn.BatchNorm2d(128), nn.ReLU(),
        nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=0), nn.BatchNorm2d(256), nn.ReLU()
    ]
    return layers

def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [38, -1]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v-2].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v-2].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[3::6], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)

# TODO: determine ideal # of boxes for kitti
mbox = [4, 6, 6, 6, 6, 4] # of boxes per feature map location

def build_ssd(phase, size=192, num_classes=11):
    print("Building SSD...")
    if phase != "test" and phase != "train":
        print("Error: Phase not recognized")
        return
    if size != 192:
        print("Error: Only supports SSD192x624.")
        return

    return SSD(phase, *multibox(vgg(3), add_extras(), mbox, num_classes), num_classes)
