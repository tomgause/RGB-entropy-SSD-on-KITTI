from __future__ import print_function
import sys
import os
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
from data import KITTIroot, KITTI_CLASSES as labelmap
from PIL import Image
from data import AnnotationTransform_kitti, BaseTransform, KittiLoader
import torch.utils.data as data
from ssd import build_ssd
from log import log
from utils.augmentations import SSDAugmentation

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--trained_model', default='weights/ssd384_0712_45000.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.5, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--data_root', default=KITTIroot, help='Location of data root directory')
parser.add_argument('--test_split', nargs='+', default=[0, 10], type=int, help='Range of images to test, default 0 10')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(save_folder, net, cuda, testset, transform, thresh):
    # dump predictions and assoc. ground truth to text file for now
    filename = save_folder+'test1.txt'
    num_images = len(testset)
    for i in range(num_images):
        log.l.info('Testing image {:d}/{:d}....'.format(i+1, num_images))
        img, annotation = testset[i]
        img_id = testset.pull_id(i)
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with open(filename, mode='a') as f:
            f.write('\nGROUND TRUTH FOR: '+img_id+'\n')
            for box in annotation:
                f.write('label: '+' || '.join(str(b) for b in box)+'\n')
        if cuda:
            x = x.cuda()

        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        for i in range(0, detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= args.visual_threshold:
                if pred_num == 0:
                    with open(filename, mode='a') as f:
                        f.write('PREDICTIONS: '+'\n')
                score = detections[0, i, j, 0]
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_num += 1
                with open(filename, mode='a') as f:
                    f.write(str(pred_num)+' label: '+label_name+' score: ' +
                            str(score) + ' '+' || '.join(str(c) for c in coords) + '\n')
                j += 1


if __name__ == '__main__':
    # load net
    num_classes = 11 # COCO
    net = build_ssd('test', 384, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    log.l.info('Finished loading model!')
    # load data
    #testset = VOCDetection(args.voc_root, [('2007', 'test')], None, AnnotationTransform())
    testset = KittiLoader(args.data_root, split="testing" ,img_size=(1280, 384),
                  transforms=None,
                  target_transform=AnnotationTransform_kitti(levels=['easy', 'medium']),
                  train_split=(args.test_split[0]+6500,args.test_split[1]+6500))
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, testset,
             BaseTransform((1280, 384), (123, 117, 104)),
             thresh=args.visual_threshold)
