# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import argparse
from torch.autograd import Variable
import torch.utils.data as data
from data import AnnotationTransform, detection_collate, KITTIroot, kitti_single_root, kitti_single_root_2
from data import KittiLoader, AnnotationTransform_kitti,Class_to_ind, KITTI_CLASSES,voc0712

import numpy as np
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
from IPython import embed
from log import log
import time

import matplotlib.pyplot as plt

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")
torch.autograd.set_detect_anomaly(True)
parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--dim', default=384, type=int, help='Size of the input image, only support 300 or 512')
parser.add_argument('-d', '--dataset', default='kitti',help='VOC or KITTI dataset')

parser.add_argument('--basenet', default='vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for training')
parser.add_argument('--resume', default=None, type=str, help='Resume from checkpoint')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--iterations', default=100000000, type=int, help='Number of training iterations')
parser.add_argument('--cuda', default=False, type=str2bool, help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--visdom', default=True, type=bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', default=True, type=bool, help='Send images to visdom for loss visualization')
parser.add_argument('--save_folder', default='weights/', help='Location to save checkpoint models')
parser.add_argument('--data_root', default=KITTIroot, type=str, help='Location of kitti root directory')
parser.add_argument('--train_split', nargs='+', default=[-1, -1], type=int, help='Split training set using tuple, default is all')
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
# train_sets = 'train'
means = (123, 117, 104)  # KITTI
if args.dataset=='VOC':
    num_classes = len(voc0712.VOC_CLASSES) + 1
elif args.dataset=='kitti':
    num_classes = 11
accum_batch_size = 2
iter_size = accum_batch_size / args.batch_size
stepvalues = (60000, 80000, 100000)
start_iter = 0

if args.visdom:
    import visdom
    viz = visdom.Visdom(port=8100)

ssd_net = build_ssd('train', args.dim, num_classes)
net = ssd_net
net = net.float()
if args.cuda:
    net = torch.nn.DataParallel(ssd_net)
    cudnn.benchmark = True

if args.resume:
    log.l.info('Resuming training, loading {}...'.format(args.resume))
    ssd_net.load_weights(args.resume)
    start_iter = int(args.resume.split('/')[-1].split('.')[0].split('_')[-1])###########
else:
#    vgg_weights = torch.load(args.save_folder + args.basenet)
#    log.l.info('Loading base network...')
#    ssd_net.vgg.load_state_dict(vgg_weights)
    start_iter = 0

if args.cuda:
    net = net.cuda()
    torch.cuda.empty_cache()

def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if not args.resume:
    log.l.info('Initializing weights...')
    # initialize newly added layers' weights with xavier method
    ssd_net.extras.apply(weights_init)
    ssd_net.loc.apply(weights_init)
    ssd_net.conf.apply(weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=args.momentum, weight_decay=args.weight_decay)
criterion = MultiBoxLoss(num_classes, args.dim, 0.5, True, 0, True, 3, 0.5, False, args.cuda)

def DatasetSync(dataset='VOC',split='training'):


    if dataset=='VOC':
        #DataRoot=os.path.join(args.data_root,'VOCdevkit')
        DataRoot=args.data_root
        dataset = voc0712.VOCDetection(DataRoot, train_sets, SSDAugmentation(
        args.dim, means), AnnotationTransform())
    elif dataset=='kitti':
        #DataRoot=os.path.join(args.data_root,'kitti')
        dataset = KittiLoader(args.data_root, split=split,img_size=(1280, 384),
                  transforms=SSDAugmentation((1280, 384),means),
                  target_transform=AnnotationTransform_kitti(),
                  train_split=tuple(args.train_split))

    return dataset

def train():
    torch.cuda.empty_cache()
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0
    log.l.info('Loading Dataset...')

    # dataset = VOCDetection(args.voc_root, train_sets, SSDAugmentation(
    #     args.dim, means), AnnotationTransform())
    dataset=DatasetSync(dataset=args.dataset,split='training')


    epoch_size = len(dataset) // args.batch_size
    log.l.info('Training SSD on {}'.format(dataset.name))
    step_index = 0
    if args.visdom:
        # initialize visdom loss plot
        lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Iteration',
                ylabel='Loss',
                title='Current SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
        epoch_lot = viz.line(
            X=torch.zeros((1,)).cpu(),
            Y=torch.zeros((1, 3)).cpu(),
            opts=dict(
                xlabel='Epoch',
                ylabel='Loss',
                title='Epoch SSD Training Loss',
                legend=['Loc Loss', 'Conf Loss', 'Loss']
            )
        )
    batch_iterator = None

    data_loader = data.DataLoader(dataset, args.batch_size, num_workers=args.num_workers,
                                 shuffle=True, collate_fn=detection_collate, pin_memory=True)


    #assign learning rate
    lr=args.lr

    #training
    for iteration in range(start_iter, args.iterations + 1):
        if (not batch_iterator) or (iteration % epoch_size == 0):
            # create batch iterator
            batch_iterator = iter(data_loader)
        if (iteration % epoch_size == 0) and (iteration != 0): #iteration in stepvalues:
            step_index += 1
            lr=adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)
            if args.visdom:
                viz.line(
                    X=torch.ones((1, 3)).cpu() * epoch,
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu() / epoch_size,
                    win=epoch_lot,
                    update='append'
                )
            # reset epoch loss counters
            loc_loss = 0
            conf_loss = 0
            epoch += 1

        # load train data
        images, targets = next(batch_iterator)
        #embed()
        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(anno) for anno in targets]
        # forward
        t0 = time.time()
        out = net(images)

        #print('TARGET SIZE: ', len(targets[0]))
        #print('OUT SIZE: ', len(out[2]))

        # backprop
        # loss_l and loss_c .data[0] replaced by .item()

        optimizer.zero_grad()

        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loss.backward()

        # GRADIENT CLIPPING
        #clip_param = 5
        #torch.nn.utils.clip_grad_norm_(net.parameters(), clip_param)

        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item() #loss_l.data[0]
        conf_loss += loss_c.item() #loss_c.data[0]
        if iteration % 1 == 0:
            log.l.info('''
                Timer: {:.5f} sec.\t LR: {}.\t Iter: {}.\t Loss_l: {:.5f}.\t Loss_c: {:.5f}.
                '''.format((t1-t0),lr,iteration,loss_l.item(),loss_c.item()))
            if args.visdom and args.send_images_to_visdom and ((iteration==0) or ((loss_l.item() + loss_c.item()) >= 60)):
                random_batch_index = np.random.randint(images.size(0))
                viz.image(images.data[random_batch_index].cpu().numpy())
                viz.text("Index: ", str(random_batch_index))
        if args.visdom:
            viz.line(
                X=torch.ones((1, 3)).cpu() * iteration,
                Y=torch.Tensor([loss_l.item(), loss_c.item(),
                    loss_l.item() + loss_c.item()]).unsqueeze(0).cpu(),
                win=lot,
                update='append'
            )
            # hacky fencepost solution for 0th epoch plot
            if iteration == 0:
                viz.line(
                    X=torch.zeros((1, 3)).cpu(),
                    Y=torch.Tensor([loc_loss, conf_loss,
                        loc_loss + conf_loss]).unsqueeze(0).cpu(),
                    win=epoch_lot,
                    update=True
                )
        if iteration % 250 == 0:
            log.l.info('Saving state, iter: {}'.format(iteration))
            torch.save(ssd_net.state_dict(), 'weights/ssd' + str(args.dim) + '_0712_' +
                       repr(iteration) + '.pth')
    torch.save(ssd_net.state_dict(), args.save_folder + 'ssd_' + str(args.dim) + '.pth')
    # if iteration == args.iterations:
    #     epochs = range(1, len(conf_loss) + 1)
    #     plt.plot(epochs, loc_loss, 'b--')
    #     plt.plot(epochs, conf_loss, 'r--')
    #     plt.legend(['Location Loss', 'Confidence Loss'])
    #     plt.xlabel('Epoch')
    #     plt.ylabel('Loss')
    #     plt.show()


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5)
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()
