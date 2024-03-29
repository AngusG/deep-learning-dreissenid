"""
task_3_train.py -- trains fully-convolutional networks to perform semantic
segmentation on novel mussel dataset.
"""
# general
import os
import os.path as osp
import subprocess
import csv
import time
import argparse
import numpy as np
# ml libs
import torch
from torch import nn

"""Need special transforms (see transforms.py in this repo) for semantic
segmentation so data augmentations with randomness are applied consistently to
the input image and mask"""
#from torchvision import transforms
import transforms as T
import torchvision
from torchvision import datasets
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

# import an off-the-shelf model for now
#from torchvision.models import segmentation as models
from models import segmentation as models

#from unet import UNet
#import pytorch_unet
from torchsummary import summary

# my utils
from task_3_utils import (save_checkpoint,
                          save_amp_checkpoint,
                          evaluate,
                          #evaluate_loss,
                          #evaluate_binary_iou,
                          evaluate_loss_and_iou_torchvision,
                          adjust_learning_rate)

from folder2lmdb import VOCSegmentationLMDB

from apex import amp

if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
        )

    parser = argparse.ArgumentParser()

    # dataset, admin, checkpointing and hw details
    parser.add_argument('--dataroot', help='path to dataset', type=str,
                        default='/scratch/' + os.environ['USER'] + '/cciw/LMDB')
    parser.add_argument('--data_version', help='dataset version according to \
                        https://semver.org/ convention', type=str,
                        default='v120', choices=['v100', 'v101', 'v110', 'v111',
                                                 'v112', 'v120'])
    parser.add_argument('--split', help='training split', type=str,
                        default='train', choices=['train', 'trainval'])
    parser.add_argument('--logdir', help='directory to store checkpoints; \
                        if None, nothing will be saved')
    parser.add_argument("--resume", default="", type=str,
                        help="path to latest checkpoint (default: none)")
    parser.add_argument('--do_print', help="print ongoing training progress",
                        action="store_true")
    parser.add_argument('--gpu', help='physical id of GPU to use')
    parser.add_argument('--seed', help='random seed', type=int, default=1)

    # model arch and training meta-parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet50',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: fcn_resnet50)')
    parser.add_argument('--epochs', help='number of epochs to train for',
                        type=int, default=80)
    parser.add_argument('--drop', help='epoch to first drop the initial \
                        learning rate', type=int, default=30)
    parser.add_argument('--bs', help='SGD mini-batch size',
                        type=int, default=40)
    parser.add_argument('--lr', help='initial learning rate',
                        type=float, default=1e-1)
    parser.add_argument('--wd', help='weight decay regularization',
                        type=float, default=5e-4)
    parser.add_argument('--bilinear', help='bilinear upsampling or transposed \
                        convolution', action="store_true")
    parser.add_argument('--fp16', help='use apex to train with fp16 parameters',
                        action="store_true")
    parser.add_argument('--tag', help='custom tag to ID debug runs')


    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    gitcommit = subprocess.check_output(['git', 'rev-parse', '--short',
                                         'HEAD']).decode('ascii').strip()

    save_path = osp.join(
        args.logdir,
        args.split + '_' + args.data_version,
        args.arch + '/lr%.e/wd%.e/bs%d/ep%d/seed%d/%s/%s' % (args.lr, args.wd,
                                                             args.bs,
                                                             args.epochs,
                                                             args.seed,
                                                             gitcommit, 
                                                             args.tag))
    print('Saving model to ', save_path)

    ckpt_name = args.arch + '_lr%.e_wd%.e_bs%d_ep%d_seed%d' % (args.lr, args.wd, args.bs, args.epochs, args.seed)

    # Logging stats
    result_folder = osp.join(save_path, 'results/')
    if not osp.exists(result_folder):
        os.makedirs(result_folder)

    #model_string = args.arch + '_bs%d' % args.bs + '_wd%.e' % args.wd + '_' + \
    #args.sess + '_' + str(args.seed)
    logname = osp.join(result_folder, ckpt_name + '.csv')

    if not osp.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'lr', 'train loss', 'val loss'])

    """Define data augmentation transformations. Rotate mussels because they do
    not have a specific orientation. Note transforms provided to `transforms`
    argument of VOCSegmentation apply to both input images and masks. The label
    values happen to be 0/1 so they are unaffected by the normalization.
    """
    """
    if 'Lab' in args.dataroot.split('/'):
        RGB_MEAN = (0.2613, 0.2528, 0.2255)  # mean (RGB)
        RGB_STD  = (0.2637, 0.2546, 0.2306)  # std (RGB)
    else:
        RGB_MEAN = (0.2533962, 0.35527486, 0.11992471)
        RGB_STD  = (0.1717031, 0.11212555, 0.08487311)
    """
    RGB_MEAN = (0.5, 0.5, 0.5)
    RGB_STD  = (0.5, 0.5, 0.5)
    #c = np.array(RGB_MEAN)
    c = torch.FloatTensor(RGB_MEAN).to(device)

    train_tform = T.Compose([
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5), # rotate image about y-axis with 50% prob
        T.RandomVerticalFlip(0.5),
        T.RandomRotation([0, 90, 180, 270, 360]), # randomly rotate image in 90 deg intervals
        T.ToTensor(),
        T.Normalize(RGB_MEAN, RGB_STD)
    ])
    test_tform = T.Compose([
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(RGB_MEAN, RGB_STD)])

    # Prepare dataset and dataloader
    trainset = VOCSegmentationLMDB(
        root=osp.join(args.dataroot, args.split + '_' + args.data_version + '.lmdb'),
        download=False, transforms=train_tform)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)

    valset = VOCSegmentationLMDB(
        root=osp.join(args.dataroot, 'val_v101.lmdb'),
        download=False, transforms=test_tform)
    valloader = DataLoader(valset, batch_size=args.bs, shuffle=False)

    """Prepare model
    NB even though there are two classes (i.e. mussel and background),
    num_classes=1 is used such that nn.Sigmoid(pred) = 0 is bkg, and 1 is mussel.

    Could instead use num_classes=2 and nn.CrossEntropyLoss() such that the
    *channel* rather than the *value* encodes the class, but this would
    require a one-hot label format.
    """
    writer = SummaryWriter(save_path, flush_secs=30)
    """
    n_channels=3 for RGB images
    n_classes is the number of probabilities you want to get per pixel
     - For 1 class and background, use n_classes=1
     - For 2 classes, use n_classes=1
     - For N > 2 classes, use n_classes=N
    """
    print("=> creating model '{}'".format(args.arch))
    net = models.__dict__[args.arch](num_classes=1, pretrained=False).to(device)
    print(net)

    #print(summary(net, input_size=(3, 224, 224)))

    # Prepare training procedure
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, weight_decay=args.wd)

    """Note: BCEWithLogitsLoss uses the log-sum-exp trick for numerical
    stability, so this is safer than nn.BCELoss(nn.Sigmoid(pred))
    Todo: implement class weights to penalize model for predicting bkg"""


    if 'Lab' in args.dataroot.split('/'):
        # 9.7662 is the inverse frequency of `mussel` pixels in the lab training set
        pos_weight = torch.FloatTensor([9.7662]).to(device)  # 4.7861 val
    else:
        # pos_weight by inverse frequency of `mussel` pixels
        if args.data_version == 'v101':
            pos_weight = 3.6891

        elif args.data_version == 'v111':
            if args.split == 'train':
                pos_weight = 3.4270 # train
            else:
                pos_weight = 3.6633 # trainval

        elif args.data_version == 'v112':
            if args.split == 'train':
                pos_weight =  3.6021 # train

        elif args.data_version == 'v120':
            if args.split == 'train':
                pos_weight = 3.1849  # train
            else:
                pos_weight = 3.4297 # trainval

        train_pos_weight = torch.FloatTensor([pos_weight]).to(device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=train_pos_weight)

    # 4.2838 for val_v101
    val_pos_weight = torch.FloatTensor([4.2838]).to(device)
    val_loss_fn = nn.BCEWithLogitsLoss(pos_weight=val_pos_weight)

    #loss_fn = nn.BCEWithLogitsLoss() # sigmoid cross entropy
    #         nn.CrossEntropyLoss() # softmax cross entropy

    # need to explicitly apply Sigmoid for IoU
    sig = nn.Sigmoid()

    best_val_loss = 10

    if args.fp16:
        net, optimizer = amp.initialize(net, optimizer, opt_level='O3')

    #save_checkpoint(net, 100, 100, 0, save_path, ckpt_name)
    #save_amp_checkpoint(net, amp, optimizer, 100, 100, 0, save_path, ckpt_name)

    # Optionally resume from existing checkpoint
    if args.resume:
        if osp.isfile(args.resume):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            checkpoint = torch.load(args.resume)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            amp.load_state_dict(checkpoint['amp'])
            start_epoch = checkpoint['epoch'] + 1
            torch.set_rng_state(checkpoint['rng_state'])
            global_step = start_epoch * (len(trainset) // args.bs)
    else:
        start_epoch = 0
        global_step = 0

    # Train
    for epoch in range(start_epoch, args.epochs):

        eval_start_time = time.time()
        net.eval()
        val_loss, val_iou = evaluate_loss_and_iou_torchvision(net, valloader, val_loss_fn, device)
        print('Epoch [%d/%d], val IoU %.4f, val loss %.4f, took %.2f sec, ' % (
            (epoch, args.epochs, val_iou, val_loss, time.time() - eval_start_time)))
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('IoU/val', val_iou, epoch)

        epoch_start_time = time.time()
        train_loss = 0
        net.train()
        for batch, (inputs, targets) in enumerate(trainloader):
            lr = adjust_learning_rate(optimizer, epoch, args.drop, args.lr)

            optimizer.zero_grad() # reset gradients

            """inputs are in NCHW format: N=nb. samples, C=channels, H=height,
            W=width. Do inputs.permute(0, 2, 3, 1) to viz in RGB format."""
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs)['out'] # fprop for torchvision.models.segmentation

            # dataloader outputs targets with shape NHW, but we need NCHW
            batch_loss = loss_fn(pred, targets.unsqueeze(dim=1).float())
            train_loss += batch_loss.item()


            if args.fp16:
                with amp.scale_loss(batch_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                batch_loss.backward() # bprop

            optimizer.step() # update parameters

            if batch % 10 == 0:
                print('Batch [{}/{}], train loss: {:.4f}'
                      .format(batch, len(trainloader), batch_loss.item()))  #, train IoU: {:.4f}'
                writer.add_scalar('Loss/train mini-batch', batch_loss.item(), global_step)

                with torch.no_grad():
                    for n, p in net.named_parameters():
                        if 'weight' in n.split('.'):
                            writer.add_scalar('L2norm/' + n, p.norm(2), global_step)
                        elif 'scale' in n.split('.'):
                            writer.add_scalar('scale/' + n, p.item(), global_step)
                            
                        # add scale here
            global_step += 1

        epoch_time = time.time() - epoch_start_time
        train_loss /= len(trainloader)

        #net.eval()
        #val_loss = evaluate_loss(net, valloader, val_loss_fn, device)
        #val_iou = evaluate_binary_iou(net, valloader, val_loss_fn, device)
        writer.add_scalar('Loss/train', train_loss, epoch + 1)
        
        
        images = inputs[:16].permute(0, 2, 3, 1) * c + c
        images = images.permute(0, 3, 1, 2)
        
        img_grid = torchvision.utils.make_grid(images)
        sig_grid = torchvision.utils.make_grid(sig(pred[:16]))
        lab_grid = torchvision.utils.make_grid(targets[:16].unsqueeze(dim=1).float())
        writer.add_image('images', img_grid, epoch)
        writer.add_image('predictions', sig_grid, epoch)
        writer.add_image('labels', lab_grid, epoch)

        if epoch % 10 == 0:
            #if val_loss < best_val_loss:
            #    best_val_loss = val_loss
            if args.fp16:
                save_amp_checkpoint(net, amp, optimizer, val_loss, train_loss, epoch, save_path, ckpt_name)
            else:
                save_checkpoint(net, val_loss, train_loss, epoch, save_path, ckpt_name)

        '''
        train_eval_start_time = time.time()
        train_iou, train_loss = evaluate(net, trainloader_noshuffle, loss_fn, device)
        train_eval_time = time.time() - train_eval_start_time
        '''
        '''
        writer.add_scalar('Loss/test', val_iou, global_step)
        writer.add_images('images', inputs, global_step)
        if net.n_classes == 1:
            writer.add_images('masks/true', targets, global_step)
            writer.add_images('masks/pred', sig(pred) > 0.5, global_step)
        '''

        #print('Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}, train IoU: {:.4f}, val IoU: {:.4f}, took {:.2f}s'
        #      .format(epoch + 1, args.epochs, train_loss, val_loss, train_iou, val_iou, epoch_time))

        print('Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}, took {:.2f} s'
              .format(epoch + 1, args.epochs, train_loss, val_loss, epoch_time))

        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                [epoch, lr, np.round(train_loss, 4), np.round(val_loss, 4)])
    if args.fp16:
        save_amp_checkpoint(net, amp, optimizer, val_loss, train_loss, epoch, save_path, ckpt_name)
    else:
        save_checkpoint(net, val_loss, train_loss, epoch, save_path, ckpt_name)

    writer.close()
