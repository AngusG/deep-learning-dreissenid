"""
task_3_train.py -- trains fully-convolutional networks to perform semantic
segmentation on novel mussel dataset.
"""
# general
import os
import csv
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
from torchvision import datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# import an off-the-shelf model for now
#from torchvision.models import segmentation as models

from unet import UNet
#import pytorch_unet
from torchsummary import summary

# my utils
from task_3_utils import (save_checkpoint,
                          eval_binary_iou,
                          adjust_learning_rate)

if __name__ == '__main__':

    '''
    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
        )
    '''

    parser = argparse.ArgumentParser()

    # dataset, admin, checkpointing and hw details
    parser.add_argument('--dataroot', help='path to dataset',
                        type=str, default='/scratch/ssd/' + os.environ['USER'] + '/cciw/')
    parser.add_argument('--logdir', help='directory to store checkpoints; \
                        if None, nothing will be saved')
    parser.add_argument("--resume", default="", type=str,
                        help="path to latest checkpoint (default: none)")
    parser.add_argument('--do_print', help="print ongoing training progress",
                        action="store_true")
    parser.add_argument('--gpu', help='physical id of GPU to use')
    parser.add_argument('--seed', help='random seed', type=int, default=1)
    parser.add_argument('--sess', default='def', type=str, help='session id')

    # model arch and training meta-parameters
    parser.add_argument('-a', '--arch', metavar='ARCH', default='unet_bn')
                        #choices=model_names, help='model architecture: ' +
                        #' | '.join(model_names) + ' (default: fcn_resnet50)')
    parser.add_argument('--epochs', help='number of epochs to train for',
                        type=int, default=100)
    parser.add_argument('--drop', help='epoch to first drop the initial \
                        learning rate', type=int, default=50)
    parser.add_argument('--bs', help='SGD mini-batch size',
                        type=int, default=32)
    parser.add_argument('--lr', help='initial learning rate',
                        type=float, default=0.1)
    parser.add_argument('--wd', help='weight decay regularization',
                        type=float, default=5e-4)
    parser.add_argument('--bilinear', help='bilinear upsampling or transposed \
                        convolution', action="store_true")

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        if torch.cuda.is_available():
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch.manual_seed(args.seed)

    # Logging stats
    result_folder = os.path.join(args.logdir, 'results/')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    model_string = args.arch + '_bs%d' % args.bs + '_wd%.e' % args.wd + '_' + \
    args.sess + '_' + str(args.seed)
    logname = os.path.join(result_folder, model_string + '.csv')

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'lr', 'train loss', 'train iou'])

    """Define data augmentation transformations. Rotate mussels because they do
    not have a specific orientation. Note transforms provided to `transforms`
    argument of VOCSegmentation apply to both input images and masks. The label
    values happen to be 0/1 so they are unaffected by the normalization.
    """
    train_tform = T.Compose([
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5), # rotate image about y-axis with 50% prob
        T.RandomVerticalFlip(0.5),
        T.ToTensor(),
        T.Normalize((0.2613, 0.2528, 0.2255), # mean (RGB)
                    (0.2637, 0.2546, 0.2306)) # std (RGB)
    ])

    test_tform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.2613, 0.2528, 0.2255), # mean (RGB)
                    (0.2637, 0.2546, 0.2306))
        ])

    # Prepare dataset and dataloader
    trainset = datasets.VOCSegmentation(
        root=args.dataroot, year='2012', image_set='train',
        download=False, transforms=train_tform)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)

    trainset_noshuffle = datasets.VOCSegmentation(
        root=args.dataroot, year='2012', image_set='train',
        download=False, transforms=test_tform)
    trainloader_noshuffle = DataLoader(trainset_noshuffle, batch_size=args.bs,
                                       shuffle=False)

    valset = datasets.VOCSegmentation(
        root=args.dataroot, year='2012', image_set='val',
        download=False, transforms=test_tform)
    valloader = DataLoader(valset, batch_size=args.bs, shuffle=False)

    """Prepare model
    NB even though there are two classes (i.e. mussel and background),
    num_classes=1 is used such that nn.Sigmoid(pred) = 0 is bkg, and 1 is mussel.

    Could instead use num_classes=2 and nn.CrossEntropyLoss() such that the
    *channel* rather than the *value* encodes the class, but this would
    require a one-hot label format.
    """

    #writer = SummaryWriter(comment=f'LR_{args.lr}_BS_{args.bs}')
    global_step = 0

    # Optionally resume from existing checkpoint
    """
    if args.resume:
        if os.path.isfile(args.resume):
            # Load checkpoint.
            print('==> Resuming from checkpoint..')
            #checkpoint_file = os.path.join(args.resume, 'checkpoint/ckpt.t7.') + \
            #                  args.sess + '_' + str(args.seed)
            checkpoint = torch.load(args.resume)
            net = checkpoint['net']
            best_acc = checkpoint['loss']
            start_epoch = checkpoint['epoch'] + 1
            torch.set_rng_state(checkpoint['rng_state'])
    else:
        print("=> creating model '{}'".format(args.arch))
        net = models.__dict__[args.arch](num_classes=1).to(device)
    """
    """
    n_channels=3 for RGB images
    n_classes is the number of probabilities you want to get per pixel
     - For 1 class and background, use n_classes=1
     - For 2 classes, use n_classes=1
     - For N > 2 classes, use n_classes=N
    """
    bilinear = True if args.bilinear else False
    net = UNet(n_channels=3, n_classes=1, bilinear=bilinear).to(device)

    #net = pytorch_unet.UNet(1).to(device)
    print(summary(net, input_size=(3, 224, 224)))

    # Prepare training procedure
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, weight_decay=args.wd)

    """Note: BCEWithLogitsLoss uses the log-sum-exp trick for numerical
    stability, so this is safer than nn.BCELoss(nn.Sigmoid(pred))
    Todo: implement class weights to penalize model for predicting bkg"""
    loss_fn = nn.BCEWithLogitsLoss() # sigmoid cross entropy
    #         nn.CrossEntropyLoss() # softmax cross entropy

    # need to explicitly apply Sigmoid for IoU
    sig = nn.Sigmoid()

    def evaluate(data_loader):
        running_iou = 0
        running_loss = 0
        for inputs, targets in valloader:
            inputs, targets = inputs.cuda(), targets.cuda()
            with torch.no_grad():
                pred = net(inputs)#['out'] # fprop
                batch_iou = eval_binary_iou(sig(pred), targets)
                running_iou += batch_iou.item()
                # dataloader outputs targets with shape NHW, but we need NCHW
                targets = targets.unsqueeze(dim=1).float()
                batch_loss = loss_fn(pred, targets)
                running_loss += batch_loss.item()
        running_iou /= len(valloader)
        running_loss /= len(valloader)
        return running_iou, running_loss

    # Train
    for epoch in range(args.epochs):
        #train_iou = 0
        #train_loss = 0
        for batch, (inputs, targets) in enumerate(trainloader):
            lr = adjust_learning_rate(optimizer, epoch, args.drop, args.lr)

            """inputs are in NCHW format: N=nb. samples, C=channels, H=height,
            W=width. Do inputs.permute(0, 2, 3, 1) to viz in RGB format."""
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs) # fprop for unet
            #pred = net(inputs)['out'] # fprop for torchvision.models.segmentation
            # evaluate diagnostic metrics
            batch_iou = eval_binary_iou(sig(pred), targets)
            #train_iou += batch_iou.item()
            # dataloader outputs targets with shape NHW, but we need NCHW
            targets = targets.unsqueeze(dim=1).float()
            batch_loss = loss_fn(pred, targets)
            #train_loss += batch_loss.item()

            optimizer.zero_grad() # reset gradients
            batch_loss.backward() # bprop
            optimizer.step() # update parameters

            #writer.add_scalar('Loss/train', batch_loss.item(), global_step)
            global_step += 1

            if batch % 10 == 0:
                print('Batch [{}/{}], train loss: {:.4f}, train IoU: {:.4f}'
                    .format(batch, len(trainloader), batch_loss.item(), batch_iou.item()))

        #train_loss /= len(trainloader)
        #train_iou /= len(trainloader)

        # Validate
        val_iou, val_loss = evaluate(valloader)
        train_iou, train_loss = evaluate(trainloader_noshuffle)
        '''
        writer.add_scalar('Loss/test', val_iou, global_step)
        writer.add_images('images', inputs, global_step)
        if net.n_classes == 1:
            writer.add_images('masks/true', targets, global_step)
            writer.add_images('masks/pred', sig(pred) > 0.5, global_step)
        '''
        print('Epoch [{}/{}], train loss: {:.4f}, val loss: {:.4f}, train IoU: {:.4f}, val IoU: {:.4f}'
              .format(epoch + 1, args.epochs, train_loss, val_loss, train_iou, val_iou))
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                [epoch, lr, np.round(train_loss, 4), np.round(train_iou, 4)])
    save_checkpoint(net, train_iou, epoch, args.logdir, model_string)
    #writer.close()
