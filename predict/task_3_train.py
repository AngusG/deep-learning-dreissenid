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

# import an off-the-shelf model for now
from torchvision.models import segmentation as models

# my utils
from task_3_utils import (save_checkpoint,
                          eval_binary_iou,
                          adjust_learning_rate)

if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
        )

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
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fcn_resnet50',
                        choices=model_names, help='model architecture: ' +
                        ' | '.join(model_names) + ' (default: fcn_resnet50)')
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
    not have a specific orientation. Note that this also rotates masks."""
    tform_image_and_mask = T.Compose([
        T.RandomCrop(224),
        T.RandomHorizontalFlip(0.5), # rotate image about y-axis with 50% prob
        T.RandomVerticalFlip(0.5),
        T.ToTensor()
    ])

    # Prepare dataset and dataloader
    trainset = datasets.VOCSegmentation(
        root=args.dataroot, year='2012', image_set='train',
        download=False, transforms=tform_image_and_mask)
    trainloader = DataLoader(trainset, batch_size=args.bs, shuffle=True)

    """Prepare model
    NB even though there are two classes (i.e. mussel and background),
    num_classes=1 is used such that nn.Sigmoid(pred) = 0 is bkg, and 1 is mussel.

    Could instead use num_classes=2 and nn.CrossEntropyLoss() such that the
    *channel* rather than the *value* encodes the class, but this would
    require a one-hot label format.
    """

    # Optionally resume from existing checkpoint
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

    # Train
    for epoch in range(args.epochs):
        train_iou = 0
        train_loss = 0
        for batch, (inputs, targets) in enumerate(trainloader):
            lr = adjust_learning_rate(optimizer, epoch, args.drop, args.lr)

            """inputs are in NCHW format: N=nb. samples, C=channels, H=height,
            W=width. Do inputs.permute(0, 2, 3, 1) to viz in RGB format."""
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs)['out'] # fprop
            # evaluate diagnostic metrics
            batch_iou = eval_binary_iou(sig(pred), targets)
            train_iou += batch_iou.item()
            # dataloader outputs targets with shape NHW, but we need NCHW
            targets = targets.unsqueeze(dim=1).float()
            batch_loss = loss_fn(pred, targets)
            train_loss += batch_loss.item()
            optimizer.zero_grad() # reset gradients
            batch_loss.backward() # bprop
            optimizer.step() # update parameters

        train_loss /= len(trainloader)
        train_iou /= len(trainloader)
        print('Epoch [{}/{}], Loss: {:.4f}, IoU: {:.4f}'
              .format(epoch + 1, args.epochs, train_loss, train_iou))
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                [epoch, lr, np.round(train_loss, 4), np.round(train_iou, 4)])
    save_checkpoint(net, train_iou, epoch, args.logdir, model_string)
