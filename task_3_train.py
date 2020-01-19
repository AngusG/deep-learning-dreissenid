# general
import os
import csv
import argparse
import numpy as np
# ml libs
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# import an off-the-shelf model for now
from torchvision.models import segmentation as models

def save_checkpoint(loss, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net,
        'loss': loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(os.path.join(args.logdir, 'checkpoint/')):
        os.mkdir(os.path.join(args.logdir, 'checkpoint/'))
    torch.save(state, os.path.join(args.logdir, 'checkpoint/') +
               args.arch + '_bs%d' % args.bs + '_wd%.e' % args.wd + '_' +
               args.sess + '_' + str(args.seed) + '.ckpt')

if __name__ == '__main__':

    model_names = sorted(
        name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name])
        )

    parser = argparse.ArgumentParser()

    # dataset, admin, checkpointing and hw details
    parser.add_argument('--dataroot', help='path to dataset',
                        type=str, default='/scratch/ssd/cciw/')
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
                        type=int, default=10)
    parser.add_argument('--bs', help='SGD mini-batch size',
                        type=int, default=50)
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
    logname = result_folder + \
    args.arch + '_bs%d' % args.bs + '_wd%.e' % args.wd + '_' + \
    args.sess + '_' + str(args.seed) + '.csv'

    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'lr', 'train loss'])
        #logwriter.writerow(['epoch', 'lr', 'train loss', 'train acc', 'test loss', 'test acc'])

    # Prepare dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.VOCSegmentation(
        root=args.dataroot, year='2012', image_set='train',
        download=False, transform=transform, target_transform=transform)
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

    loss_fn = nn.BCEWithLogitsLoss() # sigmoid cross entropy
    #         nn.CrossEntropyLoss() # softmax cross entropy

    # Train
    for epoch in range(args.epochs):
        for batch, (inputs, targets) in enumerate(trainloader):
            """inputs are in NCHW format: N=nb. samples, C=channels, H=height,
            W=width. Do inputs.permute(0, 2, 3, 1) to viz in RGB format."""
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs)['out'] # fprop
            # multiply targets by 255 to treat label 1/255 in uint8 as a float 1
            loss = loss_fn(pred, targets * 255)
            optimizer.zero_grad() # reset gradients
            loss.backward() # bprop
            optimizer.step() # update parameters
        train_loss = loss.item()
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, args.epochs, train_loss))
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                [epoch, args.lr, np.round(train_loss, 4)])
    save_checkpoint(train_loss, epoch)
