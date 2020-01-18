# general
import os
import argparse
# ml libs
import torch
from torch import nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# import an off-the-shelf model for now
from torchvision.models.segmentation import fcn_resnet50

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--log', help='directory to store checkpoints; \
                        if None, nothing will be saved')
    parser.add_argument('--do_print', help="print ongoing training progress",
                        action="store_true")
    parser.add_argument('--gpu', help='physical id of GPU to use')
    parser.add_argument('--seed', help='random seed', type=int, default=1)

    # training meta-parameters
    parser.add_argument('--epochs', help='number of epochs to train for',
                        type=int, default=10)
    parser.add_argument('--batch_size', help='SGD mini-batch size',
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

    # Prepare dataset and dataloader
    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.VOCSegmentation(
        root='/scratch/ssd/cciw/', year='2012', image_set='train',
        download=False, transform=transform, target_transform=transform)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True)

    """Prepare model
    NB even though there are two classes (i.e. mussel and background),
    num_classes=1 is used such that nn.Sigmoid(pred) = 0 is bkg, and 1 is mussel.

    Could instead use num_classes=2 and nn.CrossEntropyLoss() such that the
    *channel* rather than the *value* encodes the class, but this would
    require a one-hot label format.
    """
    net = fcn_resnet50(num_classes=1).to(device)

    # Prepare training procedure
    optimizer = torch.optim.SGD(
        net.parameters(), lr=args.lr, weight_decay=args.wd)

    loss_fn = nn.BCEWithLogitsLoss() # sigmoid cross entropy
    #         nn.CrossEntropyLoss() # softmax cross entropy

    # Train
    for epoch in range(args.epochs):
        for batch, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs)['out'] # fprop
            # multiply targets by 255 to treat label 1/255 in uint8 as a float 1
            loss = loss_fn(pred, targets * 255)
            optimizer.zero_grad() # reset gradients
            loss.backward() # bprop
            optimizer.step() # update parameters
        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, args.epochs, loss.item()))
