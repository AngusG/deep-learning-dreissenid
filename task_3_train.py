import os
import torch

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

# import an off-the-shelf model for now
from torchvision.models.segmentation import fcn_resnet50

if __name__ == '__main__':

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = datasets.VOCSegmentation(
        root='/scratch/ssd/cciw/', year='2012', image_set='train',
        download=False, transform=transform, target_transform=transform)

    trainloader = DataLoader(trainset, batch_size=5, shuffle=True)

    net = fcn_resnet50(num_classes=2).cuda()

    for inputs, targets in trainloader:
        inputs, targets = inputs.cuda(), targets.cuda()
        break
