import os
import numpy as np
from sklearn.metrics import jaccard_score as jsc

import torch
from torch import nn

from tqdm import tqdm

#from apex import amp

def save_amp_checkpoint(net, amp, optimizer, val_loss, trn_loss, epoch, logdir, model_string):
    """Saves model weights at a particular <epoch> into folder
    <logdir> with name <model_string>."""
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict(),
        'val_loss': val_loss,
        'trn_loss': trn_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(os.path.join(logdir, 'checkpoint/')):
        os.mkdir(os.path.join(logdir, 'checkpoint/'))

    torch.save(state, os.path.join(logdir, 'checkpoint/') +
               model_string + 'amp_epoch%d.pt' % epoch)


def save_checkpoint(net, val_loss, trn_loss, epoch, logdir, model_string):
    """Saves model weights at a particular <epoch> into folder
    <logdir> with name <model_string>."""
    print('Saving..')
    state = {
        'net': net,
        'val_loss': val_loss,
        'trn_loss': trn_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(os.path.join(logdir, 'checkpoint/')):
        os.mkdir(os.path.join(logdir, 'checkpoint/'))

    torch.save(state, os.path.join(logdir, 'checkpoint/') +
               model_string + '_epoch%d.ckpt' % epoch)

def evaluate(net, data_loader, loss_fn, device):
    """Evaluates the intersection over union (IoU) and
    cross entropy loss of DL model given by `net` on data from
    `data_loader`
    """
    sigmoid = nn.Sigmoid()

    running_iou = 0
    running_loss = 0

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(data_loader):
            targets_np = targets.numpy()
            inputs, targets = inputs.to(device), targets.to(device)
            pred = sigmoid(net(inputs))

            # dataloader outputs targets with shape NHW, but we need NCHW
            batch_loss = loss_fn(pred, targets.unsqueeze(dim=1).float())
            running_loss += batch_loss.item()
            # jaccard similarity (IoU) on CPU
            pred_np = pred.detach().cpu().numpy()
            # flatten predictions and targets for IoU calculation

            t_one_hot = np.zeros((targets_np.shape[0], 2, targets_np.shape[1], targets_np.shape[2]))
            t_one_hot[:, 1, :, :][targets_np == 1] = 1
            t_one_hot[:, 0, :, :][targets_np == 0] = 1

            p_one_hot = np.zeros((pred_np.shape[0], 2, pred_np.shape[2], pred_np.shape[3]))
            p_one_hot[:, 1, :, :][pred_np.squeeze().round() == 1] = 1
            p_one_hot[:, 0, :, :][pred_np.squeeze().round() == 0] = 1

            running_iou += jsc(p_one_hot.reshape(pred_np.shape[0], -1),
                               t_one_hot.reshape(targets_np.shape[0], -1),
                               average='samples')
            '''
            try:
                running_iou += jsc(
                    pred_np.round()[:, 0].reshape(pred_np.shape[0], -1),
                    targets_np.reshape(targets_np.shape[0], -1), average='samples')
            except ValueError:
                running_iou += 1.
            '''
    return running_iou / len(data_loader), running_loss / len(data_loader)


def evaluate_loss(net, data_loader, loss_fn, device):
    """Evaluates the cross entropy loss of DL model given by `net` on data from
    `data_loader`
    """
    running_loss = 0

    for inputs, targets in data_loader:
        break

    with torch.no_grad():
        for inputs, targets in tqdm(data_loader, unit=' images', unit_scale=inputs.shape[0]):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = net(inputs)
            # dataloader outputs targets with shape NHW, but we need NCHW
            batch_loss = loss_fn(pred, targets.unsqueeze(dim=1).float())
            running_loss += batch_loss.item()
    return running_loss / len(data_loader)



def eval_binary_iou(outputs, targets, eps=1e-6):
    """Returns the average binary intersection-over-union score.
    Similar to sklearn.metrics.jaccard_similarity_score.
    @param outputs are model predictions (post-sigmoid) in Nx1xHxW format.
    @param targets are the labels in NxHxW format.
    @param eps is a small constant to prevent division by zero.
    """
    outputs = outputs.squeeze(1).round().long()
    # zero if output=0 or pred=0
    intersection = (outputs & targets).float().sum((1, 2))
    union = (outputs | targets).float().sum((1, 2))
    iou = intersection / (union + eps)
    return iou


def adjust_learning_rate(optimizer, epoch, drop, base_learning_rate):
    """decrease the learning rate at <drop> epoch"""
    lr = base_learning_rate
    if epoch >= drop:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        '''
        if param_group['initial_lr'] == base_learning_rate:
            param_group['lr'] = lr
        else:
            if epoch <= 9:
                param_group['lr'] = param_group['initial_lr'] * lr / base_learning_rate
            elif epoch < 100:
                param_group['lr'] = param_group['initial_lr']
            elif epoch < 150:
                param_group['lr'] = param_group['initial_lr'] / 10.
            else:
                param_group['lr'] = param_group['initial_lr'] / 100.
        '''
    return lr


def pretty_image(axes):
    for ax in axes:
        ax.axis('off')


def pixel_acc(pred, label):
    #_, preds = torch.max(pred, dim=1)
    preds = torch.argmax(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc
