import os
import torch
from torch import nn
from sklearn.metrics import jaccard_similarity_score as jsc


def save_checkpoint(net, loss, epoch, logdir, model_string):
    """Saves model weights at a particular <epoch> into folder
    <logdir> with name <model_string>."""
    print('Saving..')
    state = {
        'net': net,
        'loss': loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir(os.path.join(logdir, 'checkpoint/')):
        os.mkdir(os.path.join(logdir, 'checkpoint/'))

    torch.save(state, os.path.join(logdir, 'checkpoint/') +
               model_string + '.ckpt')

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

            try:
                running_iou += jsc(
                    pred_np.round()[:, 0].reshape(pred_np.shape[0], -1),
                    targets_np.reshape(targets_np.shape[0], -1))
            except ValueError:
                running_iou += 1.

    return running_iou / len(data_loader), running_loss / len(data_loader)



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
    return iou.mean()


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
