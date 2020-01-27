import os
import torch


def save_checkpoint(net, loss, epoch, logdir, model_string):
    # Save checkpoint.
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
