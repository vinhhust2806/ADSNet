import torch.nn as nn
import torch

def dice_loss(pred, label):
    smooth = 1.0
    intersection = 2.0 * (pred * label).sum() + smooth
    union = pred.sum() + label.sum() + smooth
    return 1 - (intersection / union)

def bce_dice_loss(pred, label):
    dice = dice_loss(pred, label)
    bce = nn.BCELoss()(pred, label)
    return dice + bce

def dice_score(y_true, y_pred, eps=1e-9):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred)
    dice_score = (2. * intersection + eps) / (union + eps)
    return dice_score

def iou_score(y_true, y_pred, eps=1e-9):
    intersection = torch.sum(y_true * y_pred)
    union = torch.sum(y_true) + torch.sum(y_pred) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou

def mae_score(y_true, y_pred):
    mae = torch.mean(torch.abs(y_true - y_pred))
    return mae

def adjust_lr(optimizer, initial_lr, epoch, decay_rate=0.1, decay_epoch=40):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = initial_lr * decay
        print("Learning rate: ", param_group['lr'])