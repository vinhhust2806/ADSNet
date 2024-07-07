from tqdm import tqdm
from utils import *
from models import Model
import torch 
import numpy as np
from dataset import train_loader, validation_loader
torch.manual_seed(31)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def validate(model, val_loader, criterion):
    model.eval()
    val_loss = 0
    dice = 0
    iou = 0
    mae = 0
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            pred = (output > 0.5).float()  # Assuming sigmoid activation for binary segmentation
            dice += dice_coefficient(target, pred)
            iou += iou_score(target, pred)
            mae += mae_score(target, pred)
    
    num_batches = len(val_loader)
    return val_loss / num_batches, dice / num_batches, iou / num_batches, mae / num_batches

def train_model(train_loader, val_loader, loss_func, optimizer, num_epochs):
    loss_history = []
    train_history = []
    val_history = []
    max = 0

    for epoch in range(num_epochs):
        #adjust_lr(optimizer, 1e-4, epoch, 0.1, 50)
        model.train()

        losses = []
        train_dice = []

        for i, (image, mask) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            #outputs = output1 + output2 + output3 + output4
            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0

            train_dice_ = dice_coef_metric(out_cut, mask.data.cpu().numpy())
            loss = loss_func(outputs, mask) #+ loss_func(outputs1, mask) + loss_func(outputs2, mask)
            losses.append(loss.item())
            train_dice.append(train_dice_)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_dice, val_iou, val_mae = validate(model, val_loader, loss_func)
        #scheduler.step(val_mean_iou)
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_dice).mean())
        val_history.append(val_dice)
        if val_dice > max:
          max = val_dice
          torch.save(model.state_dict(), '{%.3f}-{%d}.pth' % (max, epoch+1))
          print('[Saving Snapshot:]', '{%.3f}-{%d}.pth'% (max, epoch+1))
        print('Epoch : {}/{}'.format(epoch+1, num_epochs))
        print('loss: {:.3f} - dice_coef: {:.3f} - val_loss: {:.3f} - val_dice_coef: {:.3f} - val_iou: {:.3f} - val_mae: {:.3f}'.format(np.array(losses).mean(),
                                                                               np.array(train_dice).mean(),
                                                                               val_loss,
                                                                               val_dice,
                                                                               val_iou,
                                                                               val_mae))
    return loss_history, train_history, val_history

model = Model().cuda()
params = model.parameters()

optimizer = torch.optim.AdamW(params, 1e-4, weight_decay=1e-4)
train_model(train_loader, validation_loader, bce_dice_loss, optimizer, 200)
