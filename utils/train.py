import torch 
from tqdm import tqdm 
import numpy as np
from utils.matric import *

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
            pred = (output > 0.5).float()  

            dice += dice_score(target, pred)
            iou += iou_score(target, pred)
            mae += mae_score(target, pred)
    
    num_batches = len(val_loader)
    return val_loss / num_batches, dice / num_batches, iou / num_batches, mae / num_batches

def train(model, train_loader, val_loader, loss_function, optimizer, epochs):
    loss_history = []
    train_history = []
    val_history = []
    dice_max = 0

    for epoch in range(epochs):
        model.train()

        losses = []
        train_dice = []

        for i, (image, mask) in enumerate(tqdm(train_loader)):
            image = image.to(device)
            mask = mask.to(device)
            outputs = model(image)
            
            out_cut = (outputs > 0.5).float()  
            loss = loss_function(outputs, mask)
            
            losses.append(loss.item())
            train_dice.append(dice_score(mask, out_cut).cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        val_loss, val_dice, val_iou, val_mae = validate(model, val_loader, loss_function)
    
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_dice).mean())
        val_history.append(val_dice)

        if val_dice > dice_max:
          dice_max = val_dice
          torch.save(model.state_dict(), '{%.3f}-{%d}.pth' % (dice_max, epoch+1))
          print('[Saving Snapshot:]', '{%.3f}-{%d}.pth'% (dice_max, epoch+1))

        print('Epoch : {}/{}'.format(epoch+1, epochs))
        print('train_loss: {:.3f} - train_dice: {:.3f} - val_loss: {:.3f} - val_dice: {:.3f} - val_iou: {:.3f} - val_mae: {:.3f}'.format(np.array(losses).mean(),
                                                                               np.array(train_dice).mean(),
                                                                               val_loss,
                                                                               val_dice,
                                                                               val_iou,
                                                                               val_mae))
    return loss_history, train_history, val_history