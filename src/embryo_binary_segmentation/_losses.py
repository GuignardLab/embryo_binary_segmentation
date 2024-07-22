import torch


def dice_loss(y_real, y_pred):
    num = (y_real * y_pred).sum((1, 2, 3, 4))
    den = y_real.sum((1, 2, 3, 4)) + y_pred.sum((1, 2, 3, 4))
    
    SMOOTH = 1e-5
    dice_coefficient = (2. * num + SMOOTH) / (den + SMOOTH)
    approx_dice_loss = 1 - dice_coefficient
    
    return torch.mean(approx_dice_loss)


def focal_loss(y_real, y_pred, eps=1e-7, gamma=2):
    y_real = y_real.view(-1)
    y_pred = y_pred.view(-1)
    y_pred = torch.clamp(y_pred, eps, 1 - eps)
    
    positive_loss = (1 - y_pred) ** gamma * y_real * torch.log(y_pred)
    negative_loss = y_pred ** gamma * (1 - y_real) * torch.log(1 - y_pred)
    
    loss = positive_loss + negative_loss

    return -torch.mean(loss)




