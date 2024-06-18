

def dice_score(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum((1, 2))
    union = preds.sum((1, 2)) + targets.sum((1, 2))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

def iou(preds, targets, smooth=1e-6):
    preds = (preds > 0.5).float()
    targets = (targets > 0.5).float()
    intersection = (preds * targets).sum((1, 2))
    union = preds.sum((1, 2)) + targets.sum((1, 2)) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.mean().item()

def pixel_accuracy(preds, targets):
    correct = (preds == targets).float()
    accuracy = correct.sum() / correct.numel()
    return accuracy.item()
