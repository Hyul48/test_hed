import numpy as np
import torch

# loss function
def sigmoid_cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy_with_logits(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def cross_entropy_loss(prediction, label):
    #print (label,label.max(),label.min())
    label = label.long()
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    #print (num_positive, num_negative)
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)

def weighted_nll_loss(prediction, label):
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = torch.nn.functional.nll_loss(prediction, label, reduce=False)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask != 0] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    return torch.sum(cost)

def weighted_cross_entropy_loss(prediction, label, output_mask=False):
    criterion = torch.nn.CrossEntropyLoss(reduce=False)
    label = torch.squeeze(label.long(), dim=0)
    nch = prediction.shape[1]
    label[label >= nch] = 0
    cost = criterion(prediction, label)
    mask = (label != 0).float()
    num_positive = torch.sum(mask).float()
    num_negative = mask.numel() - num_positive
    mask[mask == 1] = num_negative / (num_positive + num_negative)
    mask[mask == 0] = num_positive / (num_positive + num_negative)
    cost = torch.mul(cost, mask)
    if output_mask:
        return torch.sum(cost), (label != 0)
    else:
        return torch.sum(cost)

def l2_regression_loss(prediction, label, mask):
    label = torch.squeeze(label.float())
    prediction = torch.squeeze(prediction.float())
    mask = (mask != 0).float()
    num_positive = torch.sum(mask).float()
    cost = torch.nn.functional.mse_loss(prediction, label, reduce=False)
    cost = torch.mul(cost, mask)
    cost = cost / (num_positive + 0.00000001)
    return torch.sum(cost)

SMOOTH = 1e-6
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape

    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W

    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))  # Will be zzero if both are 0

    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    return iou


def iou_numpy(outputs: np.array, labels: np.array):
    intersection = (outputs & labels).sum((0, 1))
    union = (outputs | labels).sum((0, 1))
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    return iou  # Or thresholded.mean()


def acc_metrics(outputs, labels):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    # TP    predict 和 label 同时为1
    TP += ((outputs == 1) & (labels == 1)).sum()
    # TN    predict 和 label 同时为0
    TN += ((outputs == 0) & (labels == 0)).sum()
    # FN    predict 0 label 1
    FN += ((outputs == 0) & (labels == 1)).sum()
    # FP    predict 1 label 0
    FP += ((outputs == 1) & (labels == 0)).sum()
    p = TP / (TP + FP)
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p)
    acc = (TP + TN) / (TP + TN + FP + FN)

    return p, r, F1, acc