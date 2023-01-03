import torch
from torch import Tensor

def IoU(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    # 모든 배치 또는 단일 마스크에 대한 주사위 계수의 평균
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + (torch.sum(target)-inter)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return ( inter + epsilon) / (sets_sum + epsilon)

    else:
        # compute and average metric for each batch element
        # 각 배치 요소에 대한 계산 및 평균 메트릭
        iou = 0
        for i in range(input.shape[0]):
            iou += IoU(input[i, ...], target[i, ...])

        return iou / input.shape[0]


def mIoU(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    # 모든 클래스에 대한 IoU의 평균
    assert input.size() == target.size()
    iou = 0
    for channel in range(input.shape[1]):
        iou += IoU(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return iou / input.shape[1]

def mIou_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    # 0과 1 사이의 주사위 손실(최소화 목표)
    assert input.size() == target.size()
    fn = mIoU if multiclass else IoU
    return 1 - fn(input, target, reduce_batch_first=True)