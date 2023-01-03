import cv2
import numpy as np
import torch
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# 일반 mask
def mask_to_image(mask: np.ndarray):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    elif mask.ndim == 3:
        return Image.fromarray((np.argmax(mask, axis=0) * 255 / mask.shape[0]).astype(np.uint8))


# mask 색상
def color_mask_to_image(mask: np.ndarray):
        mask = torch.from_numpy(mask).unsqueeze(2)
        num_class = 5
        color_map = {0: np.array([0, 0, 0]), 1: np.array([255,255,0]), 2: np.array([173,255,47]), 3: np.array([30,144,255]), 4: np.array([255, 0, 0])}
        masks = []
        for k in range(num_class):
            color_mask = np.zeros((mask.shape[0], mask.shape[1], 3))
            color_mask = np.where(mask == k, color_map[k], 0)
            masks.append(color_mask)
        return masks

# mask 배경 투명화
def make_mask_transparent(mask):
    b_channel, g_channel, r_channel = cv2.split(mask)
    _alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 90  # 255 * 0.35
    _alpha_channel = np.where((mask[:, :, 0] == 0) & (mask[:, :, 1] == 0) & (mask[:, :, 2] == 0), 0, _alpha_channel) # 배경(0,0,0) 투명화
    mask = cv2.merge((b_channel, g_channel, r_channel, _alpha_channel)) # BGRA
    return mask

# mask 배경 투명화
def blending_images(image, masks):
    ALPHA = 0.65
    for mask in masks:
        for color_channel in range(3):
            image[:, :, color_channel] = np.where(mask[:, :, 3] !=0, image[:, :, color_channel] * ALPHA + mask[:, :, color_channel] * (1-ALPHA), image[:, :, color_channel])
    return image

def img_blended(image, mask, T_mask):
    color_mask_to_image
    make_mask_transparent
    blending_images

    mask = np.array(mask.cpu())
    mask_con = []
    mask_results = color_mask_to_image(mask)
    
    for i in mask_results:
        mask = i.astype(np.uint8)
        mask = make_mask_transparent(mask)
        mask_con.append(mask)
    
    topilimage = torchvision.transforms.ToPILImage()
    image = topilimage(image)
    image = np.array(image) 
    blended_mask = blending_images(image, mask_con)
    blended_mask = Image.fromarray(blended_mask.astype(np.uint8))
           
    return blended_mask
    