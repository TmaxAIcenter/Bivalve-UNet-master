import argparse
import logging
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from utils.data_loading import COCODataset
from model import UNet, DeepLabV3Plus
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime as dt
from utils.img_blended import mask_to_image, color_mask_to_image, make_mask_transparent, blending_images
from pathlib import Path

def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(COCODataset.preprocess(
        full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        full_mask = probs.squeeze().cpu().numpy()

        _, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
        fig = plt.figure(figsize=(16, 5))
        rows = 1
        cols = 5

        ax1 = fig.add_subplot(rows, cols, 1)
        ax1.set_title('BackGround')
        ax1.grid(False)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.imshow(full_mask[0])

        ax2 = fig.add_subplot(rows, cols, 2)
        ax2.set_title(args.output+'_EG')
        ax2.grid(False)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.imshow(full_mask[1])

        ax3 = fig.add_subplot(rows, cols, 3)
        ax3.set_title(args.output+'_LA')
        ax3.grid(False)
        ax3.set_xticks([])
        ax3.set_yticks([])
        ax3.imshow(full_mask[2])

        ax4 = fig.add_subplot(rows, cols, 4)
        ax4.set_title(args.output+'_SP')
        ax4.grid(False)
        ax4.set_xticks([])
        ax4.set_yticks([])
        ax4.imshow(full_mask[3])

        # ax5 = fig.add_subplot(rows, cols, 4)
        # ax5.set_title(args.output+'LA_LAD')
        # ax5.grid(False)
        # ax5.set_xticks([])
        # ax5.set_yticks([])
        # ax5.imshow(full_mask[4])

        full_mask = np.argmax(full_mask, 0)

    return full_mask


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_type', '-mt', dest='model_type', metavar='MT', type=str, default='unet', help='Model type is unet, deeplab_v3_plus')
    parser.add_argument('--model', '-m', default='MODEL.pth', metavar='FILE', help='Specify the file in which the model is stored')
    parser.add_argument('--input', '-i', dest='input', nargs='+', help='Filenames of images', required=True)
    parser.add_argument('--output', '-o', dest='output', metavar='O', type=str, default='FU', help='Dataset type is FU, MA')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--channels', '-ch', type=int, default=3, help='Number of channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    # predict 저장 경로
    dir_predict_save = './predict_save/'+ args.output +'_predict'

    # 현재 날짜 표시
    today = str(dt.today())[2:4]+str(dt.today())[5:7]+str(dt.today())[8:10]

    # gpu,cpu 동작
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f'\033[34m***************************** Predict masks from input images start!!!!!! *****************************\033[0m')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    print(f'''
    Model type:         {args.model_type}_Model
    Using device:       {device}
    Loading model:      {args.model}
    Loading image:      {args.input}
    Input Channels:     {args.channels}
    Output Classes:     {args.classes} 
    ''')

    # 모델 타입 설정 부분
    if args.model_type == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes)
    
    elif args.model_type == 'deeplab_v3_plus':
        net = DeepLabV3Plus(n_channels=args.channels, n_classes=args.classes)

    else:
        net = UNet, DeepLabV3Plus

    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    # predict_img  마스크 생성 및 저장
    print(f'\033[34m************************************ {args.output}_predict_image_saving!!!!!! ************************************\033[0m')
    print('\n')
    for i, filename in enumerate(args.input):
        img = Image.open(filename)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           device=device)

        scale = args.scale
        img = img.resize((int(img.width * scale), int(img.height * scale)))
        tr_masks = []
        results = color_mask_to_image(mask)

        Path(dir_predict_save).mkdir(parents=True, exist_ok=True)

        result = mask_to_image(mask)
        result.save(os.path.join(dir_predict_save,
                    f'{today}_{args.output}_{args.model_type}_predict_mask.jpg'))
        logging.info(f'{today}_{args.output}_{args.model_type}_predict_mask.jpg saved!')
        
        for k, r in enumerate(results):
            mask = r.astype(np.uint8)
            mask = make_mask_transparent(mask)
            tr_masks.append(mask)
            mask = Image.fromarray(r.astype(np.uint8))

        img = np.array(img)
        blended_img = blending_images(img, tr_masks)
        blended_img = Image.fromarray(blended_img.astype(np.uint8))
        blended_img.save(os.path.join(dir_predict_save,
                         f'{today}_{args.output}_{args.model_type}_predict_alpha_masks.jpg'))
        logging.info(f'{today}_{args.output}_{args.model_type}_predict_alpha_masks.jpg saved!')

        plt.savefig(os.path.join(dir_predict_save,
                    f'{today}_{args.output}_{args.model_type}_predict_plt.png'))
        logging.info(f'{today}_{args.output}_{args.model_type}_predict_plt.png saved!')
    print('\n')
    print(f'\033[34m********************************** {args.output}_predict_image_save finish!!!!! **********************************\033[0m')
