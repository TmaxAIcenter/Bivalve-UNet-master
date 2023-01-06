import os
import argparse
import logging
import torch
import numpy as np
import wandb
import torch.nn as nn
import torch.nn.functional as F

from utils.data_loading import COCODataset
from model import UNet, DeepLabV3Plus
from datetime import datetime as dt
from pathlib import Path
from torch.utils.data import DataLoader
from utils.miou_loss import mIoU
from tqdm.auto import tqdm

def eval_test(net, device, batch_size: int = 1, img_scale: float = 0.5):
    net.eval()

    # Create dataset
    # Test_Dataset load
    dataset_EG = COCODataset(dir_img_EG, dir_ann_EG, img_scale)
    dataset_LA = COCODataset(dir_img_LA, dir_ann_LA, img_scale)
    dataset_SP = COCODataset(dir_img_SP, dir_ann_SP, img_scale)
    dataset = dataset_EG + dataset_LA + dataset_SP

    # num_workers=4 = cpu 작업 코어를 몇개 사용 할건지
    n_test = len(dataset)
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(dataset,  shuffle=True, **loader_args)
                
    print('\n')
    print(f'\033[34m ******************** Evaluate test {args.dataset_type}_wandb logs!!!!!! ******************** \033[0m')
    experiment = wandb.init(project=args.model_type, resume='allow', anonymous='must')
    experiment.config.update(dict(n_test=n_test, img_scale=img_scale))
    
    print('\n')                             
    print(f'\033[34m ******************** {args.dataset_type}_Dataset testing!!!!!! ******************** \033[0m')
    print(f'''  
    Model type:         {args.model_type}_Model
    Using device:       {device}
    Loading model:      {args.load}
    Batch size:         {batch_size}
    Testing size:       {n_test}
    Images scaling:     {img_scale}
    Input Channels:     {args.channels}
    Output Classes:     {args.classes} 
    ''')
    
    criterion = nn.CrossEntropyLoss()
    num_test = len(test_loader)
    miou = 0
    loss = 0
    sum_loss = 0
    div_step = 0
    
    # iterate over the test set
    date_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
    pbar = tqdm(test_loader, total=num_test, desc=f'\033[34m{date_time}\033[0m test round', unit='batch', leave=False)
    for data in pbar:
        image, mask_true = data['image'], data['mask']
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            loss = criterion(mask_pred, mask_true)
            # convert to one-hot format
            
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                one_miou = mIoU(mask_pred, mask_true, reduce_batch_first=False)
                
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                one_miou = mIoU(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)
                
        experiment.log({
                    'test latest loss': loss.item(),
                    'step': num_test,
                    'test latest miou': one_miou.item()
        })
        pbar.set_postfix(**{'test_loss': loss.item(), 'test_mIoU': one_miou.item()})
        
        loss += loss.item()
        sum_loss += loss.item()
        miou += one_miou        
        
    if num_test == 0:
        miou = miou.item()
        sum_loss = sum_loss
        
    else:
        miou = miou.item() / num_test
        sum_loss = sum_loss / num_test
            
    print(f'\033[34m ******************** testing Log!!!!!! ******************** \033[0m')
    print(f'''
    test mIoU : {miou}
    test loss : {sum_loss}
    ''')
    print(f'\033[34m *********************************************************** \033[0m','\n')
        
def get_args():
    parser = argparse.ArgumentParser(
    description='Evaluate test data the UNet on ture images and target masks')
    parser.add_argument('--wandb_mode','-w', dest='wandb_mode', metavar='W', type=str, default='online', help='WandB Mode is disabled, online, dryrun, offline, run')
    parser.add_argument('--model_type', '-mt', dest='model_type', metavar='MT', type=str, default='unet', help='Model type is unet, deeplab_v3_plus')
    parser.add_argument('--dataset_type', '-d', dest='dataset_type', metavar='D', type=str, default='FU', help='Dataset type is FU, MA')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--channels', '-ch', type=int, default=3, help='Number of channels')

    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    print('\n',f'\033[34m ******************** Evaluate test_{args.dataset_type}_Dataset loading!!!!!! ******************** \033[0m')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # 현재 날짜 표시
    today = str(dt.today())[2:4]+str(dt.today())[5:7]+str(dt.today())[8:10]
    
    # gpu,cpu 동작
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기본 설정
    # wandb API_KEY값 적용
    os.environ['WANDB_API_KEY'] = 'c1c86d5030d2a09e53ccf00d976c25b29d4740c9'
    os.environ['WANDB_MODE'] = args.wandb_mode

    # 날짜 표시
    today = dt.now().strftime("%y%m%d")
    
    # Test 파일경로
    # 수정란 Dataset 경로
    dir_img_EG = Path('./Dataset/'+ args.dataset_type +'/test/EG/images/')
    dir_ann_EG = Path('./Dataset/'+ args.dataset_type +'/test/EG/annotations/')

    # 유생 Dataset 경로
    dir_img_LA = Path('./Dataset/'+ args.dataset_type +'/test/LA/images/')
    dir_ann_LA = Path('./Dataset/'+ args.dataset_type +'/test/LA/annotations/')

    # 치패 Dataset 경로
    dir_img_SP = Path('./Dataset/'+ args.dataset_type +'/test/SP/images/')
    dir_ann_SP = Path('./Dataset/'+ args.dataset_type +'/test/SP/annotations/')


    # 모델 타입 설정 부분
    if args.model_type == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    elif args.model_type == 'deeplab_v3_plus':
        net = DeepLabV3Plus(n_channels=args.channels, n_classes=args.classes)
    
    net.load_state_dict(torch.load(args.load, map_location=device))
    logging.info(f'Evaluate test dataset from {args.load}')
    net.to(device=device)
    
    eval_test(net=net,
              device=device,
              batch_size=args.batch_size,
              img_scale=args.scale)
