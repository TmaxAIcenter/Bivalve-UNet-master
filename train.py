import os
import argparse
import logging
import torch
import wandb
import torch.nn as nn
import torch.nn.functional as F

from evaluate import train_evaluate, validation_evaluate
from torch.utils.data import DataLoader
from utils.miou_loss import mIou_loss
from utils.data_loading import COCODataset
from utils.img_blended import img_blended
from model import UNet, DeepLabV3Plus
from datetime import datetime as dt
from pathlib import Path
from torch import optim
from tqdm import tqdm

def train_net(net,
              device,
              epochs: int = 100,
              gpus_type: int = 0,
              batch_size: int = 1,
              learning_rate: float = 1e-5,
              save_checkpoint: bool = True,
              img_scale: float = 0.5,
              amp: bool = False,
              transform_proportion: float = 0.0):

    # 1. Create dataset
    # Train_Dataset load
    T_dataset_EG = COCODataset(T_dir_img_EG, T_dir_ann_EG, img_scale, transform_proportion)
    T_dataset_LA = COCODataset(T_dir_img_LA, T_dir_ann_LA, img_scale, transform_proportion)
    T_dataset_SP = COCODataset(T_dir_img_SP, T_dir_ann_SP, img_scale, transform_proportion)
    T_dataset = T_dataset_EG + T_dataset_LA + T_dataset_SP

    # Val_Dataset load
    V_dataset_EG = COCODataset(V_dir_img_EG, V_dir_ann_EG, img_scale, transform_proportion)
    V_dataset_LA = COCODataset(V_dir_img_LA, V_dir_ann_LA, img_scale, transform_proportion)
    V_dataset_SP = COCODataset(V_dir_img_SP, V_dir_ann_SP, img_scale, transform_proportion)
    V_dataset = V_dataset_EG + V_dataset_LA + V_dataset_SP 

    # 2. Create data loaders
    # Split into train / validation partitions
    n_train = len(T_dataset)
    n_val = len(V_dataset)

    # num_workers=4 = cpu 작업 코어를 사용 개수
    loader_args = dict(batch_size=batch_size, num_workers=4, pin_memory=True)
    train_loader = DataLoader(T_dataset,  shuffle=True, **loader_args)
    val_loader = DataLoader(V_dataset, shuffle=False, drop_last=True, **loader_args)

    print('\n')
    print(f'\033[34m ******************** {args.dataset_type}_wandb logs!!!!!! ******************** \033[0m')
    experiment = wandb.init(project=args.model_type, resume='allow', anonymous='must')
    experiment.config.update(dict(epochs=epochs, batch_size=batch_size, gpus_type=gpus_type, learning_rate=learning_rate,
                                  save_checkpoint=save_checkpoint, img_scale=img_scale,
                                  amp=amp))

    print('\n')                             
    print(f'\033[34m ******************** {args.dataset_type}_Dataset training!!!!!! ******************** \033[0m')
    print(f'''
        WandB mode:             {args.wandb_mode}
        Model type:             {args.model_type}_Model
        Dataset type:           {args.dataset_type}_Dataset
        GPUS type:              {args.gpus_type}
        Input Channels:         {net.n_channels}
        Output Classes:         {net.n_classes} 
        Epochs:                 {epochs}
        Batch size:             {batch_size}
        Learning rate:          {learning_rate}
        Training size:          {n_train}
        Validation size:        {n_val}
        Checkpoints:            {save_checkpoint}
        Device:                 {device.type}
        Images scaling:         {img_scale}
        Transform Proportion:   {transform_proportion}
        Mixed Precision:        {amp}
    ''')

    # 3. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.NAdam(net.parameters(), lr=learning_rate, betas=(0.9, 0.999), weight_decay=0, momentum_decay=0.004, eps=1e-08)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=2, verbose=1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    global_step = 0

    # 4. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        train_loss = 0
        train_mIoU_loss = 0
        date_time = dt.now().strftime("%Y-%m-%d %H:%M:%S")
        with tqdm(total=n_train, desc=f'\033[34m{date_time}\033[0m Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images = batch['image']
                true_masks = batch['mask']

                assert images.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.long)\

                with torch.cuda.amp.autocast(enabled=amp):
                    masks_pred = net(images)
                    train_loss = criterion(masks_pred, true_masks)
                    train_mIoU_loss = mIou_loss(F.softmax(masks_pred, dim=1).float(),
                                      F.one_hot(true_masks, net.n_classes).permute(
                                          0, 3, 1, 2).float(),
                                      multiclass=True)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(train_loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                train_loss += train_loss.item()
                train_mIoU_loss += train_mIoU_loss.item()
                experiment.log({
                    'validation loss': train_mIoU_loss.item(),
                    'train loss': train_loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'train_loss': train_loss.item(),'train_mIoU_loss': train_mIoU_loss.item()})


                # Evaluation round
                division_step = (n_train // (1 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in net.named_parameters():
                            tag = tag.replace('/', '.')
                            histograms['Weights/' +
                                       tag] = wandb.Histogram(value.data.cpu())
                            histograms['Gradients/' +
                                       tag] = wandb.Histogram(value.grad.data.cpu())

                        # train (f1_score, mIoU)
                        train_mIoU = train_evaluate(net, train_loader, device)
                        scheduler.step(train_mIoU)

                        # validation (f1_score, mIoU)
                        val_mIoU = validation_evaluate(net, val_loader, device)
                        scheduler.step(val_mIoU)

                        # mIoU(train,validation)수치 log
                        print('\n',f'********** epoch{epoch} mIoU_Log **********')
                        logging.info(f'train mIoU: {train_mIoU}')
                        logging.info(f'validation mIoU: {val_mIoU}')

                        # wandb 그래프 log
                        experiment.log({
                            'learning rate': optimizer.param_groups[0]['lr'],
                            'train mIoU': train_mIoU,
                            'validation mIoU': val_mIoU,
                            'images': wandb.Image(images[0].cpu()),
                            'masks': {
                                'blended_true_mask': wandb.Image(img_blended(images[0], true_masks[0], T_mask=True)),
                                'blended_pred_mask': wandb.Image(img_blended(images[0], masks_pred.argmax(dim=1)[0], T_mask=False)),
                                'true_mask': wandb.Image(true_masks[0].float().cpu()),
                                'pred_mask': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                            },
                            'step': global_step,
                            'epoch': epoch,
                            **histograms
                        })

        # chackpoint save
        if save_checkpoint:
            Path(dir_checkpoint/args.model_type).mkdir(parents=True, exist_ok=True)
            torch.save(net.state_dict(), str(dir_checkpoint/args.model_type /
                       f'{today}_{args.dataset_type}_{args.model_type}_checkpoint_epoch{epoch}.pth'))
            logging.info(f'{today}_{args.dataset_type}_{args.model_type}_Checkpoint_epoch{epoch} saved!')
        print('\n')

def get_args():
    parser = argparse.ArgumentParser(
description='Train the UNet on images and target masks')
    parser.add_argument('--wandb_mode','-w', dest='wandb_mode', metavar='W', type=str, default='online', help='WandB Mode is disabled, online, dryrun, offline, run')
    parser.add_argument('--model_type', '-mt', dest='model_type', metavar='MT', type=str, default='unet', help='Model type is uent, deeplab_v3_plus')
    parser.add_argument('--dataset_type', '-d', dest='dataset_type', metavar='D', type=str, default='FU', help='Dataset type is FU, MA')
    parser.add_argument('--gpus_type', '-g', dest='gpus_type', metavar='str', type=str, default='0', help='Gpus type is 0, 1, 2, 3, 4')                    
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=8, help='Batch size')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--channels', '-ch', type=int, default=3, help='Number of channels')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--transform', '-t', type=float, default=0.0, help='Transform(horizontal, vertical, brightness, contrast) proportion of all classes')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    # gpu 장치 번호
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus_type

    print(f'\033[34m ******************** {args.dataset_type}_Dataset loading!!!!!! ******************** \033[0m')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 기본 설정
    # wandb API_KEY값 적용
    os.environ['WANDB_API_KEY'] = 'c1c86d5030d2a09e53ccf00d976c25b29d4740c9'
    os.environ['WANDB_MODE'] = args.wandb_mode

    # 날짜 표시
    now = dt.now()
    today = now.strftime("%y%m%d")
    
    # Train 파일경로
    # 수정란 Dataset 경로
    T_dir_img_EG = Path('./Dataset/'+ args.dataset_type +'/train/EG/images/')
    T_dir_ann_EG = Path('./Dataset/'+ args.dataset_type +'/train/EG/annotations/')

    # 유생 Dataset 경로
    T_dir_img_LA = Path('./Dataset/'+ args.dataset_type +'/train/LA/images/')
    T_dir_ann_LA = Path('./Dataset/'+ args.dataset_type +'/train/LA/annotations/')

    # 치패 Dataset 경로
    T_dir_img_SP = Path('./Dataset/'+ args.dataset_type +'/train/SP/images/')
    T_dir_ann_SP = Path('./Dataset/'+ args.dataset_type +'/train/SP/annotations/')

    # Val 파일경로
    # 수정란 Dataset 경로    
    V_dir_img_EG = Path('./Dataset/'+ args.dataset_type +'/validation/EG/images/')
    V_dir_ann_EG = Path('./Dataset/'+ args.dataset_type +'/validation/EG/annotations/')

    # 유생 Dataset 경로
    V_dir_img_LA = Path('./Dataset/'+ args.dataset_type +'/validation/LA/images/')
    V_dir_ann_LA = Path('./Dataset/'+ args.dataset_type +'/validation/LA/annotations/')

    # 치패 Dataset 경로
    V_dir_img_SP = Path('./Dataset/'+ args.dataset_type +'/validation/SP/images/')
    V_dir_ann_SP = Path('./Dataset/'+ args.dataset_type +'/validation/SP/annotations/')

    # 체크포인트 경로
    dir_checkpoint = Path('./checkpoints/'+ args.dataset_type +'_checkpoints')
    
    # 학습 중단 체크포인트 경로
    dir_interrupted = Path('./interrupted/'+ args.dataset_type +'_interrupted')

    # 모델 타입 설정 부분
    if args.model_type == 'unet':
        net = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)

    elif args.model_type == 'deeplab_v3_plus':
        net = DeepLabV3Plus(n_channels=args.channels, n_classes=args.classes)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')
    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  gpus_type=args.gpus_type,
                  learning_rate=args.lr,
                  device=device,
                  img_scale=args.scale,
                  amp=args.amp,
                  transform_proportion=args.transform)

    except KeyboardInterrupt:
        Path(dir_interrupted/args.model_type).mkdir(parents=True, exist_ok=True)
        torch.save(net.state_dict(), str(dir_interrupted/args.model_type /
                   f'{today}_{args.dataset_type}_{args.model_type}_interrupted.pth'))
        logging.info(
            f'{today}_{args.dataset_type}_{args.model_type}_interrupted saved!')
        print('\n')
        raise
