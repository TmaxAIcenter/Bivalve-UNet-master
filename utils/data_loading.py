# -*- coding: utf-8 -*-
import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import os
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from utils.pycocotools.coco import COCO
import albumentations as A
import glob

class COCODataset(Dataset):
    def __init__(self, img_dir: str, ann_dir: str, scale: float = 1.0, transform_proportion: float = 0.0, ann_suffix: str = '' ):
        self.img_dir = Path(img_dir)
        self.ann_dir = Path(ann_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.transform_proportion = transform_proportion
        self.ann_suffix = ann_suffix

        self.ids = [splitext(file)[0] for file in listdir(img_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {img_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {str(img_dir)} : {len(self.ids)} examples')


#predict 부분
    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))
            img_ndarray = img_ndarray / 255
        return img_ndarray


    def __len__(self):
        return len(self.ids)

    @staticmethod
    def load(filename):
        ext = splitext(filename)[1]
        if ext == '.npy':
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
        
    @staticmethod
    def filterDataset(self, annFile):
        # 인스턴스 주석에 대한 COCO API 초기화
        #annFile = f'./{Superfolder}/{task_folder}/annotations/instances_default_final.json'
        coco = COCO(annFile)
        classes = self.getCats(coco.dataset['categories'])
        image = coco.dataset['images'][0]
        return image, coco
    
    @staticmethod
    def getCats(categories):
        name_list = []
        
        for i in categories:
            name_list.append(i['name'])
        return name_list
    
    @staticmethod
    def getClassName(classID, cats):
        for i in range(len(cats)):
            if cats[i]['id']==classID:
                return cats[i]['name']
        return None
    
    @staticmethod
    def getNormalMask(self, imageObj, coco):
        classes = self.getCats(coco.dataset['categories'])
        input_image_size = (coco.dataset['images'][0]['width'], coco.dataset['images'][0]['height'])
        catIds = coco.getCatIds(catNms=classes)
        annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        cats = coco.loadCats(catIds)
        train_mask = np.zeros(input_image_size).T
        
        for a in range(len(anns)):
            className = self.getClassName(anns[a]['category_id'], cats)
            pixel_value = classes.index(className)+1
            new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, input_image_size)
            train_mask = np.maximum(new_mask, train_mask)
        return train_mask
    
    @staticmethod
    def transform(self, image, mask):
        aug = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=self.transform_proportion),
            A.VerticalFlip(p=self.transform_proportion),              
            A.HorizontalFlip(p=self.transform_proportion)
        ])
        image = np.array(image)
        mask = np.array(mask)
        augmented = aug(image=image, mask=mask)
        image_transform = augmented['image']
        mask_transform = augmented['mask']
        
        return Image.fromarray(image_transform), Image.fromarray(mask_transform)
    

    def __getitem__(self, idx):
        name = self.ids[idx]
        ann_file = list(self.ann_dir.glob(name + self.ann_suffix + '.*'))
        img_file = list(self.img_dir.glob(name + '.*'))
        
        imageObj, coco = self.filterDataset(self, ann_file[0])

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(ann_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {ann_file}'
        mask = Image.fromarray(self.getNormalMask(self, imageObj, coco))
        img = self.load(img_file[0])
        
        img, mask = self.transform(self, img, mask)
        

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }