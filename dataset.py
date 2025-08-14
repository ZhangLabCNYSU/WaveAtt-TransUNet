import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import cv2
import random
from torch import tensor

def adaptive_gamma(img, gamma_range=(0.5, 1.5)):
    if not isinstance(img, np.ndarray):
        raise TypeError("输入图像必须是 NumPy 数组。")

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = lab[:, :, 0]
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)

    mean_intensity = np.mean(enhanced_l) / 255
    gamma = gamma_range[0] + (gamma_range[1] - gamma_range[0]) * mean_intensity

    corrected = np.power(img / 255.0, gamma) * 255.0  
    return np.clip(corrected, 0, 255).astype(np.uint8)

class MyDataset(Dataset):
    def __init__(self, imgs_path, img_fm, mk_fm, data_aug=True, img_size=[224, 224], bbox_shift=5):
        self.imgs = [os.path.join(imgs_path, i) for i in os.listdir(imgs_path)]
        self.data_aug = data_aug
        self.size = img_size
        self.img_format = img_fm
        self.mask_format = mk_fm
        self.bbox_shift = bbox_shift
        self.adaptive_gamma = lambda x: adaptive_gamma(x)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        image_path = self.imgs[index]
        image = np.array(Image.open(image_path).convert('RGB'))
        image = self.adaptive_gamma(image)
        mask_path = image_path.replace('images', 'masks').replace(self.img_format, self.mask_format)
        mask = np.array(Image.open(mask_path).convert('L'))

        image = cv2.resize(image, (self.size[0], self.size[1]), interpolation=cv2.INTER_CUBIC)
        image = (image - image.min()) / (image.max() - image.min() + 1e-8)  
        image = np.transpose(image, (2, 0, 1)) 

        mask = cv2.resize(mask, (self.size[0], self.size[1]), interpolation=cv2.INTER_NEAREST)
        label_ids = np.unique(mask)[1:]
        label_id = random.choice(label_ids.tolist()) if len(label_ids) > 0 else 0
        mask = np.uint8(mask == label_id)

        # 数据增强
        if self.data_aug:
            if random.random() > 0.5:
                image = np.ascontiguousarray(image[:, :, ::-1]) 
                mask = np.ascontiguousarray(mask[:, ::-1])
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                h, w = self.size[0], self.size[1]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                image_rot = cv2.warpAffine(
                    np.transpose(image, (1, 2, 0)), 
                    M, (w, h),
                    flags=cv2.INTER_CUBIC,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )
                image = np.transpose(image_rot, (2, 0, 1)) 
                mask = cv2.warpAffine(
                    mask,
                    M, (w, h),
                    flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0
                )

        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            x_min, x_max = 0, self.size[1]-1
            y_min, y_max = 0, self.size[0]-1
        else:
            x_min, x_max = np.min(x_indices), np.max(x_indices)
            y_min, y_max = np.min(y_indices), np.max(y_indices)
        x_min = max(0, x_min - random.randint(0, self.bbox_shift))
        x_max = min(self.size[1]-1, x_max + random.randint(0, self.bbox_shift))
        y_min = max(0, y_min - random.randint(0, self.bbox_shift))
        y_max = min(self.size[0]-1, y_max + random.randint(0, self.bbox_shift))
        bboxes = np.array([x_min, y_min, x_max, y_max])
        bbox_channel = np.zeros((1, self.size[0], self.size[1]))
        bbox_channel[0, y_min:y_max+1, x_min:x_max+1] = 1.0
        image = np.concatenate([image, bbox_channel], axis=0)

        return (
            tensor(image).float(),
            tensor(mask).long(),
            tensor(bboxes).float()
        )