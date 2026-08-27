import torch
from torch.utils.data import Dataset

import albumentations as A
import cv2

# Training Dataset Class
class MissingTeethTrainDM(Dataset):
    def __init__(self, df, roi, transform=None, split_pct=0.6):
        self.df = df
        self.ImagePath = self.df["ImagePath"]
        self.roi = roi
        self.split_pct = split_pct
        
        # Mapping ROI to FDI tooth numbering
        if self.roi == "q1":
            self.target_names = ['17', '16', '15', '14', '13', '12', '11']
        elif self.roi == "q2":
            self.target_names = ['21', '22', '23', '24', '25', '26', '27']
        elif self.roi == "q3":
            self.target_names = ['37', '36', '35', '34', '33', '32', '31']
        elif self.roi == "q4":
            self.target_names = ['41', '42', '43', '44', '45', '46', '47']
    
        self.Labels = self.df[self.target_names].values.tolist()
        self.transform = transform
        
        self.base_transform = A.Compose([
            A.Resize(height=512, width=1024),
            A.Normalize(normalization='min_max', mean=(0.5,), std=(0.5,), p=1.0),
            A.ToTensorV2(p=1.0)
        ])
        
    def __getitem__(self, index):
        image_path = self.ImagePath[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Transform full image
        if self.transform:
            processed = self.transform(image=image)['image']
        else:
            processed = self.base_transform(image=image)['image']
                        
        # 2. Calculate dynamic split points
        h, w = processed.shape[1], processed.shape[2]
        split_h = int(h * self.split_pct) # 512 * 0.6 = 307
        split_w = int(w * self.split_pct) # 1024 * 0.6 = 614
        
        # 3. Quadrant Slicing Logic
        if self.roi == "q1":
            # Q1 (Upper Right Patient/Upper Left Image): Top 60%, Left 60%
            quadrant = processed[:, :split_h, :split_w]
        elif self.roi == "q2":
            # Q2 (Upper Left Patient/Upper Right Image): Top 60%, Right 60%
            quadrant = processed[:, :split_h, (w - split_w):]
        elif self.roi == "q3":
            # Q3 (Lower Left Patient/Lower Right Image): Bottom 60%, Right 60%
            quadrant = processed[:, (h - split_h):, (w - split_w):]
        elif self.roi == "q4":
            # Q4 (Lower Right Patient/Lower Left Image): Bottom 60%, Left 60%
            quadrant = processed[:, (h - split_h):, :split_w]
        
        return {
            "QuadrantPixels": quadrant, # [3, 307, 614]
            "Labels": torch.tensor(self.Labels[index], dtype=torch.float)
        }

    def __len__(self):
        return len(self.Labels)

# Training Augmentations
train_augmentations = A.Compose([
    A.ShiftScaleRotate(scale_limit=0,
                       rotate_limit=[-3, 3],
                       border_mode=cv2.BORDER_CONSTANT,       
                       shift_limit_x = (-0.05, 0.05),  
                       shift_limit_y = (-0.02, 0.02), 
                       p = 0.7),
    A.ShiftScaleRotate(shift_range = [0, 0],
                       scale_range=[0.5, 0.5],
                       rotate_range=[0, 0],
                       interpolation=1,
                       border_mode=cv2.BORDER_CONSTANT,
                       p = 1),
    A.CLAHE(clip_limit=[1, 4], tile_grid_size=[13, 26], p = 0.6),
    A.GridDistortion(num_steps = 10, distort_range = (-0.4, 0.4), border_mode = cv2.BORDER_CONSTANT, value = [232, 232, 232], p = 0.5),
    A.ElasticTransform(alpha = 300, sigma = 10, p = 0.1),
    A.Resize(height = 512, width = 1024, p = 1.0),
    A.Normalize(normalization='min_max', 
                mean=(0.5,), std=(0.5,), 
                p = 1.0),
    A.ToTensorV2(p = 1.0)
])

# Validation Dataset Class
class MissingTeethValDM(Dataset):
    def __init__(self, df, roi, transform=None, split_pct=0.6):
        self.df = df
        self.ImagePath = self.df["ImagePath"]
        self.roi = roi
        self.split_pct = split_pct
        
        # Mapping ROI to FDI tooth numbering
        if self.roi == "q1":
            self.target_names = ['17', '16', '15', '14', '13', '12', '11']
        elif self.roi == "q2":
            self.target_names = ['21', '22', '23', '24', '25', '26', '27']
        elif self.roi == "q3":
            self.target_names = ['37', '36', '35', '34', '33', '32', '31']
        elif self.roi == "q4":
            self.target_names = ['41', '42', '43', '44', '45', '46', '47']
    
        self.Labels = self.df[self.target_names].values.tolist()
        self.transform = transform
        
        self.base_transform = A.Compose([
            A.Resize(height=512, width=1024),
            A.Normalize(normalization='min_max', mean=(0.5,), std=(0.5,), p=1.0),
            A.ToTensorV2(p=1.0)
        ])
        
    def __getitem__(self, index):
        image_path = self.ImagePath[index]
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 1. Transform full image
        if self.transform:
            processed = self.transform(image=image)['image']
        else:
            processed = self.base_transform(image=image)['image']
                        
        # 2. Calculate dynamic split points
        h, w = processed.shape[1], processed.shape[2]
        split_h = int(h * self.split_pct) # 512 * 0.6 = 307
        split_w = int(w * self.split_pct) # 1024 * 0.6 = 614
        
        # 3. Quadrant Slicing Logic
        if self.roi == "q1":
            # Q1 (Upper Right Patient/Upper Left Image): Top 60%, Left 60%
            quadrant = processed[:, :split_h, :split_w]
        elif self.roi == "q2":
            # Q2 (Upper Left Patient/Upper Right Image): Top 60%, Right 60%
            quadrant = processed[:, :split_h, (w - split_w):]
        elif self.roi == "q3":
            # Q3 (Lower Left Patient/Lower Right Image): Bottom 60%, Right 60%
            quadrant = processed[:, (h - split_h):, (w - split_w):]
        elif self.roi == "q4":
            # Q4 (Lower Right Patient/Lower Left Image): Bottom 60%, Left 60%
            quadrant = processed[:, (h - split_h):, :split_w]
        
        return {
            "QuadrantPixels": quadrant, # [3, 307, 614]
            "Labels": torch.tensor(self.Labels[index], dtype=torch.float)
        }

    def __len__(self):
        return len(self.Labels)

# Training Augmentations
validation_augmentations = A.Compose([
    A.ShiftScaleRotate(shift_limit = 0,
                       scale_limit=[0.5, 0.5],
                       rotate_limit=0,
                       interpolation=1,
                       border_mode=cv2.BORDER_CONSTANT,
                       p = 1),
    A.Resize(height = 512, width = 1024, p = 1.0),
    A.Normalize(normalization='min_max', 
                mean=(0.5,), std=(0.5,), 
                p = 1.0),
    A.ToTensorV2(p = 1.0)
])
