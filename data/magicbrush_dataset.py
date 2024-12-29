import os
import io
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_from_disk
import torch
import numpy as np
import torchvision
from einops import rearrange

class MagicBrushDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        min_resize_res: int = 256,
        max_resize_res: int = 256,
        crop_res: int = 256,
        flip_prob: float = 0.0,
        transform=None
    ):
        """
        Args:
            root_dir (str): 数据集根目录
            split (str): 'train' 或 'dev'
            min_resize_res (int): 最小调整大小
            max_resize_res (int): 最大调整大小
            crop_res (int): 裁剪大小
            flip_prob (float): 水平翻转概率
            transform (callable, optional): 自定义变换
        """
        self.root_dir = root_dir
        self.dataset = load_from_disk(root_dir)[split]
        self.min_resize_res = min_resize_res
        self.max_resize_res = max_resize_res
        self.crop_res = crop_res
        self.flip_prob = flip_prob
        self.custom_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # 加载图片
        source_img = self._load_image(item['source_img'])
        target_img = self._load_image(item['target_img'])
        
        # 应用与EditDataset一致的图像处理
        if self.custom_transform is None:
            # 随机调整大小
            resize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
            source_img = source_img.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
            target_img = target_img.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
            
            # 转换为tensor并归一化到[-1,1]
            source_img = rearrange(2 * torch.tensor(np.array(source_img)).float() / 255 - 1, "h w c -> c h w")
            target_img = rearrange(2 * torch.tensor(np.array(target_img)).float() / 255 - 1, "h w c -> c h w")
            
            # 随机裁剪和翻转
            crop = torchvision.transforms.RandomCrop(self.crop_res)
            flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
            source_img, target_img = flip(crop(torch.cat((source_img, target_img)))).chunk(2)
        else:
            source_img = self.custom_transform(source_img)
            target_img = self.custom_transform(target_img)

        # 处理文本指令
        instruction = item['instruction']
        if not instruction:
            instruction = ""  # 如果没有指令,使用空字符串

        return {
            "edited": target_img,
            "edit": {
                "c_concat": source_img,  # 直接返回tensor
                "c_crossattn": instruction  # 返回原始文本字符串
            }
        }
    
    def _load_image(self, img_data):
        """统一的图像加载函数"""
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            return Image.fromarray(img_data)
        return img_data