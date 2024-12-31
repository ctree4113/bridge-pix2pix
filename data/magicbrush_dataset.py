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
        """初始化MagicBrush数据集
        
        Args:
            root_dir: 数据集根目录
            split: 'train' 或 'dev'
            min_resize_res: 最小调整大小
            max_resize_res: 最大调整大小
            crop_res: 裁剪大小
            flip_prob: 水平翻转概率
            transform: 自定义变换
        """
        super().__init__()
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
        mask_img = self._load_image(item['mask_img'])
        
        # 应用图像处理
        if self.custom_transform is None:
            # 随机调整大小
            resize_res = torch.randint(self.min_resize_res, self.max_resize_res + 1, ()).item()
            source_img = source_img.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
            target_img = target_img.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
            mask_img = mask_img.resize((resize_res, resize_res), Image.Resampling.LANCZOS)
            
            # 转换为tensor并归一化
            source_img = rearrange(2 * torch.tensor(np.array(source_img)).float() / 255 - 1, "h w c -> c h w")
            target_img = rearrange(2 * torch.tensor(np.array(target_img)).float() / 255 - 1, "h w c -> c h w")
            mask_img = rearrange(torch.tensor(np.array(mask_img)).float() / 255, "h w c -> c h w")
            
            # 随机裁剪和翻转
            crop = torchvision.transforms.RandomCrop(self.crop_res)
            flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
            
            # 同时应用变换
            imgs = torch.cat((source_img, target_img, mask_img))
            imgs = flip(crop(imgs))
            source_img, target_img, mask_img = imgs.chunk(3)
        else:
            source_img = self.custom_transform(source_img)
            target_img = self.custom_transform(target_img)
            mask_img = self.custom_transform(mask_img)

        # 处理文本指令
        instruction = item['instruction']
        if not instruction:
            instruction = ""

        # 返回完整信息
        return {
            "edited": target_img,
            "edit": {
                "c_concat": source_img,
                "c_crossattn": instruction,
                "mask": mask_img,
                "meta": {
                    "img_id": item["img_id"],
                    "turn_index": item["turn_index"]
                }
            }
        }
    
    def _load_image(self, img_data):
        """统一的图像加载函数"""
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            return Image.fromarray(img_data)
        return img_data