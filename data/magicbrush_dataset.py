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
            
            # 转换为tensor并归一化到[-1,1]范围
            source_img = self._preprocess_image(source_img)
            target_img = self._preprocess_image(target_img)
            mask_img = self._preprocess_mask(mask_img)
            
            # 随机裁剪和翻转
            crop = torchvision.transforms.RandomCrop(self.crop_res)
            flip = torchvision.transforms.RandomHorizontalFlip(float(self.flip_prob))
            
            # 同时应用变换以保持一致性
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
            
        # 确保指令是字符串
        if isinstance(instruction, (list, tuple)):
            instruction = " ".join(instruction)
        
        # 返回 ldm 格式的数据
        return {
            "edited": target_img,  # 目标图像
            "edit": {
                "c_concat": source_img,  # 源图像
                "c_crossattn": instruction,  # 文本指令
                "mask": mask_img,  # 编辑掩码
                "meta": {
                    "img_id": item["img_id"],
                    "turn_index": item["turn_index"]
                }
            }
        }
    
    def _load_image(self, img_data):
        """统一的图像加载函数，确保输出RGB图像"""
        if isinstance(img_data, bytes):
            return Image.open(io.BytesIO(img_data)).convert('RGB')
        elif isinstance(img_data, np.ndarray):
            if img_data.ndim == 2:
                img_data = np.stack([img_data] * 3, axis=-1)
            return Image.fromarray(img_data).convert('RGB')
        return img_data.convert('RGB')
        
    def _preprocess_image(self, img):
        """图像预处理:转换为3通道tensor并归一化到[-1,1]"""
        img = torch.tensor(np.array(img)).float()
        img = rearrange(img, 'h w c -> c h w')
        img = img / 127.5 - 1.0  # 归一化到[-1,1]
        return img
        
    def _preprocess_mask(self, mask):
        """掩码预处理:转换为3通道tensor并归一化到[0,1]"""
        mask = torch.tensor(np.array(mask)).float()
        mask = rearrange(mask, 'h w c -> c h w')
        mask = mask / 255.0  # 归一化到[0,1]
        return mask