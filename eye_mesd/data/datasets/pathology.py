import logging
import os
from typing import Callable, Optional, Tuple
import numpy as np
from .extended import ExtendedVisionDataset
from PIL import Image
import json
import cv2

logger = logging.getLogger("eye_mesd")

class PathologyDataset(ExtendedVisionDataset):
    def __init__(self,
                 root: str = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None) -> None:
        # root 指向你的 JSON 文件路径
        super().__init__(root, None, transform, target_transform)
        
        self.image_paths = self.get_all_files(root)
        self.transformers = transform
        
        # 尝试读取第一张图作为“备用图”，防止训练中因某张图损坏而中断
        try:
            first_path = self.image_paths[0]
            # 兼容性检查：如果是 OpenCV 读取
            tmp_img = cv2.imread(first_path)
            if tmp_img is not None:
                self.image_patch = tmp_img[..., ::-1] # BGR to RGB
            else:
                # 如果第一张图就读不到，创建一个白色块作为兜底
                self.image_patch = np.ones((224, 224, 3), dtype=np.uint8) * 255
        except Exception as e:
            logger.warning(f"无法初始化备用图: {e}")
            self.image_patch = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        # 创建一个记录损坏图片的文件，方便后续清理数据
        self.file_handle = open('./failed_paths.txt', 'a')
    
    def get_all_files(self, root):
        """解析 JSON 文件，支持列表 [path1, path2] 或 字典 {path1: label1}"""
        with open(root, 'r') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            return list(data.keys())
        else:
            raise ValueError(f"不支持的 JSON 格式: {type(data)}")
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tuple[any, int]:
        """返回 (图像, 标签)，标签默认为 0"""
        p = self.image_paths[index]
        
        try:
            # 使用 PIL 加载并确保是 RGB 模式
            img = Image.open(p).convert('RGB')
            if self.transformers is not None:
                img = self.transformers(img)
        except Exception:
            # 如果某张图读取失败（比如路径坏了），使用备用图代替，不中断训练
            img = Image.fromarray(self.image_patch.copy())
            if self.transformers is not None:
                img = self.transformers(img)
            
            # 记录失败路径以便后续检查
            self.file_handle.write(p + '\n')
            self.file_handle.flush()
            
        # 必须返回 0 而非 None，否则分布式训练的 collate_fn 会报错
        return img, 0