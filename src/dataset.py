import os
import torch
import json
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            annotation_file (string): Path to the annotation JSON file.
            transforms (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Load COCO-style JSON
        with open(annotation_file) as f:
            data = json.load(f)

        self.images = data["images"]
        self.annotations = data["annotations"]
        self.image_id_to_annotations = {}

        # Group annotations by image_id
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id not in self.image_id_to_annotations:
                self.image_id_to_annotations[img_id] = []
            self.image_id_to_annotations[img_id].append(ann)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_info = self.images[idx]
        img_id = image_info["id"]
        img_name = image_info["file_name"]  
        img_path = os.path.join(self.root_dir, img_name)

        image = Image.open(img_path).convert("RGB")

        anns = self.image_id_to_annotations.get(img_id, [])
        
        boxes = []
        labels = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann["category_id"])

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels}

        if self.transforms:
            image = self.transforms(image)

        return image, target
