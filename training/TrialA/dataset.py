# dataset.py
#
# Contains the data loading and preprocessing logic for the ScanNet dataset.

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
import json
import sys

# --- Prerequisite ---
# Ensure you have plyfile installed:
# pip install plyfile
try:
    from plyfile import PlyData
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have 'plyfile' installed (`pip install plyfile`).")
    sys.exit(1)


class RealScanNetSemanticDataset(Dataset):
    """
    Loads real images, depths, instance segmentation masks, and semantic labels from a 
    processed ScanNet scene.
    """
    def __init__(self, scene_path, scene_id, num_frames=4, max_objects_per_frame=15, img_height=336, img_width=518):
        assert img_height % 14 == 0 and img_width % 14 == 0, "Image dimensions must be a multiple of 14."
        
        self.scene_path = scene_path
        self.scene_id = scene_id
        self.num_frames = num_frames
        self.max_objects = max_objects_per_frame
        self.img_height = img_height
        self.img_width = img_width
        
        # --- Define directory and file paths ---
        self.color_dir = os.path.join(scene_path, 'color')
        self.depth_dir = os.path.join(scene_path, 'depth')
        self.instance_dir = os.path.join(scene_path, 'instance-filt')
        self.aggregation_json_path = os.path.join(scene_path, f'{scene_id}.aggregation.json')
        
        self.frame_files = sorted(os.listdir(self.color_dir), key=lambda f: int(f.split('.')[0]))
        
        # --- Pre-load and parse annotation files to create mappings ---
        self._check_paths()
        # The only mapping needed comes from the aggregation file.
        self.instance_to_label_map = self._create_instance_label_mapping()
        
        print(f"--- Initialized REAL ScanNetSemanticDataset for scene: {scene_id} ---")
        print(f"Found {len(self.frame_files)} frames. Using max {self.max_objects} objects per frame.")

    def _check_paths(self):
        """Checks if all necessary files and directories exist."""
        paths_to_check = [self.color_dir, self.depth_dir, self.instance_dir, self.aggregation_json_path]
        for p in paths_to_check:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file or directory not found: {p}")
        
    def _create_instance_label_mapping(self):
        """
        Creates the final mapping from an instance ID (from mask files) to a 
        semantic class label ID using the aggregation.json file.
        """
        with open(self.aggregation_json_path) as f:
            agg_data = json.load(f)
        
        instance_map = {}
        for group in agg_data['segGroups']:
            instance_id = group['objectId'] + 1
            label_name = group.get('label', 'unannotated')
            instance_map[instance_id] = label_name

        unique_labels = sorted(list(set(instance_map.values()) - {'unannotated'}))
        self.class_names = unique_labels
        self.class_to_idx = {name: i for i, name in enumerate(unique_labels)}
        print(f"Found {len(self.class_names)} unique classes in the scene: {self.class_names}")

        final_map = {inst_id: self.class_to_idx.get(label, -1) for inst_id, label in instance_map.items()}
        return final_map

    def __len__(self):
        return len(self.frame_files) - self.num_frames + 1

    def __getitem__(self, idx):
        frame_sequence_files = self.frame_files[idx : idx + self.num_frames]
        
        images_seq, depths_seq, masks_seq, labels_seq = [], [], [], []

        for frame_file in frame_sequence_files:
            img_path = os.path.join(self.color_dir, frame_file)
            image = Image.open(img_path).convert('RGB')
            
            depth_filename = frame_file.replace('.jpg', '.png')
            depth_path = os.path.join(self.depth_dir, depth_filename)
            depth = Image.open(depth_path)
            
            instance_filename = frame_file.replace('.jpg', '.png')
            instance_path = os.path.join(self.instance_dir, instance_filename)
            instance_img = Image.open(instance_path)

            image_tensor = TF.to_tensor(image)
            image_tensor = TF.resize(image_tensor, [self.img_height, self.img_width], antialias=True)
            
            depth_np = np.array(depth, dtype=np.float32) / 1000.0
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            depth_tensor = TF.resize(depth_tensor, [self.img_height, self.img_width], antialias=True)
            
            instance_np = np.array(instance_img, dtype=np.int32)
            instance_tensor = torch.from_numpy(instance_np).unsqueeze(0)
            instance_tensor = TF.resize(instance_tensor, [self.img_height, self.img_width], interpolation=T.InterpolationMode.NEAREST)

            instance_ids = torch.unique(instance_tensor)
            instance_ids = instance_ids[instance_ids != 0]
            
            frame_masks, frame_labels = [], []
            for inst_id in instance_ids:
                label = self.instance_to_label_map.get(inst_id.item(), -1)
                if label != -1:
                    mask = (instance_tensor == inst_id)
                    frame_masks.append(mask)
                    frame_labels.append(label)

            num_found_objects = len(frame_masks)
            if num_found_objects > 0:
                padded_masks = torch.cat(frame_masks, dim=0)
                padded_labels = torch.tensor(frame_labels, dtype=torch.long)

                if num_found_objects < self.max_objects:
                    pad_size = self.max_objects - num_found_objects
                    empty_mask = torch.zeros(pad_size, self.img_height, self.img_width, dtype=torch.bool)
                    padded_masks = torch.cat([padded_masks, empty_mask], dim=0)
                    pad_labels_tensor = torch.full((pad_size,), -100, dtype=torch.long)
                    padded_labels = torch.cat([padded_labels, pad_labels_tensor], dim=0)
                elif num_found_objects > self.max_objects:
                    padded_masks = padded_masks[:self.max_objects]
                    padded_labels = padded_labels[:self.max_objects]
            else:
                padded_masks = torch.zeros(self.max_objects, self.img_height, self.img_width, dtype=torch.bool)
                padded_labels = torch.full((self.max_objects,), -100, dtype=torch.long)

            images_seq.append(image_tensor)
            depths_seq.append(depth_tensor)
            masks_seq.append(padded_masks)
            labels_seq.append(padded_labels)

        return {
            "images": torch.stack(images_seq),
            "depths": torch.stack(depths_seq),
            "masks": torch.stack(masks_seq),
            "gt_labels": torch.stack(labels_seq)
        }

    def get_num_classes(self):
        return len(self.class_names)
