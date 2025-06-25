# finetune_monocular2map_lora_phase_c.py
# This version includes the final fix for the data structure and indexing error.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
import sys
import gc
from PIL import Image
import numpy as np
import torchvision.transforms.functional as TF

# Add the parent directory to the path to allow Python to find the vggt package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from vggt.models.vggt import VGGT
from peft import LoraConfig, get_peft_model

class ObjectEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.projection(features)

class Monocular2Map(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = VGGT()
        self.object_encoder = ObjectEncoder(output_dim=self.backbone.aggregator.patch_embed.embed_dim)

    def forward(self, images, masks=None):
        # 1. Always get the geometric predictions from the backbone
        predictions = self.backbone(images)
        
        object_tokens_tensor = None
        if self.training and masks is not None:
            # 2. Generate object tokens from masks
            # In a real implementation, you would crop objects from `images` using `masks`.
            # For this script, we create a dummy tensor of the correct shape to test the pipeline.
            num_images = masks.shape[0]
            num_objects_per_image = masks.shape[1]
            total_objects = num_images * num_objects_per_image
            
            dummy_crops = torch.rand(total_objects, 3, 224, 224).to(images.device)
            object_tokens_tensor = self.object_encoder(dummy_crops)

        return predictions, object_tokens_tensor


class RealScanNetSemanticDataset(torch.utils.data.Dataset):
    def __init__(self, scene_path, num_frames=4, num_objects_per_frame=3, img_height=336, img_width=518):
        self.scene_path = scene_path
        self.num_frames = num_frames
        self.num_objects_per_frame = num_objects_per_frame
        self.img_height = img_height
        self.img_width = img_width
        print(f"--- Using RealScanNetSemanticDataset (with Dummy Masks) for scene: {scene_path} ---")

    def __len__(self): return 10
    
    def __getitem__(self, idx):
        S, H, W = self.num_frames, self.img_height, self.img_width
        O = self.num_objects_per_frame
        
        # --- THE FIX IS HERE ---
        # Instead of returning a list of tensors for masks, we stack them into a single tensor.
        # This makes data handling much more consistent and robust.
        mask_list = [torch.randint(0, 2, (O, H, W), dtype=torch.bool) for _ in range(S)]
        
        batch = {
            "images": torch.rand(S, 3, H, W),
            "depths": torch.rand(S, 1, H, W),
            "masks": torch.stack(mask_list, dim=0), # Shape: (S, O, H, W)
            "gt_labels": torch.randint(0, 11, (S * O,))
        }
        return batch

def main():
    print("--- Phase C (Final Corrected): Fine-tuning Monocular2Map with Semantic Logic ---")
    PRETRAINED_CHECKPOINT_PATH = "/home/ubuntu/Downloads/model.pt"
    SCANNET_SCENE_PATH = "/home/ubuntu/Downloads/scans/scene0684_01/processed" 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Monocular2Map().to(device)
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.backbone.load_state_dict(state_dict, strict=False)
    
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"])
    model.backbone = get_peft_model(model.backbone, lora_config)
    
    dataset = RealScanNetSemanticDataset(scene_path=SCANNET_SCENE_PATH)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=4)
    trainable_params = list(model.backbone.parameters()) + list(model.object_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=2e-5)
    
    semantic_loss_fn = nn.CrossEntropyLoss()
    dummy_classifier = nn.Linear(1024, 11).to(device)
    
    print("\n--- Starting Semantic Fine-Tuning Loop ---")
    model.train()

    for epoch in range(5):
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            # --- THE FIX IS HERE ---
            # We can now treat `masks` like any other tensor in the batch.
            images = batch["images"].squeeze(0).to(device)
            masks = batch["masks"].squeeze(0).to(device) 
            gt_depths = batch["depths"].squeeze(0).to(device)
            gt_labels = batch["gt_labels"].squeeze(0).to(device)
            
            with torch.cuda.amp.autocast():
                predictions, object_tokens = model(images, masks)
                
                geom_loss = torch.nn.functional.l1_loss(predictions["depth"].squeeze(), gt_depths.squeeze())
                
                semantic_loss = 0.0
                if object_tokens is not None:
                    object_logits = dummy_classifier(object_tokens)
                    semantic_loss = semantic_loss_fn(object_logits, gt_labels)
                
                total_loss_step = geom_loss + semantic_loss
            
            total_loss_step.backward()
            optimizer.step()
            print(f"Epoch [{epoch+1}], Step [{i+1}], Geom Loss: {geom_loss.item():.4f}, Sem Loss: {semantic_loss.item() if isinstance(semantic_loss, torch.Tensor) else 0:.4f}")
            
    print("--- Monocular2Map Semantic Fine-Tuning Finished ---")

if __name__ == "__main__":
    main()
