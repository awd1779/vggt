# train_baseline.py
#
# A dedicated script to fine-tune the original VGGT model with LoRA on a
# ScanNet scene. This creates the essential "geometric baseline" for comparison
# against your Monocular2Map model. It focuses purely on the depth prediction task.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import sys
import json
import gc

# --- Prerequisite ---
# Ensure you have peft installed:
# pip install peft
try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have 'peft' installed (`pip install peft`).")
    sys.exit(1)

# Add parent directory to path to find vggt library
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from vggt.models.vggt import VGGT

# =============================================================================
# Simplified Dataset for Geometric Training
# =============================================================================

class GeometricScanNetDataset(Dataset):
    """A simplified dataset that loads only images and depth maps for a scene."""
    def __init__(self, scene_path, num_frames=4, img_height=336, img_width=518):
        self.scene_path = scene_path
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        
        self.color_dir = os.path.join(scene_path, 'color')
        self.depth_dir = os.path.join(scene_path, 'depth')
        
        self.frame_files = sorted(os.listdir(self.color_dir), key=lambda f: int(f.split('.')[0]))
        print(f"--- Initialized GeometricScanNetDataset for scene: {os.path.basename(scene_path)} ---")
        print(f"Found {len(self.frame_files)} frames.")

    def __len__(self):
        return len(self.frame_files) - self.num_frames + 1

    def __getitem__(self, idx):
        frame_sequence_files = self.frame_files[idx : idx + self.num_frames]
        
        images = []
        depths = []
        
        for frame_file in frame_sequence_files:
            img_path = os.path.join(self.color_dir, frame_file)
            image = Image.open(img_path).convert('RGB')
            
            depth_filename = frame_file.replace('.jpg', '.png')
            depth_path = os.path.join(self.depth_dir, depth_filename)
            depth = Image.open(depth_path)
            
            # Preprocess
            image_tensor = TF.to_tensor(image)
            image_tensor = TF.resize(image_tensor, [self.img_height, self.img_width], antialias=True)
            
            depth_np = np.array(depth, dtype=np.float32) / 1000.0
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            depth_tensor = TF.resize(depth_tensor, [self.img_height, self.img_width], antialias=True)
            
            images.append(image_tensor)
            depths.append(depth_tensor)

        return {"images": torch.stack(images), "depths": torch.stack(depths)}

# =============================================================================
# Main Baseline Training Script
# =============================================================================

def main():
    print("--- Phase A (Final): Fine-tuning the Baseline VGGT with LoRA ---")
    
    # --- Configuration ---
    PRETRAINED_VGGT_PATH = "/home/ubuntu/Downloads/model.pt"
    # Use the same scene as your main training for a fair comparison
    SCANNET_SCENE_PATH = "/home/ubuntu/vggt/training/scans/scene0684_01/processed"
    # This output directory will hold the baseline adapters
    OUTPUT_DIR = "./lora_adapters_baseline_final"
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10 # Shorter training may be sufficient for geometry-only
    BATCH_SIZE = 1
    WEIGHT_DECAY = 0.01

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Setup Dataset and Dataloader ---
    try:
        dataset = GeometricScanNetDataset(scene_path=SCANNET_SCENE_PATH)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    except Exception as e:
        print(f"\nERROR: Could not initialize dataset: {e}"); return

    # --- 2. Instantiate and Setup Model ---
    model = VGGT().to(device)

    if os.path.exists(PRETRAINED_VGGT_PATH):
        state_dict = torch.load(PRETRAINED_VGGT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pre-trained VGGT weights.")
    else:
        print(f"WARNING: VGGT checkpoint not found at {PRETRAINED_VGGT_PATH}.")

    print("Applying LoRA to the VGGT backbone...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none")
    lora_model = get_peft_model(model, lora_config)
    print("LoRA applied successfully. Trainable parameters overview:")
    lora_model.print_trainable_parameters()

    # --- 3. Setup Optimizer and Loss Functions ---
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader) * NUM_EPOCHS, eta_min=1e-7)
    loss_fn = nn.L1Loss()
    
    # --- 4. The Training Loop ---
    print("\n--- Starting Baseline Fine-Tuning Loop ---")
    lora_model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0).to(device)
            gt_depths = batch["depths"].squeeze(0).to(device)
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                predictions = lora_model(images)
                
                # Model output is batched, so squeeze batch dim before loss
                predicted_depth = predictions["depth"].squeeze(0)
                loss = loss_fn(predicted_depth, gt_depths)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            if i % 10 == 0:
                print(f"E[{epoch+1}/{NUM_EPOCHS}] S[{i+1}/{len(data_loader)}], "
                      f"Loss: {loss.item():.4f}")
            
            gc.collect()
            if device == 'cuda': torch.cuda.empty_cache()

        print(f"--- End of Epoch {epoch+1}, Avg Loss: {total_loss / len(data_loader):.4f} ---")

    # --- 5. Save Final Trained Adapters ---
    lora_model.save_pretrained(OUTPUT_DIR)
    print(f"\nBaseline LoRA adapters saved to '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
