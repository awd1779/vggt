# finetune_baseline_lora_final.py
# This version fixes the tensor shape mismatch for the loss calculation.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

# =========================================================================================
# --- Step 1: Define a Real ScanNet Dataset Loader ---
# =========================================================================================

class ScanNetSceneDataset(torch.utils.data.Dataset):
    """
    A simplified dataset loader for a single processed ScanNet scene.
    It loads a sequence of real images and depth maps.
    """
    def __init__(self, scene_path, num_frames=4, img_height=336, img_width=518):
        assert img_height % 14 == 0 and img_width % 14 == 0, "Image dimensions must be a multiple of 14."
        
        self.scene_path = scene_path
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        
        self.color_dir = os.path.join(scene_path, 'color')
        self.depth_dir = os.path.join(scene_path, 'depth')
        
        self.frame_files = sorted(os.listdir(self.color_dir), key=lambda f: int(f.split('.')[0]))
        
        print(f"--- Initialized ScanNetSceneDataset for scene: {os.path.basename(scene_path)} ---")
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
            image_tensor = TF.to_tensor(image)
            images.append(image_tensor)
            
            depth_filename = frame_file.replace('.jpg', '.png')
            depth_path = os.path.join(self.depth_dir, depth_filename)
            depth = Image.open(depth_path)
            depth_np = np.array(depth, dtype=np.float32) / 1000.0
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            depths.append(depth_tensor)

        images_tensor = torch.stack(images)
        depths_tensor = torch.stack(depths)
        
        images_tensor = TF.resize(images_tensor, [self.img_height, self.img_width], antialias=True)
        depths_tensor = TF.resize(depths_tensor, [self.img_height, self.img_width], antialias=True)
        
        batch = {"images": images_tensor, "depths": depths_tensor}
        return batch

def main():
    print("--- Phase A (Final): Fine-tuning the Baseline VGGT with REAL Data ---")

    PRETRAINED_CHECKPOINT_PATH = "/home/ubuntu/Downloads/model.pt"
    # !! IMPORTANT: Update this path to your PROCESSED scene0000_00 folder !!
    SCANNET_SCENE_PATH = "/home/ubuntu/Downloads/scans/scene0684_01/processed" 
    
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    BATCH_SIZE = 1
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup Model ---
    model = VGGT()
    model.aggregator.patch_embed.use_gradient_checkpointing = True
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.load_state_dict(state_dict)
    else:
        print(f"WARNING: Checkpoint not found at {PRETRAINED_CHECKPOINT_PATH}. Training from scratch.")
    
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none")
    lora_model = get_peft_model(model, lora_config).to(device)
    lora_model.print_trainable_parameters()
    
    # --- Setup REAL Dataset and Optimizer ---
    if not os.path.exists(SCANNET_SCENE_PATH):
        print(f"ERROR: ScanNet scene path not found at: {SCANNET_SCENE_PATH}")
        return
        
    dataset = ScanNetSceneDataset(scene_path=SCANNET_SCENE_PATH)
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    optimizer = torch.optim.AdamW(lora_model.parameters(), lr=LEARNING_RATE)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("\n--- Starting Fine-Tuning Loop with REAL DATA ---")
    lora_model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0).to(device)
            gt_depths = batch["depths"].squeeze(0).to(device) # Shape: (S, C, H, W) -> (4, 1, 336, 518)
            
            with torch.cuda.amp.autocast():
                predictions = lora_model(images)
                predicted_depth = predictions["depth"] # Likely Shape: (1, 4, 336, 518, 1) or (4, 336, 518, 1)

                # --- THE FIX IS HERE ---
                # Squeeze out any singleton dimensions from both tensors to ensure they match.
                # This handles both (1, 4, 336, 518, 1) -> (4, 336, 518)
                # and (4, 1, 336, 518) -> (4, 336, 518)
                predicted_depth_squeezed = predicted_depth.squeeze()
                gt_depths_squeezed = gt_depths.squeeze()

                loss = torch.nn.functional.l1_loss(predicted_depth_squeezed, gt_depths_squeezed)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {loss.item():.4f}")
            
        print(f"--- End of Epoch {epoch+1}, Average Loss: {total_loss / len(data_loader):.4f} ---\n")

    lora_model.save_pretrained("./lora_adapters_baseline_real_data")
    print("Trained LoRA adapters saved to './lora_adapters_baseline_real_data'")

if __name__ == "__main__":
    main()
