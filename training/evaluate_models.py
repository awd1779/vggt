# evaluate_models.py
# This script loads the fine-tuned baseline and Monocular2Map models,
# runs them on a validation set, and compares their performance.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os
import sys
import gc
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms.functional as TF

# Add the parent directory to the path to allow Python to find the vggt package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from vggt.models.vggt import VGGT
from peft import PeftModel

# =========================================================================================
# --- Re-define the Custom Architecture and Dataset ---
# We need these class definitions to reconstruct the Monocular2Map model and load real data.
# =========================================================================================

class ObjectEncoder(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        resnet = torchvision.models.resnet18() # No need for pre-trained weights during inference
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)
    def forward(self, x):
        return self.projection(self.encoder(x).flatten(1))

class Monocular2Map(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = VGGT()
        self.object_encoder = ObjectEncoder(output_dim=self.backbone.aggregator.patch_embed.embed_dim)
    def forward(self, images):
        # In evaluation mode, the forward pass is simpler. We just want predictions.
        return self.backbone(images)

class ValidationScanNetDataset(torch.utils.data.Dataset):
    def __init__(self, scene_path, num_frames=4, img_height=336, img_width=518):
        self.scene_path = scene_path
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        
        self.color_dir = os.path.join(scene_path, 'color')
        self.depth_dir = os.path.join(scene_path, 'depth')
        # In a real experiment, you'd add a path to ground truth semantic labels here
        
        self.frame_files = sorted(os.listdir(self.color_dir), key=lambda f: int(f.split('.')[0]))
        # Use a subset of frames for validation to avoid overlap with training
        self.frame_files = self.frame_files[-100:] # e.g., use the last 100 frames
        
        print(f"--- Initialized ValidationScanNetDataset for scene: {os.path.basename(scene_path)} ---")
        print(f"Found {len(self.frame_files)} validation frames.")

    def __len__(self):
        return len(self.frame_files) - self.num_frames + 1

    def __getitem__(self, idx):
        frame_sequence_files = self.frame_files[idx : idx + self.num_frames]
        images = []
        gt_depths = []
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
            gt_depths.append(depth_tensor)

        images_tensor = TF.resize(torch.stack(images), [self.img_height, self.img_width], antialias=True)
        gt_depths_tensor = TF.resize(torch.stack(gt_depths), [self.img_height, self.img_width], antialias=True)
        
        # We also need to return ground truth semantic labels for evaluation
        # For now, we'll return dummy labels.
        gt_labels_tensor = torch.randint(0, 11, (self.num_frames * 5,)) # 5 dummy objects per frame
        
        batch = {"images": images_tensor, "gt_depths": gt_depths_tensor, "gt_labels": gt_labels_tensor}
        return batch

# =========================================================================================
# --- Placeholder Metric Calculation Functions ---
# =========================================================================================

def calculate_geometric_loss(pred_depth, gt_depth):
    # This serves as our placeholder for geometric metrics like Chamfer Distance or ATE.
    # Lower is better.
    return torch.nn.functional.l1_loss(pred_depth, gt_depth).item()

def calculate_semantic_accuracy(model, images, gt_labels):
    # This placeholder simulates evaluating semantic accuracy.
    # In a real test, you'd compare the model's segmentation output to ground truth masks.
    # Higher is better.
    
    # We can't evaluate the baseline semantically as it has no object encoder.
    if not isinstance(model, Monocular2Map):
        return 0.0
        
    # Simulate getting object tokens and classifying them
    dummy_crops = torch.rand(gt_labels.shape[0], 3, 224, 224).to(images.device)
    object_tokens = model.object_encoder(dummy_crops)
    dummy_classifier = nn.Linear(1024, 11).to(images.device)
    logits = dummy_classifier(object_tokens)
    preds = torch.argmax(logits, dim=-1)
    accuracy = (preds == gt_labels).float().mean().item()
    return accuracy

# =========================================================================================
# --- Main Evaluation Script ---
# =========================================================================================

@torch.no_grad() # Disable gradient calculations for the entire function
def main():
    print("--- Starting Validation and Comparison ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Paths to Models and Data ---
    # !! IMPORTANT: Update these paths !!
    SCANNET_SCENE_PATH = "/home/ubuntu/Downloads/scans/scene0684_01/processed"
    BASELINE_ADAPTER_PATH = "./lora_adapters_baseline_real_data"
    M2M_ADAPTER_PATH = "./lora_adapters_monocular2map"
    OBJECT_ENCODER_PATH = "./object_encoder.pth"

    # --- Load Baseline Fine-tuned Model ---
    print("\nLoading Baseline VGGT + LoRA Adapters...")
    base_vggt = VGGT()
    baseline_model = PeftModel.from_pretrained(base_vggt, BASELINE_ADAPTER_PATH).to(device)
    baseline_model.eval()
    print("Baseline model loaded.")

    # --- Load Monocular2Map Fine-tuned Model ---
    print("\nLoading Monocular2Map + LoRA + ObjectEncoder...")
    m2m_model = Monocular2Map().to(device)
    m2m_model.backbone = PeftModel.from_pretrained(m2m_model.backbone, M2M_ADAPTER_PATH)
    m2m_model.object_encoder.load_state_dict(torch.load(OBJECT_ENCODER_PATH))
    m2m_model.eval()
    print("Monocular2Map model loaded.")

    # --- Prepare Validation Data ---
    # --- THE FIX IS HERE ---
    # Corrected the class name from ValidationDataset to ValidationScanNetDataset
    val_dataset = ValidationScanNetDataset(scene_path=SCANNET_SCENE_PATH)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # --- Run Evaluation Loop ---
    baseline_metrics = {'geom_loss': []}
    m2m_metrics = {'geom_loss': [], 'semantic_acc': []}

    for i, batch in enumerate(val_loader):
        print(f"Processing validation sequence {i+1}/{len(val_loader)}...")
        images = batch["images"].squeeze(0).to(device)
        gt_depths = batch["gt_depths"].squeeze(0).to(device)
        gt_labels = batch["gt_labels"].squeeze(0).to(device)

        # Evaluate Baseline Model
        baseline_preds = baseline_model(images)
        geom_loss_base = calculate_geometric_loss(baseline_preds['depth'].squeeze(), gt_depths.squeeze())
        baseline_metrics['geom_loss'].append(geom_loss_base)

        # Evaluate Monocular2Map Model
        m2m_preds = m2m_model(images)
        geom_loss_m2m = calculate_geometric_loss(m2m_preds['depth'].squeeze(), gt_depths.squeeze())
        semantic_acc_m2m = calculate_semantic_accuracy(m2m_model, images, gt_labels)
        m2m_metrics['geom_loss'].append(geom_loss_m2m)
        m2m_metrics['semantic_acc'].append(semantic_acc_m2m)

    # --- Print Final Results Table ---
    avg_geom_loss_base = np.mean(baseline_metrics['geom_loss'])
    avg_geom_loss_m2m = np.mean(m2m_metrics['geom_loss'])
    avg_semantic_acc_m2m = np.mean(m2m_metrics['semantic_acc'])

    print("\n\n" + "="*50)
    print(" " * 15 + "EVALUATION RESULTS")
    print("="*50)
    print(f"| {'Model':<25} | {'Avg. Geometric Loss (↓)':<25} |")
    print("-"*50)
    print(f"| {'Baseline VGGT (LoRA)':<25} | {avg_geom_loss_base:<25.4f} |")
    print(f"| {'Monocular2Map (Ours)':<25} | {avg_geom_loss_m2m:<25.4f} |")
    print("="*50 + "\n")

    print("="*50)
    print(f"| {'Model':<25} | {'Semantic Accuracy (%) (↑)':<25} |")
    print("-"*50)
    print(f"| {'Baseline VGGT (LoRA)':<25} | {'N/A':<25} |")
    print(f"| {'Monocular2Map (Ours)':<25} | {avg_semantic_acc_m2m * 100:<25.2f} |")
    print("="*50)
    print("\nValidation complete.")

if __name__ == "__main__":
    main()
