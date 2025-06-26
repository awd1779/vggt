# train.py
#
# Main training script for the Monocular2Map project.
# This script imports the dataset and model from other files and runs the
# fine-tuning process.
#
# FIX: Added a learning rate scheduler and adjusted learning rates
# to prevent overfitting of the semantic head.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import sys
import json
import gc

# --- Local Imports ---
from dataset import RealScanNetSemanticDataset
from model import Monocular2Map

# --- Prerequisite ---
# Ensure you have peft installed:
# pip install peft
try:
    from peft import LoraConfig, get_peft_model
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have 'peft' installed (`pip install peft`).")
    sys.exit(1)


def main():
    print("--- Phase C (Final): Fine-tuning Monocular2Map with Real Data and Semantic Logic ---")
    
    # --- Configuration ---
    # !! IMPORTANT: Update these paths to your local environment !!
    PRETRAINED_VGGT_PATH = "/home/ubuntu/Downloads/model.pt"
    SCANNET_SCENE_PATH = "/home/ubuntu/Downloads/scans/scene0684_01/" 
    SCANNET_SCENE_ID = "scene0684_01"
    OUTPUT_DIR = "./final_monocular2map_output"
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 20 # Increased epochs for scheduler to work effectively
    BATCH_SIZE = 1
    GEOM_LOSS_WEIGHT = 1.0
    SEMANTIC_LOSS_WEIGHT = 0.5
    WEIGHT_DECAY = 0.01 # Added weight decay for regularization

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Setup Dataset and Dataloader ---
    try:
        dataset = RealScanNetSemanticDataset(
            scene_path=SCANNET_SCENE_PATH, 
            scene_id=SCANNET_SCENE_ID
        )
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        num_classes = dataset.get_num_classes()
        if num_classes == 0:
            print("\nERROR: No classes were found. Check annotation files and parsing logic.")
            return
    except (FileNotFoundError, IndexError, json.JSONDecodeError) as e:
        print(f"\nERROR: Could not initialize dataset. Please check paths and file formats.")
        print(f"Details: {e}")
        return

    # --- 2. Instantiate and Setup Model ---
    model = Monocular2Map(num_classes=num_classes).to(device)

    if os.path.exists(PRETRAINED_VGGT_PATH):
        state_dict = torch.load(PRETRAINED_VGGT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.backbone.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pre-trained VGGT weights into the backbone.")
    else:
        print(f"WARNING: VGGT checkpoint not found at {PRETRAINED_VGGT_PATH}.")

    print("Applying LoRA to the VGGT backbone...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none")
    model.backbone = get_peft_model(model.backbone, lora_config)
    print("LoRA applied successfully. Trainable parameters overview:")
    model.backbone.print_trainable_parameters()

    # --- 3. Setup Optimizer and Loss Functions ---
    # FIX: Reduced the learning rate multiplier for new components to prevent rapid overfitting.
    trainable_params = [
        {'params': model.backbone.parameters()},
        {'params': model.object_encoder.parameters(), 'lr': LEARNING_RATE * 2},
        {'params': model.semantic_classifier.parameters(), 'lr': LEARNING_RATE * 2}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # FIX: Added a learning rate scheduler to help prevent overfitting.
    scheduler = CosineAnnealingLR(optimizer, T_max=len(data_loader) * NUM_EPOCHS, eta_min=1e-7)

    geom_loss_fn = nn.L1Loss()
    semantic_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"Optimizer and Scheduler configured.")

    # --- 4. The Final Training Loop ---
    print("\n--- Starting Final Fine-Tuning Loop for Monocular2Map ---")
    model.train()

    for epoch in range(NUM_EPOCHS):
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0).to(device)
            masks = batch["masks"].squeeze(0).to(device)
            gt_depths = batch["depths"].squeeze(0).to(device)
            gt_labels_packed = batch["gt_labels"].squeeze(0)
            
            valid_gt_labels = gt_labels_packed.flatten()[gt_labels_packed.flatten() != -100].to(device)
            
            with torch.cuda.amp.autocast(enabled=(device=='cuda')):
                geom_predictions, semantic_logits = model(images, masks)
                
                predicted_depth = geom_predictions["depth"].squeeze()
                geom_loss = geom_loss_fn(predicted_depth, gt_depths.squeeze())
                
                semantic_loss = torch.tensor(0.0, device=device)
                if semantic_logits is not None and len(valid_gt_labels) > 0:
                    if semantic_logits.shape[0] == valid_gt_labels.shape[0]:
                         semantic_loss = semantic_loss_fn(semantic_logits, valid_gt_labels)
                    else:
                        print(f"Warning: Mismatch logits ({semantic_logits.shape[0]}) vs labels ({valid_gt_labels.shape[0]}). Skipping step.")
                
                total_loss = (GEOM_LOSS_WEIGHT * geom_loss) + (SEMANTIC_LOSS_WEIGHT * semantic_loss)
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step() # Step the scheduler after each optimizer step

            if i % 10 == 0: # Print less frequently to clean up logs
                lr = scheduler.get_last_lr()[0]
                print(f"E[{epoch+1}/{NUM_EPOCHS}] S[{i+1}/{len(data_loader)}], "
                      f"LR: {lr:.2e}, "
                      f"G_Loss: {geom_loss.item():.4f}, S_Loss: {semantic_loss.item():.4f}, "
                      f"Total: {total_loss.item():.4f}")
            
            gc.collect()
            if device == 'cuda':
                torch.cuda.empty_cache()

    print("\n--- Monocular2Map Final Fine-Tuning Finished ---")
    
    # --- 5. Save Final Trained Models ---
    lora_adapter_path = os.path.join(OUTPUT_DIR, "lora_adapters_monocular2map_final")
    object_encoder_path = os.path.join(OUTPUT_DIR, "object_encoder_final.pth")
    classifier_path = os.path.join(OUTPUT_DIR, "semantic_classifier_final.pth")

    model.backbone.save_pretrained(lora_adapter_path)
    torch.save(model.object_encoder.state_dict(), object_encoder_path)
    torch.save(model.semantic_classifier.state_dict(), classifier_path)
    
    print(f"Trained LoRA adapters saved to '{lora_adapter_path}'")
    print(f"Trained ObjectEncoder saved to '{object_encoder_path}'")
    print(f"Trained SemanticClassifier saved to '{classifier_path}'")


if __name__ == "__main__":
    main()
