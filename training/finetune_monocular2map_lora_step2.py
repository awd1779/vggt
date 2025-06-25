# finetune_monocular2map_lora.py
# This version includes the fix for the channel mismatch error.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import os
import sys
import gc

# Add the parent directory to the path to allow Python to find the vggt package.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
from vggt.models.vggt import VGGT
from peft import LoraConfig, get_peft_model

# =========================================================================================
# --- Step 1: Define the New Architectural Components for Monocular2Map ---
# =========================================================================================

class ObjectEncoder(nn.Module):
    """
    A simple CNN to encode masked object image regions into single tokens.
    This module is new and will be trained from scratch.
    """
    def __init__(self, output_dim=1024):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.projection(features)

class Monocular2Map(nn.Module):
    """
    Your novel Monocular2Map architecture.
    """
    def __init__(self):
        super().__init__()
        print("--- Initializing Monocular2Map Architecture ---")
        self.backbone = VGGT()
        self.object_encoder = ObjectEncoder(output_dim=self.backbone.aggregator.patch_embed.embed_dim)
        print("ObjectEncoder initialized.")

    def forward(self, images, masks):
        # 1. Get the standard geometric predictions from the VGGT backbone
        predictions = self.backbone(images)
        
        # 2. Simulate object processing to include the object_encoder in the computation graph.
        # --- THE FIX IS HERE ---
        # We correctly select the FIRST IMAGE from the sequence (shape: 3, H, W),
        # crop it, and then add a batch dimension (unsqueeze(0)) for the encoder.
        first_image_in_sequence = images[0]
        dummy_object_crop = torchvision.transforms.functional.center_crop(first_image_in_sequence, [224, 224])
        object_tokens = self.object_encoder(dummy_object_crop.unsqueeze(0))
        
        # 3. Create a dummy loss for the object encoder.
        dummy_obj_loss = object_tokens.mean() * 0.0
        
        return predictions, dummy_obj_loss

# --- A Simple Dataset that now also provides dummy masks ---
class SemanticScanNetDataset(torch.utils.data.Dataset):
    def __init__(self, num_frames=4, img_height=336, img_width=518):
        self.num_frames = num_frames
        self.img_height = img_height
        self.img_width = img_width
        print(f"--- Using SemanticScanNetDataset (Dummy Data) with dimensions: {img_height}x{img_width} ---")

    def __len__(self):
        return 10
        
    def __getitem__(self, idx):
        S, H, W = self.num_frames, self.img_height, self.img_width
        batch = {
            "images": torch.rand(S, 3, H, W),
            "depths": torch.rand(S, 1, H, W),
            "masks": torch.randint(0, 2, (S, 5, H, W), dtype=torch.bool) 
        }
        return batch

def main():
    print("--- Phase B: Fine-tuning the Monocular2Map Architecture with LoRA ---")

    PRETRAINED_CHECKPOINT_PATH = "/home/ubuntu/Downloads/model.pt"
    
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 5
    BATCH_SIZE = 1
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Step 1: Instantiate Your Custom Monocular2Map Model ---
    model = Monocular2Map().to(device)

    # --- Step 2: Load Pre-trained Weights into the Backbone ---
    if os.path.exists(PRETRAINED_CHECKPOINT_PATH):
        state_dict = torch.load(PRETRAINED_CHECKPOINT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.backbone.load_state_dict(state_dict, strict=False)
        print("Pre-trained VGGT weights loaded into Monocular2Map backbone.")
    else:
        print(f"WARNING: Pre-trained checkpoint not found. Backbone is randomly initialized.")

    # --- Step 3: Apply LoRA to the VGGT Backbone ---
    print("Applying LoRA to the VGGT backbone...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"])
    model.backbone = get_peft_model(model.backbone, lora_config)
    print("LoRA applied successfully to backbone. Parameter overview:")
    model.backbone.print_trainable_parameters()

    # --- Step 4: Setup Dataset and Optimizer for ALL Trainable Parts ---
    dataset = SemanticScanNetDataset()
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    
    trainable_params = list(model.backbone.parameters()) + list(model.object_encoder.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    print("Optimizer configured to train LoRA adapters and the new ObjectEncoder.")
    
    # --- Step 5: The Training Loop ---
    print("\n--- Starting Fine-Tuning Loop for Monocular2Map ---")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_loss_epoch = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0).to(device)
            masks = batch["masks"].squeeze(0).to(device)
            gt_depths = batch["depths"].squeeze(0).to(device)
            
            with torch.cuda.amp.autocast():
                predictions, dummy_obj_loss = model(images, masks)
                
                predicted_depth = predictions["depth"].squeeze()
                gt_depths_squeezed = gt_depths.squeeze()
                geom_loss = torch.nn.functional.l1_loss(predicted_depth, gt_depths_squeezed)
                
                total_loss_step = geom_loss + dummy_obj_loss
            
            total_loss_step.backward()
            optimizer.step()
            total_loss_epoch += total_loss_step.item()
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], Loss: {total_loss_step.item():.4f}")
            
        print(f"--- End of Epoch {epoch+1}, Average Loss: {total_loss_epoch / len(data_loader):.4f} ---\n")

    print("--- Monocular2Map Fine-Tuning Finished ---")
    
    model.backbone.save_pretrained("./lora_adapters_monocular2map")
    torch.save(model.object_encoder.state_dict(), "./object_encoder.pth")
    print("Trained LoRA adapters saved to './lora_adapters_monocular2map'")
    print("Trained ObjectEncoder saved to './object_encoder.pth'")

if __name__ == "__main__":
    main()
