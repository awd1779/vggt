# finetune_monocular2map_lora_final.py
#
# This script represents the final implementation for Phase C of the Monocular2Map project.
# It integrates a real ScanNet data loader with ground truth segmentation and a
# complete forward pass implementing the hybrid tokenization logic.
#
# Key features:
# - RealScanNetSemanticDataset: Loads real images, depth, and instance masks.
#   Includes placeholder logic for parsing ScanNet's JSON and PLY annotation files.
# - Monocular2Map: Implements the full forward pass, generating object tokens from
#   cropped objects and calculating both geometric and semantic losses.
# - Main training loop: Jointly fine-tunes the LoRA adapters and the ObjectEncoder.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
import torchvision
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image
import numpy as np
import os
import sys
import json
import gc

# --- Prerequisite ---
# Ensure you have peft and plyfile installed:
# pip install peft plyfile

# Add the parent directory to the path to allow Python to find the vggt package.
# This assumes your script is in a 'scripts' folder and 'vggt' is in the parent.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
try:
    from vggt.models.vggt import VGGT
    from peft import LoraConfig, get_peft_model
    from plyfile import PlyData
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure you have all required packages installed and the vggt library is in your Python path.")
    sys.exit(1)


# =========================================================================================
# --- Step 1: Define the Real Semantic ScanNet Dataset ---
# =========================================================================================

class RealScanNetSemanticDataset(torch.utils.data.Dataset):
    """
    Loads real images, depths, instance segmentation masks, and semantic labels from a 
    processed ScanNet scene.
    
    This implementation assumes a specific directory structure for the processed scene.
    You may need to adjust paths based on your data preprocessing.
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
        self.instance_dir = os.path.join(scene_path, 'instance-filt') # Filtered instance masks
        
        # !! IMPORTANT: Update these filenames to match your ScanNet scene files !!
        self.aggregation_json_path = os.path.join(scene_path, f'{scene_id}.aggregation.json')
        self.segs_json_path = os.path.join(scene_path, f'{scene_id}_vh_clean_2.0.010000.segs.json')
        self.labels_ply_path = os.path.join(scene_path, f'{scene_id}_vh_clean_2.labels.ply')
        
        self.frame_files = sorted(os.listdir(self.color_dir), key=lambda f: int(f.split('.')[0]))
        
        # --- Pre-load and parse annotation files to create mappings ---
        self._check_paths()
        self.label_map = self._parse_labels_ply()
        self.segment_map = self._parse_segs_json()
        self.instance_to_label_map = self._create_instance_label_mapping()
        
        print(f"--- Initialized REAL ScanNetSemanticDataset for scene: {scene_id} ---")
        print(f"Found {len(self.frame_files)} frames. Using max {self.max_objects} objects per frame.")

    def _check_paths(self):
        """Checks if all necessary files and directories exist."""
        paths_to_check = [self.color_dir, self.depth_dir, self.instance_dir, 
                          self.aggregation_json_path, self.segs_json_path, self.labels_ply_path]
        for p in paths_to_check:
            if not os.path.exists(p):
                raise FileNotFoundError(f"Required file or directory not found: {p}")

    def _parse_labels_ply(self):
        """Parses the .labels.ply file to map a raw category ID to a class name."""
        plydata = PlyData.read(self.labels_ply_path)
        # FIX: Accessing PLY properties by name ('id', 'label') is more robust than by index (0, 2).
        # This resolves the AttributeError where the code expected a string but got a float at index 2.
        vertex_data = plydata['vertex'].data
        return {item['id']: item['label'].decode('utf-8') for item in vertex_data}

    def _parse_segs_json(self):
        """Parses the segs.json file to map segment IDs to a label index."""
        with open(self.segs_json_path) as f:
            segs_data = json.load(f)
        # Maps segment index -> objectId (which is the label index)
        return {i: seg['objectId'] for i, seg in enumerate(segs_data['segIndices'])}
        
    def _create_instance_label_mapping(self):
        """
        Creates the final mapping from an instance ID (from mask files) to a 
        semantic class label ID. This is the most complex part.
        Instance PNG -> Instance ID -> Segments -> Semantic Label
        """
        with open(self.aggregation_json_path) as f:
            agg_data = json.load(f)
        
        instance_map = {}
        for group in agg_data['segGroups']:
            instance_id = group['objectId'] + 1 # Align with instance PNG files
            label_name = group.get('label', 'unannotated')

            # Here you would establish a mapping from `label_name` to an integer index
            # for your classifier. For now, let's just store the name.
            # In a real scenario, you'd have a predefined class list.
            # e.g., class_to_id = {'wall': 0, 'chair': 1, ...}
            instance_map[instance_id] = label_name

        # Create a final mapping from instance ID to an integer class ID
        # Let's create a dynamic mapping for this example
        unique_labels = sorted(list(set(instance_map.values())))
        self.class_names = unique_labels
        self.class_to_idx = {name: i for i, name in enumerate(unique_labels)}
        print(f"Found {len(self.class_names)} unique classes in the scene.")

        final_map = {inst_id: self.class_to_idx[label] for inst_id, label in instance_map.items()}
        return final_map

    def __len__(self):
        return len(self.frame_files) - self.num_frames + 1

    def __getitem__(self, idx):
        frame_sequence_files = self.frame_files[idx : idx + self.num_frames]
        
        images_seq, depths_seq, masks_seq, labels_seq = [], [], [], []

        for frame_file in frame_sequence_files:
            # --- 1. Load Image and Depth ---
            img_path = os.path.join(self.color_dir, frame_file)
            image = Image.open(img_path).convert('RGB')
            
            depth_filename = frame_file.replace('.jpg', '.png')
            depth_path = os.path.join(self.depth_dir, depth_filename)
            depth = Image.open(depth_path)
            
            instance_filename = frame_file.replace('.jpg', '.png')
            instance_path = os.path.join(self.instance_dir, instance_filename)
            instance_img = Image.open(instance_path)

            # --- 2. Resize all inputs consistently ---
            image_tensor = TF.to_tensor(image)
            image_tensor = TF.resize(image_tensor, [self.img_height, self.img_width], antialias=True)
            
            depth_np = np.array(depth, dtype=np.float32) / 1000.0
            depth_tensor = torch.from_numpy(depth_np).unsqueeze(0)
            depth_tensor = TF.resize(depth_tensor, [self.img_height, self.img_width], antialias=True)
            
            instance_np = np.array(instance_img, dtype=np.int32)
            instance_tensor = torch.from_numpy(instance_np).unsqueeze(0)
            instance_tensor = TF.resize(instance_tensor, [self.img_height, self.img_width], interpolation=T.InterpolationMode.NEAREST)

            # --- 3. Create boolean masks and collect labels for each object ---
            instance_ids = torch.unique(instance_tensor)
            instance_ids = instance_ids[instance_ids != 0] # Remove background
            
            frame_masks, frame_labels = [], []
            for inst_id in instance_ids:
                label = self.instance_to_label_map.get(inst_id.item(), -1) # Use -1 for unmapped
                if label != -1:
                    mask = (instance_tensor == inst_id)
                    frame_masks.append(mask)
                    frame_labels.append(label)

            # --- 4. Pad masks and labels to a fixed size for batching ---
            num_found_objects = len(frame_masks)
            if num_found_objects > 0:
                padded_masks = torch.cat(frame_masks, dim=0)
                padded_labels = torch.tensor(frame_labels, dtype=torch.long)

                if num_found_objects < self.max_objects:
                    # Pad masks with empty masks
                    pad_size = self.max_objects - num_found_objects
                    empty_mask = torch.zeros(pad_size, self.img_height, self.img_width, dtype=torch.bool)
                    padded_masks = torch.cat([padded_masks, empty_mask], dim=0)
                    # Pad labels with an ignore_index
                    pad_labels_tensor = torch.full((pad_size,), -100, dtype=torch.long) # CrossEntropyLoss ignores -100
                    padded_labels = torch.cat([padded_labels, pad_labels_tensor], dim=0)
                elif num_found_objects > self.max_objects:
                    # Truncate to max_objects
                    padded_masks = padded_masks[:self.max_objects]
                    padded_labels = padded_labels[:self.max_objects]
            else:
                # No objects found, create all dummy tensors
                padded_masks = torch.zeros(self.max_objects, self.img_height, self.img_width, dtype=torch.bool)
                padded_labels = torch.full((self.max_objects,), -100, dtype=torch.long)


            images_seq.append(image_tensor)
            depths_seq.append(depth_tensor)
            masks_seq.append(padded_masks)
            labels_seq.append(padded_labels)

        batch = {
            "images": torch.stack(images_seq),           # Shape: (S, 3, H, W)
            "depths": torch.stack(depths_seq),           # Shape: (S, 1, H, W)
            "masks": torch.stack(masks_seq),             # Shape: (S, O, H, W)
            "gt_labels": torch.stack(labels_seq)         # Shape: (S, O)
        }
        return batch

    def get_num_classes(self):
        return len(self.class_names)


# =========================================================================================
# --- Step 2: Define the Final Monocular2Map Architecture ---
# =========================================================================================

class ObjectEncoder(nn.Module):
    """A simple CNN to encode masked object image regions into single tokens."""
    def __init__(self, output_dim=1024):
        super().__init__()
        # Use a pretrained ResNet18 as the encoder
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1]) # Remove the final FC layer
        self.projection = nn.Linear(resnet.fc.in_features, output_dim)

    def forward(self, x):
        features = self.encoder(x).flatten(1)
        return self.projection(features)


class Monocular2Map(nn.Module):
    """
    The final Monocular2Map architecture. It integrates a VGGT backbone for geometry,
    an ObjectEncoder for semantics, and a classifier head.
    """
    def __init__(self, num_classes, embed_dim=1024):
        super().__init__()
        print("--- Initializing Final Monocular2Map Architecture ---")
        self.backbone = VGGT()
        self.object_encoder = ObjectEncoder(output_dim=embed_dim)
        
        # Add a classifier head to predict class logits from object tokens
        self.semantic_classifier = nn.Linear(embed_dim, num_classes)
        
        # Transformation for resizing object crops before feeding to the encoder
        self.object_transform = T.Compose([
            T.Resize((224, 224), antialias=True),
            # Normalize with ImageNet stats, as expected by the pre-trained ResNet
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def forward(self, images, masks=None):
        """
        Implements the final hybrid forward pass.
        - `images`: Tensor of shape (S, 3, H, W)
        - `masks`: Tensor of shape (S, NumObjects, H, W)
        """
        # --- Part 1: Standard Geometric Prediction ---
        # This provides an auxiliary loss for geometry, which stabilizes training.
        geom_predictions = self.backbone(images)

        # If not training or no masks are provided, we only return geometric predictions.
        if not self.training or masks is None:
            return geom_predictions, None

        # --- Part 2: Hybrid Tokenization for Semantic Head ---
        object_tokens_list = []
        
        S, O, H, W = masks.shape
        
        # Iterate over each image in the sequence
        for i in range(S):
            img = images[i]
            img_masks = masks[i] # Shape: (O, H, W)
            
            object_crops = []
            # Iterate over potential objects in the frame
            for j in range(O):
                mask = img_masks[j]
                if not mask.any(): continue # Skip empty/padded masks

                # Get bounding box from the mask to crop the object
                rows = torch.any(mask, dim=1)
                cols = torch.any(mask, dim=0)
                rmin, rmax = torch.where(rows)[0][[0, -1]]
                cmin, cmax = torch.where(cols)[0][[0, -1]]

                # Ensure the crop is not empty
                if rmax > rmin and cmax > cmin:
                    crop = img.unsqueeze(0)[:, :, rmin:rmax+1, cmin:cmax+1]
                    resized_crop = self.object_transform(crop)
                    object_crops.append(resized_crop.squeeze(0))
            
            # If we found any valid objects in this frame, encode them
            if len(object_crops) > 0:
                object_crops_batch = torch.stack(object_crops, dim=0)
                tokens = self.object_encoder(object_crops_batch)
                object_tokens_list.append(tokens)

        # If no objects were found across the whole sequence, return
        if not object_tokens_list:
            return geom_predictions, None

        # Concatenate all valid object tokens into a single tensor
        all_object_tokens = torch.cat(object_tokens_list, dim=0)

        # Pass the object tokens through the classifier to get logits
        semantic_logits = self.semantic_classifier(all_object_tokens)
        
        return geom_predictions, semantic_logits


# =========================================================================================
# --- Step 3: Main Training and Evaluation Script ---
# =========================================================================================

def main():
    print("--- Phase C (Final): Fine-tuning Monocular2Map with Real Data and Semantic Logic ---")
    
    # --- Configuration ---
    # !! IMPORTANT: Update these paths to your local environment !!
    PRETRAINED_VGGT_PATH = "/home/ubuntu/Downloads/model.pt"
    SCANNET_SCENE_PATH = "/home/ubuntu/Downloads/scans/scene0684_01/" 
    SCANNET_SCENE_ID = "scene0684_01" # Used for constructing filenames inside the dataset class
    OUTPUT_DIR = "./final_monocular2map_output"
    
    # Training Hyperparameters
    LEARNING_RATE = 2e-5
    NUM_EPOCHS = 10 # Increased for meaningful training on real data
    BATCH_SIZE = 1  # Kept at 1 for simplicity with variable numbers of objects
    GEOM_LOSS_WEIGHT = 1.0
    SEMANTIC_LOSS_WEIGHT = 0.5 # Weight for the new semantic loss

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- 1. Setup Dataset and Dataloader ---
    try:
        dataset = RealScanNetSemanticDataset(
            scene_path=SCANNET_SCENE_PATH, 
            scene_id=SCANNET_SCENE_ID
        )
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        num_classes = dataset.get_num_classes()
        if num_classes == 0:
            print("\nERROR: No classes were found in the scene. Check your annotation files and parsing logic.")
            return
    except (FileNotFoundError, IndexError, json.JSONDecodeError) as e:
        print(f"\nERROR: Could not initialize dataset. Please check your paths and file formats.")
        print(f"Details: {e}")
        return

    # --- 2. Instantiate Your Custom Monocular2Map Model ---
    model = Monocular2Map(num_classes=num_classes).to(device)

    # --- 3. Load Pre-trained Weights and Apply LoRA ---
    if os.path.exists(PRETRAINED_VGGT_PATH):
        state_dict = torch.load(PRETRAINED_VGGT_PATH, map_location="cpu")
        if 'model' in state_dict: state_dict = state_dict['model']
        model.backbone.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pre-trained VGGT weights into the backbone.")
    else:
        print(f"WARNING: VGGT checkpoint not found at {PRETRAINED_VGGT_PATH}. Backbone is randomly initialized.")

    print("Applying LoRA to the VGGT backbone...")
    lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["qkv"], lora_dropout=0.05, bias="none")
    model.backbone = get_peft_model(model.backbone, lora_config)
    print("LoRA applied successfully. Trainable parameters overview:")
    model.backbone.print_trainable_parameters()

    # --- 4. Setup Optimizer and Loss Functions ---
    # The optimizer will train the LoRA adapters and the entire ObjectEncoder + Classifier
    trainable_params = [
        {'params': model.backbone.parameters()},
        {'params': model.object_encoder.parameters(), 'lr': LEARNING_RATE * 10}, # Higher LR for new parts
        {'params': model.semantic_classifier.parameters(), 'lr': LEARNING_RATE * 10}
    ]
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    
    # We use L1 for geometry and CrossEntropy for semantics (it ignores -100)
    geom_loss_fn = nn.L1Loss()
    semantic_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    
    print(f"Optimizer configured.")

    # --- 5. The Final Training Loop ---
    print("\n--- Starting Final Fine-Tuning Loop for Monocular2Map ---")
    model.train()

    for epoch in range(NUM_EPOCHS):
        total_geom_loss_epoch = 0
        total_sem_loss_epoch = 0
        for i, batch in enumerate(data_loader):
            optimizer.zero_grad()
            
            images = batch["images"].squeeze(0).to(device)
            masks = batch["masks"].squeeze(0).to(device)
            gt_depths = batch["depths"].squeeze(0).to(device)
            gt_labels_packed = batch["gt_labels"].squeeze(0) # Shape (S, O)
            
            # We need to filter out the padded labels (-100) and get a flat list of valid labels
            valid_gt_labels = gt_labels_packed.flatten()[gt_labels_packed.flatten() != -100].to(device)
            
            with torch.cuda.amp.autocast():
                geom_predictions, semantic_logits = model(images, masks)
                
                # --- Calculate Geometric Loss ---
                predicted_depth = geom_predictions["depth"].squeeze()
                gt_depths_squeezed = gt_depths.squeeze()
                geom_loss = geom_loss_fn(predicted_depth, gt_depths_squeezed)
                
                # --- Calculate Semantic Loss ---
                semantic_loss = torch.tensor(0.0).to(device)
                if semantic_logits is not None and len(valid_gt_labels) > 0:
                    # Ensure the number of logits matches the number of valid labels
                    if semantic_logits.shape[0] == valid_gt_labels.shape[0]:
                         semantic_loss = semantic_loss_fn(semantic_logits, valid_gt_labels)
                    else:
                        print(f"Warning: Mismatch between logits ({semantic_logits.shape[0]}) and labels ({valid_gt_labels.shape[0]}). Skipping semantic loss for this step.")

                # --- Combine Losses ---
                total_loss_step = (GEOM_LOSS_WEIGHT * geom_loss) + (SEMANTIC_LOSS_WEIGHT * semantic_loss)
            
            total_loss_step.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping for stability
            optimizer.step()

            total_geom_loss_epoch += geom_loss.item()
            if isinstance(semantic_loss, torch.Tensor) and semantic_loss.item() > 0:
                total_sem_loss_epoch += semantic_loss.item()

            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{i+1}/{len(data_loader)}], "
                  f"Geom Loss: {geom_loss.item():.4f}, Sem Loss: {semantic_loss.item():.4f}, "
                  f"Total Loss: {total_loss_step.item():.4f}")
            
            gc.collect()
            torch.cuda.empty_cache()

        avg_geom_loss = total_geom_loss_epoch / len(data_loader)
        avg_sem_loss = total_sem_loss_epoch / len(data_loader)
        print(f"--- End of Epoch {epoch+1} ---")
        print(f"Average Geom Loss: {avg_geom_loss:.4f}, Average Sem Loss: {avg_sem_loss:.4f}\n")

    print("--- Monocular2Map Final Fine-Tuning Finished ---")
    
    # --- 6. Save Final Trained Models ---
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