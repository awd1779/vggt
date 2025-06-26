# compare.py
#
# A dedicated script to debug and compare the outputs of the original VGGT model
# against our fine-tuned Monocular2Map model on the same data.
# This helps isolate whether issues are in the model or the visualization logic.

import torch
import numpy as np
import os
import sys

# --- Add correct directories to path for local imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, "training")
if training_dir not in sys.path:
    sys.path.append(training_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# --- Local Imports ---
from training.dataset import RealScanNetSemanticDataset
from training.model import Monocular2Map

# --- Prerequisite Imports ---
try:
    from peft import PeftModel
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'peft' is installed and the 'vggt' library is in your path.")
    sys.exit(1)

# =============================================================================
# Helper Functions
# =============================================================================

def load_monocular2map_model(num_classes, model_output_dir, device):
    """Loads our fine-tuned Monocular2Map model."""
    print(f"\n--- Loading Our Model (Monocular2Map) ---")
    lora_path = os.path.join(model_output_dir, "lora_adapters_monocular2map_final")
    encoder_path = os.path.join(model_output_dir, "object_encoder_final.pth")
    classifier_path = os.path.join(model_output_dir, "semantic_classifier_final.pth")

    model = Monocular2Map(num_classes=num_classes).to(device)
    model.backbone = PeftModel.from_pretrained(model.backbone, lora_path, is_trainable=False)
    model.object_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.semantic_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    model.eval()
    return model

def load_original_vggt_model(device):
    """Loads the original, pre-trained VGGT model from torch.hub."""
    print(f"\n--- Loading Original VGGT Model ---")
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.to(device)
    model.eval()
    return model

def generate_point_cloud(predictions, device):
    """Generates a 3D point cloud from geometric predictions."""
    if not predictions or "depth" not in predictions or "pose_enc" not in predictions:
        return None
        
    depth_map = predictions["depth"]
    pose_enc = predictions["pose_enc"]
    
    # The utility function handles the batched dimension correctly
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, depth_map.shape[2:4])
    
    all_points = []
    # FIX: Squeeze the batch dimension before looping through frames
    depth_map_seq = depth_map.squeeze(0)
    extrinsic_seq = extrinsic.squeeze(0)
    intrinsic_seq = intrinsic.squeeze(0)

    for i in range(depth_map_seq.shape[0]):
        depth_frame = depth_map_seq[i].squeeze(-1)
        H, W = depth_frame.shape
        vs, us = torch.meshgrid(torch.arange(H, device=device), torch.arange(W, device=device), indexing="ij")
        zs = depth_frame
        xs = (us - intrinsic_seq[i, 0, 2]) * zs / intrinsic_seq[i, 0, 0]
        ys = (vs - intrinsic_seq[i, 1, 2]) * zs / intrinsic_seq[i, 1, 1]
        cam_coords = torch.stack([xs, ys, zs], dim=-1)
        
        # FIX: The extrinsic matrix is already the camera-to-world transform
        # No inversion is needed.
        T_world_cam = extrinsic_seq[i]
        cam_coords_hom = torch.cat([cam_coords.reshape(-1, 3), torch.ones(H*W, 1, device=device)], dim=-1)
        world_coords = (T_world_cam @ cam_coords_hom.T).T[:, :3]
        all_points.append(world_coords)
    
    return torch.cat(all_points, dim=0).cpu().numpy()

def save_point_cloud(points, output_path):
    """Saves a point cloud to a .obj file."""
    if points is None: return
    print(f"Saving point cloud to {output_path}...")
    with open(output_path, 'w') as f:
        for p in points:
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}\n")
    print("Save complete.")

# =============================================================================
# Main Debugging Script
# =============================================================================

def main():
    print("--- Starting Debug Comparison Script ---")
    
    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    SCENE_PATH = os.path.join(training_dir, "/home/ubuntu/Downloads/scans", "scene0684_01", "processed")
    SCENE_ID = "scene0684_01"
    
    OUR_MODEL_OUTPUT_DIR = os.path.join(training_dir, "final_monocular2map_output")
    
    OUTPUT_DIR = os.path.join(script_dir, "debug_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    FRAME_INDEX = 10
    NUM_FRAMES = 10

    # --- 1. Load Dataset ---
    try:
        dataset = RealScanNetSemanticDataset(scene_path=SCENE_PATH, scene_id=SCENE_ID, num_frames=NUM_FRAMES)
        batch = dataset[FRAME_INDEX]
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}"); return

    # --- 2. Load Both Models ---
    original_model = load_original_vggt_model(device)
    our_model = load_monocular2map_model(dataset.get_num_classes(), OUR_MODEL_OUTPUT_DIR, device)
    
    # --- 3. Prepare Input Data ---
    images = batch["images"].to(device)

    # --- 4. Run Inference on Both Models ---
    print("\n--- Running Inference ---")
    with torch.no_grad():
        original_preds = original_model(images)
        our_preds, _ = our_model(images, masks=None)

    # --- 5. Print Diagnostic Comparison ---
    print("\n\n--- DIAGNOSTIC REPORT ---")
    print("-" * 25)
    
    keys_to_compare = ["depth", "pose_enc"]
    for key in keys_to_compare:
        if key in original_preds and key in our_preds:
            orig_t = original_preds[key]
            our_t = our_preds[key]
            print(f"\nComparing output key: '{key}'")
            print(f"  Original VGGT Shape: {orig_t.shape}")
            print(f"  Our Model Shape:     {our_t.shape}")
            
            orig_t_cpu = orig_t.cpu().float()
            our_t_cpu = our_t.cpu().float()
            
            print(f"  Original VGGT Stats (Mean/Std/Min/Max): {orig_t_cpu.mean():.4f} / {orig_t_cpu.std():.4f} / {orig_t_cpu.min():.4f} / {orig_t_cpu.max():.4f}")
            print(f"  Our Model Stats    (Mean/Std/Min/Max): {our_t_cpu.mean():.4f} / {our_t_cpu.std():.4f} / {our_t_cpu.min():.4f} / {our_t_cpu.max():.4f}")
            
            diff = torch.abs(orig_t_cpu - our_t_cpu).mean()
            print(f"  => Mean Absolute Difference: {diff:.6f}")
        else:
            print(f"\nKey '{key}' not found in one of the model outputs.")

    print("\n" + "-" * 25)

    # --- 6. Generate and Save Point Clouds for Visual Comparison ---
    original_pc = generate_point_cloud(original_preds, device)
    our_model_pc = generate_point_cloud(our_preds, device)

    save_point_cloud(original_pc, os.path.join(OUTPUT_DIR, "original_vg_gt_pc.obj"))
    save_point_cloud(our_model_pc, os.path.join(OUTPUT_DIR, "our_monocular2map_pc.obj"))

    print("\n--- Debug Comparison Complete ---")


if __name__ == "__main__":
    main()
