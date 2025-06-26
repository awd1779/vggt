# visualize.py
#
# A dedicated script to load a trained Monocular2Map model and visualize BOTH
# its 2D semantic predictions and its 3D colored point cloud output.

import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Add correct directories to path for local imports ---
script_dir = os.path.dirname(os.path.abspath(__file__))
training_dir = os.path.join(script_dir, "training")
if training_dir not in sys.path:
    sys.path.append(training_dir)
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)


# --- Local Imports ---
from dataset import RealScanNetSemanticDataset
from model import Monocular2Map

# --- Prerequisite Imports ---
try:
    from peft import PeftModel
    from vggt.models.vggt import VGGT
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    from vggt.utils.geometry import unproject_depth_map_to_point_map
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'peft' is installed and 'vggt' is in your path.")
    sys.exit(1)

# =============================================================================
# Helper Functions
# =============================================================================

def load_monocular2map_model(num_classes, model_output_dir, device):
    """Loads the full Monocular2Map model with all its trained components."""
    print(f"\n--- Loading Model (Monocular2Map) ---")
    lora_path = os.path.join(model_output_dir, "lora_adapters_monocular2map_final")
    encoder_path = os.path.join(model_output_dir, "object_encoder_final.pth")
    classifier_path = os.path.join(model_output_dir, "semantic_classifier_final.pth")

    model = Monocular2Map(num_classes=num_classes).to(device)
    model.backbone = PeftModel.from_pretrained(model.backbone, lora_path, is_trainable=False)
    model.object_encoder.load_state_dict(torch.load(encoder_path, map_location=device))
    model.semantic_classifier.load_state_dict(torch.load(classifier_path, map_location=device))
    
    model.eval()
    return model

def visualize_2d_predictions(image, masks, pred_labels, class_names, output_path):
    """Saves a 2D image with bounding boxes and predicted labels drawn on it."""
    img_np = image.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img_np)
    ax.axis('off')

    if masks is not None and pred_labels:
        for i, mask in enumerate(masks):
            if not mask.any() or i >= len(pred_labels): continue
            rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
            if not torch.any(rows) or not torch.any(cols): continue
            
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]
            class_name = class_names[pred_labels[i]]
            
            rect = patches.Rectangle((cmin, rmin), cmax - cmin, rmax - rmin, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(cmin, rmin - 5, class_name, bbox=dict(facecolor='lime', alpha=0.7), fontsize=10, color='black')

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved 2D visualization to {output_path}")

def save_colored_point_cloud(points, colors, output_path):
    """Saves a colored point cloud to a .obj file."""
    with open(output_path, 'w') as f:
        for i in range(points.shape[0]):
            p = points[i]
            c = colors[i]
            f.write(f"v {p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]:.4f} {c[1]:.4f} {c[2]:.4f}\n")
    print(f"Saved 3D point cloud to {output_path}")

# =============================================================================
# Main Visualization Script
# =============================================================================

def main():
    print("--- Starting Visualization Script (2D and 3D) ---")
    
    # --- Configuration ---
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    VIS_SCENE_PATH = os.path.join(training_dir, "/home/ubuntu/Downloads/scans", "scene0684_01", "processed")
    VIS_SCENE_ID = "scene0684_01"
    
    OUR_MODEL_OUTPUT_DIR = os.path.join(training_dir, "final_monocular2map_output")
    
    OUTPUT_DIR = os.path.join(script_dir, "visualization_output")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    FRAME_TO_VISUALIZE = 10 
    NUM_FRAMES_TO_PROCESS = 10

    # --- 1. Load Dataset ---
    try:
        vis_dataset = RealScanNetSemanticDataset(
            scene_path=VIS_SCENE_PATH, 
            scene_id=VIS_SCENE_ID,
            num_frames=NUM_FRAMES_TO_PROCESS
        )
        batch = vis_dataset[FRAME_TO_VISUALIZE] 
        num_classes = vis_dataset.get_num_classes()
        class_names = vis_dataset.class_names
    except Exception as e:
        print(f"\nERROR: Failed to load dataset: {e}")
        return

    # --- 2. Load Model ---
    model = load_monocular2map_model(num_classes, OUR_MODEL_OUTPUT_DIR, device)

    # --- 3. Run Inference ---
    print("\n--- Running Inference for Visualization ---")
    
    # Add a batch dimension (B=1) before passing to the model.
    images_batch = batch["images"].unsqueeze(0).to(device)
    masks_batch = batch["masks"].unsqueeze(0).to(device)
    
    with torch.no_grad():
        geom_predictions, semantic_logits = model(images_batch, masks_batch)
    
    predicted_labels = []
    if semantic_logits is not None:
        predicted_labels = torch.argmax(semantic_logits, dim=1).cpu().tolist()

    # --- 4. Generate and Save 2D Visualization ---
    image_to_show = batch["images"][0]
    masks_to_show = batch["masks"][0]
    output_2d_path = os.path.join(OUTPUT_DIR, f"{VIS_SCENE_ID}_frame_{FRAME_TO_VISUALIZE}_2D_semantic.png")
    
    visualize_2d_predictions(
        image=image_to_show,
        masks=masks_to_show,
        pred_labels=predicted_labels,
        class_names=class_names,
        output_path=output_2d_path
    )

    # --- 5. Generate and Save 3D Point Cloud ---
    if "depth" in geom_predictions and "pose_enc" in geom_predictions:
        # Pass the raw, batched predictions directly to the utility function.
        depth_map_pred = geom_predictions["depth"]
        pose_enc_pred = geom_predictions["pose_enc"]
        
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc_pred, depth_map_pred.shape[2:4])

        # FIX: The library function returns a NumPy array, so we don't call .cpu().numpy()
        points = unproject_depth_map_to_point_map(
            depth_map_pred.squeeze(0), # Remove batch dimension for the library function
            extrinsic.squeeze(0), 
            intrinsic.squeeze(0)
        ).reshape(-1, 3)

        H, W = image_to_show.shape[1], image_to_show.shape[2]
        pixel_labels = torch.full((H, W), -1, dtype=torch.long, device=device)
        palette = torch.tensor(plt.colormaps.get_cmap("tab20").colors, dtype=torch.float32, device=device)

        for i, mask in enumerate(masks_to_show):
            if mask.any() and i < len(predicted_labels):
                pixel_labels[mask.squeeze(0)] = predicted_labels[i]

        colors = torch.zeros(H * W, 3, device=device)
        valid_pixels = pixel_labels.flatten() != -1
        if torch.any(valid_pixels):
            valid_labels = pixel_labels.flatten()[valid_pixels]
            colors[valid_pixels] = palette[valid_labels % len(palette)]
        
        num_frames = depth_map_pred.shape[1]
        tiled_colors = colors.repeat(num_frames, 1).cpu().numpy()

        output_3d_path = os.path.join(OUTPUT_DIR, f"{VIS_SCENE_ID}_frame_{FRAME_TO_VISUALIZE}_3D_pointcloud.obj")
        save_colored_point_cloud(points, tiled_colors, output_3d_path)

    print("\n--- Visualization Complete ---")


if __name__ == "__main__":
    main()
