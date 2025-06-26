# app.py
#
# An interactive Gradio demo for the Monocular2Map model.
# This version replicates the full functionality of the original VGGT demo,
# including advanced 3D visualization controls and camera pose rendering.

import torch
import numpy as np
import os
import sys
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import gradio as gr
import gc
import tempfile
import shutil
from datetime import datetime
import cv2

# --- Prerequisite: Install trimesh for advanced 3D visualization ---
# pip install trimesh
try:
    import trimesh
except ImportError:
    print("Error: 'trimesh' library not found. Please install it using 'pip install trimesh'")
    sys.exit(1)


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
except ImportError as e:
    print(f"Import Error: {e}")
    print("Please ensure 'peft' is installed and the 'vggt' library is in your Python path.")
    sys.exit(1)


# --- Global Caches for Performance ---
MODEL_CACHE = {}
DATASET_CACHE = {}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Configuration ---
BASE_SCAN_PATH = os.path.join(training_dir, "/home/ubuntu/Downloads/scans")
MODEL_OUTPUT_DIR = os.path.join(training_dir, "final_monocular2map_output")

try:
    AVAILABLE_SCENES = sorted([d for d in os.listdir(BASE_SCAN_PATH) if os.path.isdir(os.path.join(BASE_SCAN_PATH, d))])
    if not AVAILABLE_SCENES:
        raise FileNotFoundError
except FileNotFoundError:
    print(f"Warning: Could not find any scene folders in '{BASE_SCAN_PATH}'. Please check the path.")
    AVAILABLE_SCENES = ["scene_not_found"]


# =============================================================================
# Core Logic: Model Loading, Prediction, and Visualization
# =============================================================================

def load_model(scene_id=None):
    """Loads a model into the cache. If scene_id is None, loads a generic model."""
    cache_key = scene_id if scene_id else "generic"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    print(f"\n--- Loading model for: {cache_key} ---")
    
    num_classes = 1
    if scene_id:
        if scene_id not in DATASET_CACHE:
            load_dataset_for_scene(scene_id)
        vis_dataset = DATASET_CACHE[scene_id]
        num_classes = vis_dataset.get_num_classes()

    lora_path = os.path.join(MODEL_OUTPUT_DIR, "lora_adapters_monocular2map_final")
    model = Monocular2Map(num_classes=num_classes).to(DEVICE)
    model.backbone = PeftModel.from_pretrained(model.backbone, lora_path, is_trainable=False)
    
    if scene_id:
        encoder_path = os.path.join(MODEL_OUTPUT_DIR, "object_encoder_final.pth")
        classifier_path = os.path.join(MODEL_OUTPUT_DIR, "semantic_classifier_final.pth")
        model.object_encoder.load_state_dict(torch.load(encoder_path, map_location=DEVICE))
        model.semantic_classifier.load_state_dict(torch.load(classifier_path, map_location=DEVICE))
    
    model.eval()
    MODEL_CACHE[cache_key] = model
    print(f"--- Model for {cache_key} loaded and cached. ---")
    return model

def load_dataset_for_scene(scene_id):
    """Loads a scene-specific dataset into the cache."""
    if scene_id in DATASET_CACHE:
        return DATASET_CACHE[scene_id]
    
    print(f"\n--- Loading dataset for scene: {scene_id} ---")
    scene_path = os.path.join(BASE_SCAN_PATH, scene_id, "processed")
    dataset = RealScanNetSemanticDataset(scene_path=scene_path, scene_id=scene_id)
    DATASET_CACHE[scene_id] = dataset
    return dataset

# --- Self-Contained Geometry Functions (from vggt library to bypass errors) ---
def depth_to_cam_coords(depth_map, intrinsic):
    H, W = depth_map.shape
    vs, us = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    vs, us = vs.to(depth_map.device), us.to(depth_map.device)
    zs = depth_map
    xs = (us - intrinsic[0, 2]) * zs / intrinsic[0, 0]
    ys = (vs - intrinsic[1, 2]) * zs / intrinsic[1, 1]
    return torch.stack([xs, ys, zs], dim=-1)

def cam_to_world_coords(cam_coords, extrinsic):
    world_coords = torch.linalg.inv(extrinsic)[:3, :3] @ cam_coords.T + torch.linalg.inv(extrinsic)[:3, 3, None]
    return world_coords.T

def create_and_export_glb(predictions, show_cam, prediction_mode, conf_thres, colors=None):
    """Creates a 3D scene with point cloud and cameras, exports to GLB."""
    if not predictions:
        return None

    for k, v in predictions.items():
        if isinstance(v, np.ndarray):
            predictions[k] = torch.from_numpy(v).to(DEVICE)

    if prediction_mode == "Depthmap and Camera Branch":
        depth_map = predictions["depth"]
        pose_enc = predictions["pose_enc"]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, depth_map.shape[1:3])
        
        # FIX: Custom unprojection loop to avoid library errors
        all_points = []
        for i in range(depth_map.shape[0]):
            depth_frame = depth_map[i].squeeze()
            cam_coords = depth_to_cam_coords(depth_frame, intrinsic[i])
            world_coords = cam_to_world_coords(cam_coords.reshape(-1, 3), extrinsic[i])
            all_points.append(world_coords)
        points = torch.cat(all_points, dim=0)
        conf = predictions.get("depth_conf", torch.ones_like(depth_map)).flatten()
    else: 
        points = predictions["point_map"].reshape(-1, 3)
        conf = predictions.get("point_map_conf", torch.ones_like(points[:,0])).flatten()
        extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], (0,0))

    conf_mask = conf > (conf_thres / 100.0)
    points = points[conf_mask].cpu().numpy()
    if colors is not None:
        colors = colors.reshape(-1, 3)[conf_mask.cpu().numpy()]
    
    pcd = trimesh.points.PointCloud(points, colors=colors if colors is not None else None)
    scene = trimesh.Scene(pcd)
    
    if show_cam and extrinsic is not None:
        for T_world_cam in extrinsic.cpu().numpy():
            cam_marker = trimesh.creation.camera_marker(transform=T_world_cam)
            scene.add_geometry(cam_marker)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".glb") as f:
        scene.export(file_obj=f, file_type='glb')
        return f.name

def process_scannet_request(scene_id, frame_index):
    """Handles the logic for the ScanNet demo."""
    frame_index = int(frame_index)
    dataset = load_dataset_for_scene(scene_id)
    batch = dataset[frame_index]
    
    model = load_model(scene_id)
    images = batch["images"].unsqueeze(0).to(DEVICE)
    masks = batch["masks"].unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        predictions, semantic_logits = model(images.squeeze(0), masks.squeeze(0))
    
    if semantic_logits is not None:
        predictions["semantic_logits"] = semantic_logits
    
    if 'extrinsic' not in predictions:
        predictions["extrinsic"], predictions["intrinsic"] = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    return predictions, batch

def process_upload_request(target_dir):
    """Handles logic for user uploads."""
    model = load_model(scene_id=None)
    image_names = sorted([os.path.join(target_dir, f) for f in os.listdir(target_dir)])
    images = load_and_preprocess_images(image_names).to(DEVICE)
    
    with torch.no_grad():
        predictions, _ = model(images, masks=None)
    
    if 'extrinsic' not in predictions:
        predictions["extrinsic"], predictions["intrinsic"] = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])

    return predictions

def render_2d_visualization(image_tensor, mask_tensor, pred_labels, class_names):
    """Renders the 2D matplotlib visualization from tensors."""
    if image_tensor is None: return None
    img_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    fig, ax = plt.subplots(1, figsize=(12, 9), dpi=150)
    ax.imshow(img_np)
    ax.axis('off')

    if mask_tensor is not None and pred_labels:
        for i, mask in enumerate(mask_tensor):
            if not mask.any() or i >= len(pred_labels): continue
            rows, cols = torch.any(mask, dim=1), torch.any(mask, dim=0)
            if not torch.any(rows) or not torch.any(cols): continue
            
            rmin, rmax = torch.where(rows)[0][[0, -1]]
            cmin, cmax = torch.where(cols)[0][[0, -1]]
            class_name = class_names[pred_labels[i]]
            
            rect = patches.Rectangle((cmin.cpu(), rmax.cpu()), cmax.cpu() - cmin.cpu(), rmin.cpu() - rmax.cpu(), linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            ax.text(cmin.cpu(), rmin.cpu() - 5, class_name, bbox=dict(facecolor='lime', alpha=0.7), fontsize=10, color='black')

    fig.canvas.draw()
    rendered_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close(fig)
    gc.collect()
    return rendered_img

# =============================================================================
# Gradio UI Layout
# =============================================================================

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Monocular2Map: Interactive Demo")
    
    prediction_state = gr.State({})
    batch_state = gr.State({})
    upload_prediction_state = gr.State({})

    with gr.Tabs():
        with gr.TabItem("Demo from ScanNet Scenes"):
            with gr.Row():
                with gr.Column(scale=1):
                    scene_selector = gr.Dropdown(AVAILABLE_SCENES, label="ScanNet Scene ID", value=AVAILABLE_SCENES[0] if AVAILABLE_SCENES else None)
                    frame_slider = gr.Slider(label="Frame Index", minimum=0, maximum=1500, value=10, step=1)
                with gr.Column(scale=3):
                    output_image_2d = gr.Image(label="2D Semantic View", type="numpy")
            
            scannet_btn = gr.Button("1. Load Scene and Run Prediction", variant="primary")
            gr.Markdown("---")
            gr.Markdown("### 2. Adjust 3D Visualization")

            with gr.Row():
                with gr.Column():
                    prediction_mode_scan = gr.Radio(["Depthmap and Camera Branch", "Pointmap Branch"], label="Prediction Mode", value="Depthmap and Camera Branch")
                    show_cam_scan = gr.Checkbox(label="Show Cameras", value=True)
                with gr.Column():
                    conf_thres_scan = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")
            
            output_model_3d_scannet = gr.Model3D(label="3D Reconstruction", camera_position=(45, 45, 2.5))
        
        with gr.TabItem("Upload Your Own Images/Video"):
            with gr.Row():
                 with gr.Column(scale=1):
                    upload_video = gr.Video(label="Upload Video")
                    upload_images = gr.File(label="Upload Images", file_count="multiple")
                    upload_gallery = gr.Gallery(label="Uploaded Frames Preview", columns=4, height=300)
                 with gr.Column(scale=3):
                    output_model_3d_upload = gr.Model3D(label="Reconstructed 3D Point Cloud", camera_position=(45, 45, 2.5))
            
            upload_btn = gr.Button("1. Reconstruct from Upload", variant="primary")
            gr.Markdown("---")
            gr.Markdown("### 2. Adjust 3D Visualization")

            with gr.Row():
                with gr.Column():
                    prediction_mode_upload = gr.Radio(["Depthmap and Camera Branch", "Pointmap Branch"], label="Prediction Mode", value="Depthmap and Camera Branch")
                    show_cam_upload = gr.Checkbox(label="Show Cameras", value=True)
                with gr.Column():
                    conf_thres_upload = gr.Slider(minimum=0, maximum=100, value=50, step=0.1, label="Confidence Threshold (%)")

    # --- Event Handlers ---
    def scannet_tab_controller(scene_id, frame_index, progress=gr.Progress()):
        progress(0, desc="Loading Scene and Model...")
        predictions, batch_data_tensors = process_scannet_request(scene_id, frame_index)
        
        progress(0.5, desc="Generating 2D Visualization...")
        
        predicted_labels = []
        if "semantic_logits" in predictions:
            predicted_labels = torch.argmax(predictions["semantic_logits"], dim=1).cpu().tolist()
        
        dataset = DATASET_CACHE[scene_id]
        
        img_2d = render_2d_visualization(
            batch_data_tensors["images"][0], 
            batch_data_tensors["masks"][0], 
            predicted_labels, dataset.class_names
        )
        
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()
        for key in batch_data_tensors:
            if isinstance(batch_data_tensors[key], torch.Tensor):
                batch_data_tensors[key] = batch_data_tensors[key].cpu().numpy()

        progress(1.0, desc="Done.")
        return predictions, batch_data_tensors, img_2d

    def update_scannet_3d_view(predictions, batch_data, show_cam, pred_mode, conf_thres):
        if not predictions: return None
        
        masks_to_show = torch.from_numpy(batch_data["masks"][0])
        predicted_labels = []
        if "semantic_logits" in predictions:
            predicted_labels = np.argmax(predictions["semantic_logits"], axis=1).tolist()

        H, W = batch_data["images"][0].shape[1], batch_data["images"][0].shape[2]
        pixel_labels = torch.full((H, W), -1, dtype=torch.long)
        palette = torch.tensor(plt.colormaps.get_cmap("tab20").colors, dtype=torch.float32)

        for i, mask in enumerate(masks_to_show):
            if mask.any() and i < len(predicted_labels):
                pixel_labels[mask.squeeze(0)] = predicted_labels[i]
        
        colors = torch.zeros(H * W, 3)
        valid_pixels = pixel_labels.flatten() != -1
        if torch.any(valid_pixels):
            valid_labels = pixel_labels.flatten()[valid_pixels]
            colors[valid_pixels] = palette[valid_labels % len(palette)]

        num_frames = predictions["depth"].shape[0]
        tiled_colors = colors.repeat(num_frames, 1).numpy()
        
        return create_and_export_glb(predictions, show_cam, pred_mode, conf_thres, tiled_colors)

    scannet_btn.click(
        fn=scannet_tab_controller,
        inputs=[scene_selector, frame_slider],
        outputs=[prediction_state, batch_state, output_image_2d]
    ).then(
        fn=update_scannet_3d_view,
        inputs=[prediction_state, batch_state, show_cam_scan, prediction_mode_scan, conf_thres_scan],
        outputs=[output_model_3d_scannet]
    )
    
    vis_inputs_scan = [prediction_state, batch_state, show_cam_scan, prediction_mode_scan, conf_thres_scan]
    show_cam_scan.change(update_scannet_3d_view, inputs=vis_inputs_scan, outputs=[output_model_3d_scannet])
    prediction_mode_scan.change(update_scannet_3d_view, inputs=vis_inputs_scan, outputs=[output_model_3d_scannet])
    conf_thres_scan.change(update_scannet_3d_view, inputs=vis_inputs_scan, outputs=[output_model_3d_scannet])

    # --- Upload Tab Handlers ---
    upload_dir_state = gr.State()

    def handle_uploads_wrapper(video, images, progress=gr.Progress()):
        progress(0, desc="Copying files...")
        target_dir = tempfile.mkdtemp()
        image_paths = []
        files_to_process = images if images else []
        if video:
            files_to_process.append(video)

        for file_path in files_to_process:
            is_video = video and (file_path == video)
            if is_video:
                vs = cv2.VideoCapture(file_path.name)
                fps = vs.get(cv2.CAP_PROP_FPS, 30)
                frame_interval = int(fps) if fps > 0 else 1
                count, frame_num = 0, 0
                while True:
                    gotit, frame = vs.read()
                    if not gotit: break
                    count += 1
                    if count % frame_interval == 0:
                        image_path = os.path.join(target_dir, f"{frame_num:06}.png")
                        cv2.imwrite(image_path, frame)
                        image_paths.append(image_path)
                        frame_num += 1
            else:
                shutil.copy(file_path.name, target_dir)
                image_paths.append(os.path.join(target_dir, os.path.basename(file_path.name)))
        
        image_paths = sorted(image_paths)
        progress(1.0, "Files ready.")
        return target_dir, image_paths

    upload_video.upload(handle_uploads_wrapper, inputs=[upload_video, upload_images], outputs=[upload_dir_state, upload_gallery])
    upload_images.upload(handle_uploads_wrapper, inputs=[upload_video, upload_images], outputs=[upload_dir_state, upload_gallery])

    def upload_tab_controller(target_dir, progress=gr.Progress()):
        if not target_dir:
            raise gr.Error("Please upload files first.")
        progress(0, desc="Loading model and running inference...")
        predictions = process_upload_request(target_dir)
        progress(0.5, desc="Converting predictions...")
        for key in predictions:
            if isinstance(predictions[key], torch.Tensor):
                predictions[key] = predictions[key].cpu().numpy()
        progress(1.0, "Prediction ready.")
        return predictions

    def update_upload_3d_view(predictions, show_cam, pred_mode, conf_thres):
        if not predictions: return None
        return create_and_export_glb(predictions, show_cam, pred_mode, conf_thres, colors=None)

    upload_btn.click(
        fn=upload_tab_controller,
        inputs=[upload_dir_state],
        outputs=[upload_prediction_state]
    ).then(
        fn=update_upload_3d_view,
        inputs=[upload_prediction_state, show_cam_upload, prediction_mode_upload, conf_thres_upload],
        outputs=[output_model_3d_upload]
    )
    
    vis_inputs_upload = [upload_prediction_state, show_cam_upload, prediction_mode_upload, conf_thres_upload]
    show_cam_upload.change(update_upload_3d_view, inputs=vis_inputs_upload, outputs=[output_model_3d_upload])
    prediction_mode_upload.change(update_upload_3d_view, inputs=vis_inputs_upload, outputs=[output_model_3d_upload])
    conf_thres_upload.change(update_upload_3d_view, inputs=vis_inputs_upload, outputs=[output_model_3d_upload])


if __name__ == "__main__":
    demo.launch(share=True)
