import torch
from PIL import Image
from ultralytics import YOLOWorld
from transformers import AutoImageProcessor, AutoModel
import open_clip

models = {}

def load_models():
    """Loads the GOD TIER Models (Max Accuracy)"""
    print("⏳ Loading GOD TIER Models... (Downloads ~3.5GB)")

    # 1. THE EYE (YOLO)
    print("   - Loading YOLO-World...")
    yolo = YOLOWorld("yolov8s-world.pt")
    yolo.set_classes(["product", "watch", "shoe", "clothing", "electronic device"])
    models['yolo'] = yolo

    # 2. THE SCOUT (SigLIP Large) - 1152 Dimensions
    print("   - Loading SigLIP (Large)...")
    siglip, _, siglip_pre = open_clip.create_model_and_transforms(
        'ViT-SO400M-14-SigLIP', 
        pretrained='webli'
    )
    models['siglip'] = siglip
    models['siglip_pre'] = siglip_pre

    # 3. THE JUDGE (DINOv2 Large) - 1024 Dimensions
    # This is the specific fix for Hexagon vs Round.
    # The 'Large' model sees geometry much better than 'Base'.
    print("   - Loading DINOv2 (Large)...")
    models['dino_proc'] = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
    models['dino_model'] = AutoModel.from_pretrained('facebook/dinov2-large')
    
    print("✅ God Mode Active!")

def smart_crop(image: Image.Image) -> Image.Image:
    # ... (Keep your existing smart_crop code, but CHANGE PADDING TO 0) ...
    # See Step 3 below
    yolo = models['yolo']
    results = yolo(image, verbose=False)
    
    best_box = None
    max_conf = 0.0
    
    for r in results:
        for box in r.boxes:
            if box.conf > max_conf:
                max_conf = box.conf.item()
                best_box = box.xyxy[0].tolist()
    
    if best_box and max_conf > 0.15:
        # FIX: Set padding to 0 or very small (e.g. 5)
        # This removes background distraction so DINO sees the shape clearly.
        padding = 5 
        x1, y1, x2, y2 = best_box
        return image.crop((
            max(0, x1 - padding), 
            max(0, y1 - padding), 
            min(image.width, x2 + padding), 
            min(image.height, y2 + padding)
        ))
    return image

def get_siglip_vector(image: Image.Image) -> list:
    preprocess = models['siglip_pre']
    model = models['siglip']
    with torch.no_grad():
        emb = model.encode_image(preprocess(image).unsqueeze(0))
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().tolist()

def get_dino_vector(image: Image.Image) -> list:
    processor = models['dino_proc']
    model = models['dino_model']
    with torch.no_grad():
        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)
        emb = outputs.last_hidden_state[:, 0, :] 
        emb /= emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().tolist()