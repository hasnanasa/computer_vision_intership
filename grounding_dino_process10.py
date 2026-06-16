import os
import sys
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
import torchvision.transforms as transforms
sys.path.insert(0, r"C:\Users\hp\video_processor\GroundingDINO")

from groundingdino.util.inference import load_model, load_image, predict

# ========== CONFIGURATION ==========
IMAGE_DIR    = r"C:\Users\hp\video_processor\output\frames_uniques"
OUTPUT_DIR   = r"C:\Users\hp\video_processor\smart_output"
CONFIG_PATH  = r"C:\Users\hp\video_processor\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py"
WEIGHTS_PATH = r"C:\Users\hp\video_processor\GroundingDINO\groundingdino_swint_ogc.pth"

DINO_BOX        = 0.25
DINO_TEXT       = 0.20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Utilisation du device : {DEVICE.upper()}")
if DEVICE == "cuda":
    print(f"✅ GPU trouvé : {torch.cuda.get_device_name(0)}")
else:
    print("⚠️  GPU non trouvé, exécution sur CPU (lent)")
SIMILARITY_MIN  = 0.90             # ← plus strict
MIN_MEMORY_SIZE = 3                  # ← rejeter si mémoire trop petite
USE_DINOV2      = True               # ← meilleure qualité, mais plus lent

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== SHARPNESS FUNCTION (from your other script) ==========
def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s1 = cv2.Laplacian(gray, cv2.CV_64F).var()
    s2 = cv2.Laplacian(cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2)), cv2.CV_64F).var()
    s3 = cv2.Laplacian(cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4)), cv2.CV_64F).var()
    return (s1 + s2 + s3) / 3.0

# ========== CHARGEMENT MODÈLES ==========
print("⚙️  Chargement Grounding DINO...")
dino = load_model(CONFIG_PATH, WEIGHTS_PATH, device=DEVICE)
print("   ✅ Grounding DINO")

if USE_DINOV2:
    print("⚙️  Chargement DINOv2 (extracteur de features)...")
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    dinov2.eval().to(DEVICE)
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    print("   ✅ DINOv2")
else:
    print("⚙️  Chargement ResNet50 (fallback)...")
    import torchvision.models as models
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    feature_ext = torch.nn.Sequential(*list(resnet.children())[:-1])
    feature_ext.eval()
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    print("   ✅ ResNet50")

# ========== EXTRACTION FEATURE (DINOv2 ou ResNet) ==========
def get_feature(img_bgr):
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = preprocess(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if USE_DINOV2:
            feat = dinov2(tensor)  # shape [1, 768]
        else:
            feat = feature_ext(tensor)  # shape [1, 2048, 1, 1]
            feat = feat.squeeze(-1).squeeze(-1)
    return F.normalize(feat.squeeze(), dim=0)

def cosine_similarity(f1, f2):
    return float(torch.dot(f1, f2).item())

# ========== AUGMENTATIONS (pour construire la mémoire) ==========
def build_augmented_views(crop_bgr):
    views = []
    h, w = crop_bgr.shape[:2]
    if h < 10 or w < 10:
        return views
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    views.append(crop_bgr.copy())
    views.append(cv2.flip(crop_bgr, 1))
    views.append(cv2.flip(crop_bgr, 0))
    views.append(cv2.convertScaleAbs(crop_bgr, alpha=1.6, beta=40))
    views.append(cv2.convertScaleAbs(crop_bgr, alpha=0.5, beta=-20))
    views.append(cv2.convertScaleAbs(crop_bgr, alpha=2.0, beta=-60))

    for g in [0.5, 0.8, 1.5, 2.0]:
        table = np.array([(i/255.0)**g*255 for i in range(256)]).astype("uint8")
        views.append(cv2.LUT(crop_bgr, table))

    edges = cv2.Canny(gray, 30, 100)
    views.append(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR))

    sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.normalize(np.sqrt(sx**2 + sy**2), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    views.append(cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR))

    views.append(gray_3ch)

    lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    views.append(cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR))

    cy, cx = h//2, w//2
    nh, nw = int(h*0.66), int(w*0.66)
    y1 = max(0, cy-nh//2); y2 = min(h, cy+nh//2)
    x1 = max(0, cx-nw//2); x2 = min(w, cx+nw//2)
    if y2 > y1 and x2 > x1:
        views.append(cv2.resize(crop_bgr[y1:y2, x1:x2], (w, h)))

    views.append(cv2.GaussianBlur(crop_bgr, (15, 15), 0))
    views.append(cv2.convertScaleAbs(crop_bgr, alpha=2.5, beta=-80))
    return views

# ========== AUGMENTATIONS POUR LA REQUÊTE (test‑time) ==========
def build_query_views(crop_bgr):
    """Version plus légère pour la requête (5 vues)"""
    views = [crop_bgr.copy()]
    views.append(cv2.flip(crop_bgr, 1))
    views.append(cv2.convertScaleAbs(crop_bgr, alpha=1.4, beta=20))
    # petite rotation (simulée par un shift de colonnes)
    h, w = crop_bgr.shape[:2]
    if h > 20 and w > 20:
        M = cv2.getRotationMatrix2D((w/2, h/2), 5, 1)
        rotated = cv2.warpAffine(crop_bgr, M, (w, h))
        views.append(rotated)
    # flou léger
    views.append(cv2.GaussianBlur(crop_bgr, (5, 5), 0))
    return views

# ========== PROMPT ==========
print("\n🔍 Que veux-tu détecter ?")
print("   Exemple : chair . computer . person . table")
user_input = input("   Ton prompt → ").strip().lower()
target_list = [c.strip() for c in user_input.replace(",", ".").split(".") if c.strip()]
dino_prompt = " . ".join(target_list) + " ."
print(f"   ✅ Prompt DINO : {dino_prompt}\n")

# ========== ÉTAPE 1 : CONSTRUCTION MÉMOIRE (avec filtres qualité) ==========
def build_memory(image_files, target_list):
    print("\n" + "="*55)
    print("📚 ÉTAPE 1 — Construction de la mémoire visuelle")
    print("   (seules les détections nettes et confidentes sont mémorisées)")
    print("="*55)

    memory = {cls: [] for cls in target_list}

    for image_file in tqdm(image_files, desc="Scan mémoire"):
        img_path = os.path.join(IMAGE_DIR, image_file)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue

        H, W = img_bgr.shape[:2]
        _, tensor = load_image(img_path)

        try:
            boxes, logits, phrases = predict(
                model=dino, image=tensor,
                caption=dino_prompt,
                box_threshold=DINO_BOX,
                text_threshold=DINO_TEXT,
                device=DEVICE
            )
        except:
            continue

        for box, logit, phrase in zip(boxes, logits, phrases):
            if logit < 0.5:   # ← confiance DINO trop basse → ignorer
                continue
            matched_cls = None
            for cls in target_list:
                if cls in phrase.lower() or phrase.lower() in cls:
                    matched_cls = cls
                    break
            if matched_cls is None:
                continue

            xc, yc, bw, bh = box.tolist()
            x1 = max(0, int((xc-bw/2)*W))
            y1 = max(0, int((yc-bh/2)*H))
            x2 = min(W, int((xc+bw/2)*W))
            y2 = min(H, int((yc+bh/2)*H))

            if (x2-x1) < 20 or (y2-y1) < 20:
                continue

            crop = img_bgr[y1:y2, x1:x2]
            # Vérifier netteté du crop
            if get_sharpness(crop) < 100:   # ← flou → ne pas mémoriser
                continue

            views = build_augmented_views(crop)
            for view in views:
                try:
                    feat = get_feature(view)
                    memory[matched_cls].append(feat)
                except:
                    continue

    print("\n   📊 Mémoire construite :")
    for cls, feats in memory.items():
        print(f"      {cls:20} : {len(feats):4} vues mémorisées")
        if len(feats) < MIN_MEMORY_SIZE:
            print(f"      ⚠️  Classe '{cls}' a trop peu d'exemples → sera rejetée en phase 2")
    return memory

# ========== ÉTAPE 2 : DÉTECTION AVEC VALIDATION MÉMOIRE ==========
def detect_with_memory(img_path, img_bgr, memory):
    H, W = img_bgr.shape[:2]
    validated = []

    _, tensor = load_image(img_path)
    try:
        boxes, logits, phrases = predict(
            model=dino, image=tensor,
            caption=dino_prompt,
            box_threshold=DINO_BOX,
            text_threshold=DINO_TEXT,
            device=DEVICE
        )
    except Exception as e:
        print(f"   ⚠️  DINO : {e}")
        return []

    for box, logit, phrase in zip(boxes, logits, phrases):
        matched_cls = None
        for cls in target_list:
            if cls in phrase.lower() or phrase.lower() in cls:
                matched_cls = cls
                break
        if matched_cls is None:
            continue

        xc, yc, bw, bh = box.tolist()
        x1 = max(0, int((xc-bw/2)*W))
        y1 = max(0, int((yc-bh/2)*H))
        x2 = min(W, int((xc+bw/2)*W))
        y2 = min(H, int((yc+bh/2)*H))

        if (x2-x1) < 20 or (y2-y1) < 20:
            continue

        crop = img_bgr[y1:y2, x1:x2]
        # Netteté du crop candidat
        if get_sharpness(crop) < 150:
            continue

        mem_feats = memory.get(matched_cls, [])
        # Rejeter si mémoire insuffisante
        if len(mem_feats) < MIN_MEMORY_SIZE:
            continue

        # Test‑time augmentation sur le crop requête
        query_views = build_query_views(crop)
        best_sim = 0.0
        for qv in query_views:
            try:
                qf = get_feature(qv)
                sims = [cosine_similarity(qf, mf) for mf in mem_feats]
                max_sim = max(sims) if sims else 0.0
                if max_sim > best_sim:
                    best_sim = max_sim
            except:
                continue

        if best_sim >= SIMILARITY_MIN:
            validated.append({
                "label": matched_cls,
                "confidence": round(float(logit), 3),
                "similarity": round(best_sim, 3),
                "box": [x1, y1, x2, y2]
            })
    return validated

# ========== NMS ==========
def nms_filter(detections, iou_threshold=0.4):
    if not detections:
        return []
    kept = []
    used = set()
    for i, d1 in enumerate(detections):
        if i in used:
            continue
        x1a, y1a, x2a, y2a = d1["box"]
        for j, d2 in enumerate(detections):
            if i == j or j in used:
                continue
            if d1["label"] != d2["label"]:
                continue
            x1b, y1b, x2b, y2b = d2["box"]
            ix1 = max(x1a, x1b); iy1 = max(y1a, y1b)
            ix2 = min(x2a, x2b); iy2 = min(y2a, y2b)
            inter = max(0, ix2-ix1) * max(0, iy2-iy1)
            area1 = (x2a-x1a) * (y2a-y1a)
            area2 = (x2b-x1b) * (y2b-y1b)
            iou = inter / (area1 + area2 - inter + 1e-6)
            if iou > iou_threshold:
                used.add(j)
        kept.append(d1)
    return kept

# ========== COULEURS ==========
COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 165, 0), (255, 0, 255), (0, 255, 255),
    (255, 255, 0), (128, 0, 128)
]
color_map = {cls: COLORS[i % len(COLORS)] for i, cls in enumerate(target_list)}

# ========== MAIN ==========
image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))
])

if not image_files:
    print(f"❌ Aucune image dans {IMAGE_DIR}")
    exit(1)

print(f"✅ {len(image_files)} images trouvées")

memory = build_memory(image_files, target_list)

print("\n" + "="*55)
print("🔍 ÉTAPE 2 — Détection validée par mémoire visuelle")
print("="*55 + "\n")

all_results = {}

for image_file in tqdm(image_files, desc="Détection"):
    img_path = os.path.join(IMAGE_DIR, image_file)
    base_name = os.path.splitext(image_file)[0]

    img = cv2.imread(img_path)
    if img is None:
        continue

    raw_dets = detect_with_memory(img_path, img, memory)
    detections = nms_filter(raw_dets)

    for det in detections:
        x1, y1, x2, y2 = det["box"]
        cls_name = det["label"]
        conf = det["confidence"]
        sim = det["similarity"]
        color = color_map.get(cls_name, (0, 255, 255))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
        label_text = f"{cls_name} C:{conf:.2f} S:{sim:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img, (x1, max(y1-th-8, 0)), (x1+tw+6, y1), color, -1)
        cv2.putText(img, label_text, (x1+3, max(y1-5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{base_name}_detected.png"), img)

    with open(os.path.join(OUTPUT_DIR, f"{base_name}_detections.txt"), "w") as f:
        for det in detections:
            x1, y1, x2, y2 = det["box"]
            f.write(f"{det['label']} conf:{det['confidence']:.3f} sim:{det['similarity']:.3f} {x1} {y1} {x2} {y2}\n")

    all_results[image_file] = detections

total = sum(len(v) for v in all_results.values())
print(f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
✅ TERMINÉ
   Images traitées  : {len(all_results)}
   Objets détectés  : {total}
   Résultats        : {OUTPUT_DIR}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
