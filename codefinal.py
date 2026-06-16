import subprocess
import cv2
import numpy as np
import os
import hashlib
from pathlib import Path
from PIL import Image
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.cluster import AgglomerativeClustering
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet
import urllib.request
from neuro_low_light import EnhanceModel

# ========== CONFIGURATION ==========
INPUT_VIDEO         = "input/video.mp4"
FRAMES_DIR          = "output/frames"
UNIQUE_DIR          = "output/frames_uniques"
ENHANCED_DIR        = "output/frames_ameliorees"
OUTPUT_VIDEO        = "output/video_finale.mp4"

QUALITY_THRESHOLD   = 100
SHARPNESS_MIN_KEEP  = 300
CLARITY_SCORE_MIN   = 0.10

for d in [FRAMES_DIR, UNIQUE_DIR, ENHANCED_DIR]:
    Path(d).mkdir(parents=True, exist_ok=True)

# ========== MODÈLE CNN ==========
print("⚙️  Chargement ResNet50...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
feature_extractor = torch.nn.Sequential(*list(resnet50.children())[:-1])
feature_extractor.eval().to(DEVICE)

efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
efficientnet.eval().to(DEVICE)

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

preprocess_quality = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print(f"   ✅ ResNet50 + EfficientNet-B0 chargés sur {DEVICE.upper()}")

# ========== FONCTIONS ==========
def normalize_lighting(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l_eq, a, b]), cv2.COLOR_LAB2BGR)

def get_cnn_feature(img):
    img_norm = normalize_lighting(img)
    img_rgb  = cv2.cvtColor(img_norm, cv2.COLOR_BGR2RGB)
    tensor   = preprocess(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        feat = feature_extractor(tensor)
    return feat.squeeze().cpu().numpy()

def get_clarity_score(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = preprocess_quality(Image.fromarray(img_rgb)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        features = efficientnet.features(tensor)
        score = features.var(dim=[2, 3]).mean().item()
    return min(score / 5.0, 1.0)

def get_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    s1 = cv2.Laplacian(gray, cv2.CV_64F).var()
    s2 = cv2.Laplacian(cv2.resize(gray, (gray.shape[1]//2, gray.shape[0]//2)), cv2.CV_64F).var()
    s3 = cv2.Laplacian(cv2.resize(gray, (gray.shape[1]//4, gray.shape[0]//4)), cv2.CV_64F).var()
    return (s1 + s2 + s3) / 3.0

def get_sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

# ========== REAL-ESRGAN ==========
def load_realesrgan():
    model_path = "weights/RealESRGAN_x4plus.pth"
    os.makedirs("weights", exist_ok=True)
    if not os.path.exists(model_path):
        print("   📥 Téléchargement RealESRGAN...")
        url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
        urllib.request.urlretrieve(url, model_path)
        print("   ✅ Modèle téléchargé")
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=256, tile_pad=10, pre_pad=0, half=False)
    return upsampler

print("⚙️  Chargement RealESRGAN...")
realesrgan_model = load_realesrgan()
print("   ✅ RealESRGAN chargé")

def super_resolve(img, target_w=1920, target_h=1080):
    try:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output, _ = realesrgan_model.enhance(img_rgb, outscale=4)
        img = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"   ⚠️  RealESRGAN échoué ({e}), fallback Lanczos4")
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    h, w = img.shape[:2]
    if w != target_w or h != target_h:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    blurred = cv2.GaussianBlur(img, (0, 0), 1.5)
    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

# ========== EXTRACTION FRAMES ==========
def extract_frames():
    print("\n📹 Extraction des frames...")
    for f in os.listdir(FRAMES_DIR):
        os.remove(os.path.join(FRAMES_DIR, f))
    subprocess.run([
        "ffmpeg", "-i", INPUT_VIDEO, "-vsync", "0",
        f"{FRAMES_DIR}/frame_%06d.png"
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = sorted([f for f in os.listdir(FRAMES_DIR) if f.endswith('.png')])
    print(f"   → {len(frames)} frames extraites")
    return frames

# ========== DÉDUPLICATION ==========
def remove_duplicates(frame_list):
    print("\n🔬 Phase 1 — Analyse CNN (ResNet50 + EfficientNet)...")
    valid_frames  = []
    unique_sha    = {}
    skipped_blur  = 0
    skipped_sha   = 0
    skipped_cnn   = 0

    for i, frame in enumerate(frame_list):
        path = os.path.join(FRAMES_DIR, frame)
        img  = cv2.imread(path)
        if img is None:
            continue
        sharp = get_sharpness(img)
        if sharp < QUALITY_THRESHOLD:
            skipped_blur += 1
            continue
        sha = get_sha256(path)
        if sha in unique_sha:
            skipped_sha += 1
            continue
        unique_sha[sha] = frame
        clarity = get_clarity_score(img)
        if clarity < CLARITY_SCORE_MIN:
            skipped_cnn += 1
            continue
        feat = get_cnn_feature(img)
        valid_frames.append((frame, img, feat, sharp, clarity))
        if (i + 1) % 50 == 0:
            pct = (i + 1) / len(frame_list) * 100
            print(f"   [{pct:5.1f}%] {i+1}/{len(frame_list)} | Valides: {len(valid_frames)}")

    print(f"   → {len(valid_frames)} frames valides après triple filtrage")
    if not valid_frames:
        return []

    print("\n🔍 Phase 2 — Clustering hiérarchique (ResNet50 2048-dims)...")
    features = np.array([f[2] for f in valid_frames])
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms == 0] = 1
    features_norm = features / norms
    n_total = len(valid_frames)
    n_clust = max(5, int(n_total * (1 - 0.90)))
    print(f"   Frames valides  : {n_total}")
    print(f"   Clusters cibles : {n_clust}  (~{(1 - n_clust/n_total)*100:.0f}% réduction)")

    clustering = AgglomerativeClustering(n_clusters=n_clust, metric='cosine', linkage='average')
    labels = clustering.fit_predict(features_norm)

    print("\n✂️  Phase 3 — Sélection par score combiné (netteté × clarté CNN)...")
    cluster_best = {}
    for idx, (frame_name, img, feat, sharp, clarity) in enumerate(valid_frames):
        label = labels[idx]
        combined_score = sharp * clarity
        if label not in cluster_best or combined_score > cluster_best[label][3]:
            cluster_best[label] = (frame_name, img, sharp, combined_score)

    kept_frames = []
    skipped_final = 0
    for label, (frame_name, img, sharp, score) in cluster_best.items():
        if sharp < SHARPNESS_MIN_KEEP:
            skipped_final += 1
            continue
        img_hd = super_resolve(img, 1920, 1080)
        kept_frames.append((frame_name, img_hd))

    frame_order = {f[0]: i for i, f in enumerate(valid_frames)}
    kept_frames.sort(key=lambda x: frame_order.get(x[0], 0))
    actual_reduction = (1 - len(kept_frames) / max(len(frame_list), 1)) * 100

    print(f"""
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   📊 Résultats :
      Total extraites      : {len(frame_list)}
      Floues (Laplacian)   : {skipped_blur}
      Doublons exacts      : {skipped_sha}
      Rejetées (CNN)       : {skipped_cnn}
      Clusters             : {n_clust}
      Floues (sortie)      : {skipped_final}
      ✅ Frames Full HD     : {len(kept_frames)}
      📉 Réduction          : {actual_reduction:.1f}%
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")
    return kept_frames

# ========== ENHANCE IMAGE (Deep Learning + Tonemap + Sharpening) ==========
print("⚙️  Chargement du modèle d'illumination Zero-DCE++...")
enhancer = EnhanceModel(device=DEVICE)
print("   ✅ Modèle chargé")

def enhance_image(img):
    """
    Combines:
    1. Multi‑scale CLAHE (local contrast)
    2. Reinhard tonemap (removes sun rays)
    3. Zero-DCE++ AI (global illumination balance)
    4. Unsharp mask + saturation (text & objects sharp)
    """
    # ----- 1. Multi‑scale CLAHE on LAB L channel -----
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe1 = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    l1 = clahe1.apply(l)
    l2 = clahe2.apply(l)
    l_eq = cv2.addWeighted(l1, 0.7, l2, 0.3, 0)
    l_eq = cv2.equalizeHist(l_eq)  # extra brightness boost
    lab_eq = cv2.merge([l_eq, a, b])
    img_clahe = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    # ----- 2. Reinhard tonemap – kills sun rays (highlights compression) -----
    img_float = img_clahe.astype(np.float32) / 255.0
    tonemap = cv2.createTonemapReinhard(intensity=3.0, light_adapt=1.0, color_adapt=0.0)
    img_tonemap = tonemap.process(img_float)
    img_tonemap = np.clip(img_tonemap * 255, 0, 255).astype(np.uint8)

    # ----- 3. Zero-DCE++ AI (improves shadows and overall clarity) -----
    img_rgb = cv2.cvtColor(img_tonemap, cv2.COLOR_BGR2RGB)
    temp_input = "temp_frame_input.png"
    temp_output = "temp_frame_enhanced.png"
    cv2.imwrite(temp_input, img_rgb)
    enhancer.enhance_image(temp_input, temp_output)
    img_ai = cv2.imread(temp_output)   # already RGB
    os.remove(temp_input)
    os.remove(temp_output)
    img_ai = cv2.cvtColor(img_ai, cv2.COLOR_RGB2BGR)

    # ----- 4. Final sharpening + saturation + contrast -----
    # Strong unsharp mask
    blurred = cv2.GaussianBlur(img_ai, (0, 0), 2.5)
    img_sharp = cv2.addWeighted(img_ai, 2.2, blurred, -1.2, 0)

    # Adaptive gamma lift (if still dark)
    gray = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2GRAY)
    mean_bright = gray.mean()
    if mean_bright < 120:
        gamma = 0.75
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
        img_sharp = cv2.LUT(img_sharp, table)

    # Final contrast & saturation boost
    img_sharp = cv2.convertScaleAbs(img_sharp, alpha=1.2, beta=8)
    hsv = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.convertScaleAbs(s, alpha=1.3)   # more color
    v = cv2.convertScaleAbs(v, alpha=1.1)   # more brightness
    img_final = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)

    return img_final

# ========== SAVE FRAMES ==========
def enhance_and_save(kept_frames):
    print("\n✨ Application de l'amélioration d'illumination...")
    for f in os.listdir(UNIQUE_DIR):
        os.remove(os.path.join(UNIQUE_DIR, f))
    for f in os.listdir(ENHANCED_DIR):
        os.remove(os.path.join(ENHANCED_DIR, f))

    saved = 0
    skipped = 0
    for i, (frame_name, img) in enumerate(kept_frames, 1):
        sharp = get_sharpness(img)
        if sharp < SHARPNESS_MIN_KEEP:
            skipped += 1
            continue
        cv2.imwrite(os.path.join(UNIQUE_DIR, frame_name), img)
        enhanced = enhance_image(img)
        saved += 1
        cv2.imwrite(os.path.join(ENHANCED_DIR, f"frame_{saved:06d}.png"), enhanced)
        if saved % 20 == 0:
            print(f"   Sauvegardé : {saved}")

    print(f"""
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   ✅ Frames claires sauvegardées : {saved}
   🗑️  Frames floues supprimées   : {skipped}
   📁 Dossier unique              : {UNIQUE_DIR}
   📁 Dossier Full HD             : {ENHANCED_DIR}
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━""")

# ========== REBUILD VIDEO ==========
def rebuild_video():
    print("\n🎬 Reconstruction vidéo...")
    subprocess.run([
        "ffmpeg", "-framerate", "5",
        "-i", f"{ENHANCED_DIR}/frame_%06d.png",
        "-c:v", "libx264", "-crf", "16",
        "-pix_fmt", "yuv420p", "-y", OUTPUT_VIDEO
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"   ✅ {OUTPUT_VIDEO}")

# ========== MAIN ==========
if __name__ == "__main__":
    print("=" * 60)
    print("🚀  ResNet50 + EfficientNet — 90% RÉDUCTION + FULL HD")
    print("=" * 60)
    if not os.path.exists(INPUT_VIDEO):
        print(f"❌ Introuvable : {INPUT_VIDEO}")
        exit(1)
    try:
        frames = extract_frames()
        kept = remove_duplicates(frames)
        if not kept:
            print("❌ Aucune frame valide trouvée.")
            exit(1)
        enhance_and_save(kept)
        rebuild_video()
        print(f"\n✅ TERMINÉ — {len(frames)} → {len(kept)} frames Full HD")
    except Exception as e:
        import traceback
        traceback.print_exc()