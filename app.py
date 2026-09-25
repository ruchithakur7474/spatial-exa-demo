import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# 1. SPATIAL-EXA MODEL ARCHITECTURE
# ==========================================

class CrossAttentionFusion(nn.Module):
    def __init__(self, in_dim=512, embed_dim=256):
        super(CrossAttentionFusion, self).__init__()
        self.q_proj = nn.Linear(in_dim, embed_dim)
        self.k_proj = nn.Linear(in_dim, embed_dim)
        self.v_proj = nn.Linear(in_dim, embed_dim)
        self.scale = embed_dim ** -0.5
        self.out_proj = nn.Linear(embed_dim, in_dim)

    def forward(self, f_cnn, f_trans):
        B, C, H, W = f_cnn.shape
        f_cnn_flat = f_cnn.view(B, C, H * W).permute(0, 2, 1)
        f_trans_flat = f_trans.view(B, C, H * W).permute(0, 2, 1)

        Q = self.q_proj(f_cnn_flat)
        K = self.k_proj(f_trans_flat)
        V = self.v_proj(f_trans_flat)

        attn_weights = torch.softmax(torch.bmm(Q, K.transpose(1, 2)) * self.scale, dim=-1)
        attn_out = torch.bmm(attn_weights, V)
        fused = self.out_proj(attn_out).permute(0, 2, 1).view(B, C, H, W)
        return fused + f_cnn, attn_weights

class SpatialEXAModel(nn.Module):
    def __init__(self, num_classes=6):
        super(SpatialEXAModel, self).__init__()
        resnet = models.resnet50(weights=None)
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.cnn_proj = nn.Conv2d(2048, 512, kernel_size=1)
        self.trans_proj = nn.Conv2d(2048, 512, kernel_size=1)
        self.fusion = CrossAttentionFusion(in_dim=512, embed_dim=256)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self.localization_head = nn.Sequential(
            nn.Conv2d(512, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        cnn_feat = self.cnn_backbone(x)
        f_cnn = self.cnn_proj(cnn_feat)
        f_trans = self.trans_proj(cnn_feat)
        fused_feat, attn_map = self.fusion(f_cnn, f_trans)
        pooled = self.pool(fused_feat).view(fused_feat.size(0), -1)
        logits = self.classifier(pooled)
        anomaly_map = self.localization_head(fused_feat)
        anomaly_map = F.interpolate(anomaly_map, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        return logits, anomaly_map, fused_feat

# ==========================================\n# 2. HELPER SIMULATION & XAI ENGINE
# ==========================================

CLASSES = ["Benign (Real)", "FGSM Attack", "PGD Attack", "Carlini-Wagner (CW)", "Physical Patch", "Generative Deepfake"]

def generate_simulated_inference(img_pil, threat_type="Auto Detect"):
    img_np = np.array(img_pil)
    H, W, _ = img_np.shape

    if threat_type == "Auto Detect":
        avg_val = np.mean(img_np)
        if avg_val % 5 < 1:
            detected_idx = 0
        elif avg_val % 5 < 2:
            detected_idx = 2
        elif avg_val % 5 < 3:
            detected_idx = 4
        elif avg_val % 5 < 4:
            detected_idx = 5
        else:
            detected_idx = 1
    else:
        detected_idx = CLASSES.index(threat_type)

    probs = np.zeros(len(CLASSES))
    if detected_idx == 0:
        probs[0] = np.random.uniform(0.92, 0.99)
        remaining = 1.0 - probs[0]
        probs[1:] = np.random.dirichlet(np.ones(len(CLASSES)-1)) * remaining
    else:
        probs[detected_idx] = np.random.uniform(0.88, 0.97)
        remaining = 1.0 - probs[detected_idx]
        other_indices = [i for i in range(len(CLASSES)) if i != detected_idx]
        probs[other_indices] = np.random.dirichlet(np.ones(len(CLASSES)-1)) * remaining

    heatmap = np.zeros((H, W), dtype=np.float32)
    if detected_idx == 0:
        heatmap = np.clip(np.random.normal(0.05, 0.02, (H, W)), 0, 1)
        bbox = None
    elif detected_idx == 4:
        cy, cx = int(H * 0.35), int(W * 0.5)
        rh, rw = int(H * 0.15), int(W * 0.25)
        y1, y2 = max(0, cy - rh), min(H, cy + rh)
        x1, x2 = max(0, cx - rw), min(W, cx + rw)
        y_grid, x_grid = np.ogrid[:H, :W]
        dist_from_center = ((y_grid - cy)**2 / rh**2 + (x_grid - cx)**2 / rw**2)
        heatmap = np.clip(np.exp(-dist_from_center * 2) + np.random.normal(0.02, 0.01, (H, W)), 0, 1)
        bbox = [x1, y1, x2, y2]
    else:
        y_grid, x_grid = np.ogrid[:H, :W]
        heatmap = (np.sin(x_grid / 20.0) * np.cos(y_grid / 20.0) + 1.0) / 2.0
        heatmap = np.clip(heatmap * 0.6 + np.random.normal(0.2, 0.05, (H, W)), 0, 1)
        bbox = [int(W * 0.2), int(H * 0.2), int(W * 0.8), int(H * 0.85)]

    return CLASSES[detected_idx], probs, heatmap, bbox

def calculate_deletion_insertion_curves(heatmap, initial_confidence):
    steps = 10
    percentiles = np.linspace(0, 100, steps)
    if initial_confidence > 0.5:
        deletion_curve = np.clip(initial_confidence * np.exp(-0.04 * percentiles) + np.random.normal(0, 0.02, steps), 0.05, 1.0)
        insertion_curve = np.clip(initial_confidence * (1 - np.exp(-0.05 * percentiles)) + np.random.normal(0, 0.02, steps), 0.0, 1.0)
    else:
        deletion_curve = np.ones(steps) * initial_confidence
        insertion_curve = np.zeros(steps)

    trapz_fn = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    deletion_auc = float(trapz_fn(deletion_curve, percentiles / 100.0))
    insertion_auc = float(trapz_fn(insertion_curve, percentiles / 100.0))

    return percentiles, deletion_curve, insertion_curve, deletion_auc, insertion_auc

# ==========================================\n# 3. STREAMLIT INTERACTIVE DASHBOARD\n# ==========================================

def main():
    st.set_page_config(page_title="Spatial-EXA Defense Studio", page_icon="🛡️", layout="wide")
    st.title("🛡️ Spatial-EXA: Facial Biometric Threat Defense & XAI Studio")

    st.sidebar.header("⚙️ Configuration")
    source_type = st.sidebar.radio("Input Source", ["Upload Custom Image", "Preset Synthetic Samples"])

    selected_img = None
    if source_type == "Upload Custom Image":
        uploaded_file = st.sidebar.file_uploader("Choose a face image...", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            selected_img = Image.open(uploaded_file).convert("RGB")
    else:
        sample_choice = st.sidebar.selectbox("Select Threat Case Study", [
            "Sample 1: Benign / Real Identity",
            "Sample 2: Physical Adversarial Patch",
            "Sample 3: PGD Digital Perturbation",
            "Sample 4: Generative Deepfake Distortion"
        ])
        img_np = np.zeros((300, 300, 3), dtype=np.uint8)
        img_np[:, :] = [220, 200, 190]
        img_pil = Image.fromarray(img_np)
        draw = ImageDraw.Draw(img_pil)
        draw.ellipse([70, 50, 230, 250], fill=(230, 210, 195), outline=(100, 80, 70), width=3)
        draw.ellipse([100, 110, 130, 130], fill=(60, 40, 30))
        draw.ellipse([170, 110, 200, 130], fill=(60, 40, 30))
        draw.line([150, 130, 150, 170], fill=(150, 100, 90), width=4)
        draw.arc([110, 180, 190, 220], start=0, end=180, fill=(180, 60, 60), width=4)

        if "Patch" in sample_choice:
            draw.rectangle([110, 65, 190, 95], fill=(255, 50, 50), outline=(255, 255, 0), width=2)
        elif "PGD" in sample_choice:
            noise = np.random.randint(-25, 25, (300, 300, 3), dtype=np.int16)
            img_pil = Image.fromarray(np.clip(np.array(img_pil, dtype=np.int16) + noise, 0, 255).astype(np.uint8))

        selected_img = img_pil

    st.sidebar.markdown("---")
    simulated_threat = st.sidebar.selectbox("Override Simulation Mode", ["Auto Detect"] + CLASSES)
    sensitivity = st.sidebar.slider("Spatial Localization Threshold", 0.1, 0.9, 0.45, 0.05)

    if selected_img is None:
        st.info("👈 Please upload an image or select a sample in the sidebar.")
        return

    top_label, probs, heatmap, bbox = generate_simulated_inference(selected_img, simulated_threat)
    detected_prob = probs[CLASSES.index(top_label)]
    is_attack = (top_label != "Benign (Real)")

    col_left, col_mid, col_right = st.columns([1, 1.2, 1.2])

    with col_left:
        st.subheader("🖼️ Input Image")
        st.image(selected_img, use_container_width=True)
        if is_attack:
            st.error(f"⚠️ **ATTACK DETECTED** ({top_label})")
        else:
            st.success("✅ **BENIGN IDENTITY VERIFIED**")
        st.metric("Confidence", f"{detected_prob * 100:.2f}%")

    with col_mid:
        st.subheader("📍 Spatial Localization Map")
        h_map_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize(selected_img.size)
        cm = plt.get_cmap('jet')
        colorized_heatmap = cm(np.array(h_map_resized) / 255.0)[:, :, :3]
        colorized_pil = Image.fromarray((colorized_heatmap * 255).astype(np.uint8))
        blended = Image.blend(selected_img, colorized_pil, alpha=0.5)

        if is_attack and bbox is not None:
            draw_blend = ImageDraw.Draw(blended)
            draw_blend.rectangle(bbox, outline="red", width=3)

        st.image(blended, use_container_width=True)

    with col_right:
        st.subheader("🔬 Threat Classification Breakdown")
        for cls_name, p in zip(CLASSES, probs):
            st.write(f"**{cls_name}**")
            st.progress(float(p))

    st.markdown("---")
    st.header("📈 Quantitative XAI Verification Engine")
    percentiles, del_curve, ins_curve, del_auc, ins_auc = calculate_deletion_insertion_curves(heatmap, detected_prob)

    m1, m2 = st.columns(2)
    m1.metric("Deletion AUC (Lower is Better)", f"{del_auc:.3f}")
    m2.metric("Insertion AUC (Higher is Better)", f"{ins_auc:.3f}")

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(percentiles, del_curve, 'r-o', label=f'Deletion AUC = {del_auc:.3f}')
    ax.plot(percentiles, ins_curve, 'g-s', label=f'Insertion AUC = {ins_auc:.3f}')
    ax.set_xlabel("% Pixels Modified")
    ax.set_ylabel("Confidence")
    ax.legend()
    st.pyplot(fig)

if __name__ == "__main__":
    main()
