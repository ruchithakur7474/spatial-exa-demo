import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import streamlit as st

from PIL import Image
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VICTIM_PATH = os.path.join(
    BASE_DIR,
    "victim_model_fp16.pth"
)

EXA_GUARD_PATH = os.path.join(
    BASE_DIR,
    "exa_guard_model.pth"
)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# VICTIM MODEL
# ============================================================

NUM_CLASSES = 200


class FaceRecognitionModel(nn.Module):

    def __init__(self, num_classes):

        super().__init__()

        self.backbone = models.resnet18(
            weights=None
        )

        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# EXA-GUARD MODEL
# ============================================================

class EXAGuardNetwork(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self.detector = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 1)
        )

        self.localization = nn.Sequential(

            nn.Conv2d(
                256,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        features = self.features(x)

        detection = self.detector(features)

        localization = self.localization(features)

        localization = F.interpolate(
            localization,
            size=x.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        localization = torch.sigmoid(
            localization
        )

        return detection, localization


# ============================================================
# CHECKPOINT LOADER
# ============================================================

def load_checkpoint(model, path):

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            checkpoint = checkpoint["state_dict"]

        elif "model_state_dict" in checkpoint:

            checkpoint = checkpoint["model_state_dict"]

    cleaned = {}

    for key, value in checkpoint.items():

        key = key.replace(
            "module.",
            ""
        )

        if torch.is_tensor(value):

            if torch.is_floating_point(value):

                value = value.float()

        cleaned[key] = value

    model.load_state_dict(
        cleaned,
        strict=True
    )

    model.to(DEVICE)

    model.eval()

    return model


# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():

    victim = FaceRecognitionModel(
        NUM_CLASSES
    )

    victim = load_checkpoint(
        victim,
        VICTIM_PATH
    )

    exa_guard = EXAGuardNetwork()

    exa_guard = load_checkpoint(
        exa_guard,
        EXA_GUARD_PATH
    )

    return victim, exa_guard


# ============================================================
# PREPROCESSING
# ============================================================

transform = transforms.Compose([

    transforms.Resize(
        (112, 112)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# ============================================================
# FGSM
# ============================================================

def fgsm_attack(
    model,
    image,
    label,
    epsilon=4/255
):

    x = image.clone().detach()
    x.requires_grad = True

    output = model(x)

    loss = F.cross_entropy(
        output,
        label
    )

    model.zero_grad()

    loss.backward()

    perturbed = (
        x +
        epsilon *
        x.grad.sign()
    )

    return perturbed.detach()


# ============================================================
# PGD
# ============================================================

def pgd_attack(
    model,
    image,
    label,
    epsilon=8/255,
    alpha=2/255,
    steps=5
):

    original = image.clone().detach()

    x = original.clone().detach()

    for _ in range(steps):

        x.requires_grad = True

        output = model(x)

        loss = F.cross_entropy(
            output,
            label
        )

        model.zero_grad()

        loss.backward()

        x = x.detach() + (
            alpha *
            x.grad.sign()
        )

        delta = torch.clamp(
            x - original,
            -epsilon,
            epsilon
        )

        x = (
            original + delta
        ).detach()

    return x


# ============================================================
# ADVERSARIAL PATCH
# ============================================================

def patch_attack(image):

    x = image.clone()

    _, _, h, w = x.shape

    patch_size = 28

    y1 = (h - patch_size) // 2
    x1 = (w - patch_size) // 2

    x[
        :,
        :,
        y1:y1 + patch_size,
        x1:x1 + patch_size
    ] = 0.9

    return x


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(
        self,
        model,
        target_layer
    ):

        self.model = model

        self.target_layer = target_layer

        self.activations = None

        self.gradients = None

        target_layer.register_forward_hook(
            self.forward_hook
        )

        target_layer.register_full_backward_hook(
            self.backward_hook
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        image
    ):

        self.model.zero_grad()

        output = self.model(image)

        target = output[:, 0].sum()

        target.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights *
            self.activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam -= cam.min()

        if cam.max() > 0:

            cam /= cam.max()

        return cam.detach().cpu().numpy()


# ============================================================
# STREAMLIT UI
# ============================================================

st.set_page_config(
    page_title="EXA-Guard",
    page_icon="🛡️",
    layout="wide"
)

st.title(
    "🛡️ EXA-Guard"
)

st.subheader(
    "Explainable Adversarial Analytics for Facial Biometric Systems"
)

st.write(
    "Adversarial attack detection, spatial localization "
    "and Grad-CAM based explainability."
)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(VICTIM_PATH):

    st.error(
        "victim_model_fp16.pth not found."
    )

    st.stop()


if not os.path.exists(EXA_GUARD_PATH):

    st.error(
        "exa_guard_model.pth not found."
    )

    st.stop()


# ============================================================
# LOAD MODELS
# ============================================================

try:

    victim_model, exa_guard = load_models()

except Exception as e:

    st.error(
        f"Model loading failed: {e}"
    )

    st.stop()


st.success(
    "EXA-Guard models loaded successfully."
)


# ============================================================
# INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Device",
        str(DEVICE)
    )

with col2:

    st.metric(
        "Input Size",
        "112 × 112"
    )

with col3:

    st.metric(
        "Victim Classes",
        NUM_CLASSES
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload a face image",
    type=["jpg", "jpeg", "png"]
)


attack_type = st.selectbox(
    "Select Attack",
    [
        "Clean",
        "FGSM",
        "PGD",
        "Patch"
    ]
)


if uploaded_file is not None:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.image(
        image,
        caption="Input Image",
        width=300
    )

    x = transform(
        image
    ).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        clean_output = victim_model(x)

        clean_probs = torch.softmax(
            clean_output,
            dim=1
        )

        clean_confidence, clean_class = (
            clean_probs.max(dim=1)
        )

    label = clean_class.detach()

    # --------------------------------------------------------
    # ATTACK
    # --------------------------------------------------------

    if attack_type == "FGSM":

        attacked = fgsm_attack(
            victim_model,
            x,
            label
        )

    elif attack_type == "PGD":

        attacked = pgd_attack(
            victim_model,
            x,
            label
        )

    elif attack_type == "Patch":

        attacked = patch_attack(x)

    else:

        attacked = x

    # --------------------------------------------------------
    # VICTIM PREDICTION
    # --------------------------------------------------------

    with torch.no_grad():

        victim_output = victim_model(
            attacked
        )

        victim_probs = torch.softmax(
            victim_output,
            dim=1
        )

        victim_confidence, victim_class = (
            victim_probs.max(dim=1)
        )

    # --------------------------------------------------------
    # EXA-GUARD
    # --------------------------------------------------------

    with torch.no_grad():

        detection_logits, localization = (
            exa_guard(attacked)
        )

        attack_probability = torch.sigmoid(
            detection_logits
        ).item()

    # --------------------------------------------------------
    # RESULTS
    # --------------------------------------------------------

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Victim Prediction",
            int(victim_class.item())
        )

    with c2:

        st.metric(
            "Victim Confidence",
            f"{victim_confidence.item()*100:.2f}%"
        )

    with c3:

        st.metric(
            "Attack Probability",
            f"{attack_probability*100:.2f}%"
        )


    if attack_probability >= 0.5:

        st.error(
            "🚨 ATTACK DETECTED"
        )

    else:

        st.success(
            "✅ NO ATTACK DETECTED"
        )


    # --------------------------------------------------------
    # LOCALIZATION
    # --------------------------------------------------------

    st.subheader(
        "Spatial Attack Localization"
    )

    localization_map = (
        localization[0, 0]
        .detach()
        .cpu()
        .numpy()
    )

    st.image(
        localization_map,
        caption="EXA-Guard Localization Map",
        clamp=True,
        use_container_width=True
    )


    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    st.subheader(
        "Grad-CAM Explanation"
    )

    try:

        target_layer = (
            exa_guard.features[-1]
        )

        gradcam = GradCAM(
            exa_guard,
            target_layer
        )

        attacked_for_cam = (
            attacked.clone()
            .detach()
            .requires_grad_(True)
        )

        cam = gradcam.generate(
            attacked_for_cam
        )

        st.image(
            cam,
            caption="EXA-Guard Grad-CAM",
            clamp=True,
            use_container_width=True
        )

    except Exception as e:

        st.warning(
            f"Grad-CAM could not be generated: {e}"
        )


# ============================================================
# TECHNICAL DETAILS
# ============================================================

with st.expander(
    "Technical Details"
):

    st.write(
        """
        **Victim Model:** ResNet-18

        **Recognition Classes:** 200

        **Input Resolution:** 112 × 112

        **Attacks:** FGSM, PGD and Adversarial Patch

        **EXA-Guard:** CNN-based attack detector and
        spatial localization network

        **Explainability:** Grad-CAM

        **Localization:** Pixel-level attack localization map
        """
    )


st.caption(
    "EXA-Guard — Explainable Adversarial Analytics for Facial Biometric Systems"
)
