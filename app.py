# ============================================================
# EXA-GUARD
# Explainable Adversarial Analytics for Facial Biometrics
# ============================================================

import os
import traceback
from pathlib import Path

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18


# ============================================================
# 1. CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EXA-Guard",
    page_icon="🔍",
    layout="wide"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 112
NUM_CLASSES = 200

FGSM_EPSILON = 8 / 255

PGD_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

PATCH_SIZE = 20

VICTIM_CHECKPOINT = BASE_DIR / "victim_model_fp16.pth"

EXA_CHECKPOINT = BASE_DIR / "exa_guard_model.pth"


# ============================================================
# 2. CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 800;
        text-align: center;
        color: #2563eb;
    }

    .subtitle {
        text-align: center;
        color: #64748b;
        font-size: 16px;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 23px;
        font-weight: bold;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])


def image_to_tensor(image):

    image = image.convert("RGB")

    tensor = transform(image)

    tensor = tensor.unsqueeze(0)

    return tensor.to(DEVICE)


def tensor_to_image(tensor):

    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze(0)

    tensor = tensor * 0.5 + 0.5

    tensor = torch.clamp(tensor, 0, 1)

    image = tensor.permute(1, 2, 0).numpy()

    image = (image * 255).astype(np.uint8)

    return Image.fromarray(image)


# ============================================================
# 4. VICTIM MODEL
# ============================================================

class VictimModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.model = resnet18(
            weights=None
        )

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# 5. EXA-GUARD MODEL
# MATCHES YOUR CHECKPOINT ARCHITECTURE
# ============================================================

class EXAGuardModel(nn.Module):

    def __init__(self):

        super().__init__()

        # -----------------------------------------
        # Feature extraction
        # -----------------------------------------

        self.features = nn.Sequential(

            # Layer 0
            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            # Layer 1
            nn.BatchNorm2d(32),

            # Layer 2
            nn.ReLU(inplace=True),

            # Layer 3
            nn.MaxPool2d(2),

            # Layer 4
            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            # Layer 5
            nn.BatchNorm2d(64),

            # Layer 6
            nn.ReLU(inplace=True),

            # Layer 7
            nn.MaxPool2d(2),

            # Layer 8
            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            # Layer 9
            nn.BatchNorm2d(128),

            # Layer 10
            nn.ReLU(inplace=True),

            # Layer 11
            nn.MaxPool2d(2),

            # Layer 12
            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),

            # Layer 13
            nn.BatchNorm2d(256),

            # Layer 14
            nn.ReLU(inplace=True)
        )

        # -----------------------------------------
        # Attack detector
        # -----------------------------------------

        self.detector = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(
                128,
                1
            )
        )

        # -----------------------------------------
        # Spatial localization
        # -----------------------------------------

        self.localization = nn.Sequential(

            nn.Conv2d(
                256,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        features = self.features(x)

        # Attack classification
        attack_logits = self.detector(features)

        attack_probability = torch.sigmoid(
            attack_logits
        )

        # Spatial localization
        localization_logits = self.localization(
            features
        )

        localization_map = torch.sigmoid(
            localization_logits
        )

        localization_map = F.interpolate(
            localization_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return (
            attack_probability,
            localization_map
        )


# ============================================================
# 6. CHECKPOINT LOADER
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint:

            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint:

            return checkpoint["model_state_dict"]

        if "model" in checkpoint:

            model_object = checkpoint["model"]

            if isinstance(model_object, nn.Module):

                return model_object.state_dict()

            return model_object

        return checkpoint

    if isinstance(checkpoint, nn.Module):

        return checkpoint.state_dict()

    raise TypeError(
        "Unsupported checkpoint format."
    )


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):

            new_key = new_key[len("module."):]

        if new_key.startswith("model."):

            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def load_checkpoint(model, checkpoint_path):

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():

        return False, (
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        state_dict = extract_state_dict(
            checkpoint
        )

        state_dict = clean_state_dict(
            state_dict
        )

        # Strict loading for EXA-Guard architecture

        result = model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(DEVICE)

        model.eval()

        return True, (
            "Checkpoint loaded successfully.\n"
            f"Missing keys: {len(result.missing_keys)}\n"
            f"Unexpected keys: {len(result.unexpected_keys)}"
        )

    except Exception as e:

        return False, str(e)


# ============================================================
# 7. INITIALIZE MODELS
# ============================================================

@st.cache_resource
def initialize_models():

    victim_model = VictimModel(
        NUM_CLASSES
    )

    exa_guard_model = EXAGuardModel()

    victim_model.to(DEVICE)

    exa_guard_model.to(DEVICE)

    victim_model.eval()

    exa_guard_model.eval()

    victim_status, victim_message = load_checkpoint(
        victim_model,
        VICTIM_CHECKPOINT
    )

    exa_status, exa_message = load_checkpoint(
        exa_guard_model,
        EXA_CHECKPOINT
    )

    return (
        victim_model,
        exa_guard_model,
        victim_status,
        victim_message,
        exa_status,
        exa_message
    )


(
    victim_model,
    exa_guard_model,
    victim_status,
    victim_message,
    exa_status,
    exa_message
) = initialize_models()


# ============================================================
# 8. FGSM
# ============================================================

def generate_fgsm_attack(image_tensor):

    image_tensor = image_tensor.clone().detach()

    image_tensor.requires_grad = True

    output = victim_model(
        image_tensor
    )

    predicted_class = output.argmax(
        dim=1
    )

    loss = F.cross_entropy(
        output,
        predicted_class
    )

    victim_model.zero_grad()

    loss.backward()

    gradient = image_tensor.grad.data

    attacked_image = (
        image_tensor +
        FGSM_EPSILON * gradient.sign()
    )

    attacked_image = torch.clamp(
        attacked_image,
        -1,
        1
    )

    return attacked_image.detach()


# ============================================================
# 9. PGD
# ============================================================

def generate_pgd_attack(image_tensor):

    original_image = image_tensor.clone().detach()

    attacked_image = original_image.clone().detach()

    attacked_image += torch.empty_like(
        attacked_image
    ).uniform_(
        -PGD_EPSILON,
        PGD_EPSILON
    )

    attacked_image = torch.clamp(
        attacked_image,
        -1,
        1
    )

    target_class = victim_model(
        original_image
    ).argmax(
        dim=1
    )

    for _ in range(PGD_STEPS):

        attacked_image.requires_grad = True

        output = victim_model(
            attacked_image
        )

        loss = F.cross_entropy(
            output,
            target_class
        )

        victim_model.zero_grad()

        loss.backward()

        gradient = attacked_image.grad.data

        attacked_image = (
            attacked_image.detach() +
            PGD_ALPHA * gradient.sign()
        )

        perturbation = (
            attacked_image -
            original_image
        )

        perturbation = torch.clamp(
            perturbation,
            -PGD_EPSILON,
            PGD_EPSILON
        )

        attacked_image = (
            original_image +
            perturbation
        )

        attacked_image = torch.clamp(
            attacked_image,
            -1,
            1
        )

    return attacked_image.detach()


# ============================================================
# 10. SIMULATED PATCH
# ============================================================

def generate_patch_attack(image_tensor):

    attacked_image = image_tensor.clone().detach()

    _, _, height, width = attacked_image.shape

    patch_size = min(
        PATCH_SIZE,
        height,
        width
    )

    start_x = (
        width - patch_size
    ) // 2

    start_y = (
        height - patch_size
    ) // 2

    patch = torch.zeros(
        1,
        3,
        patch_size,
        patch_size,
        device=DEVICE
    )

    patch[:, 0] = 1.0

    patch[:, 1] = -1.0

    patch[:, 2] = 1.0

    attacked_image[
        :,
        :,
        start_y:start_y + patch_size,
        start_x:start_x + patch_size
    ] = patch

    return attacked_image.detach()


# ============================================================
# 11. ATTACK SELECTION
# ============================================================

def apply_attack(image_tensor, attack_name):

    if attack_name == "Clean":

        return image_tensor.clone().detach()

    if attack_name == "FGSM":

        if not victim_status:

            raise RuntimeError(
                "Victim model checkpoint is unavailable. "
                "FGSM cannot be generated."
            )

        return generate_fgsm_attack(
            image_tensor
        )

    if attack_name == "PGD":

        if not victim_status:

            raise RuntimeError(
                "Victim model checkpoint is unavailable. "
                "PGD cannot be generated."
            )

        return generate_pgd_attack(
            image_tensor
        )

    if attack_name == "Patch Attack":

        return generate_patch_attack(
            image_tensor
        )

    return image_tensor.clone().detach()


# ============================================================
# 12. VICTIM PREDICTION
# ============================================================

def get_victim_prediction(image_tensor):

    with torch.no_grad():

        output = victim_model(
            image_tensor
        )

        probabilities = F.softmax(
            output,
            dim=1
        )

        confidence, predicted_class = probabilities.max(
            dim=1
        )

    return (
        predicted_class.item(),
        confidence.item()
    )


# ============================================================
# 13. EXA-GUARD PREDICTION
# ============================================================

def get_exa_guard_prediction(image_tensor):

    with torch.no_grad():

        attack_probability, localization_map = (
            exa_guard_model(
                image_tensor
            )
        )

    attack_probability = (
        attack_probability.squeeze().item()
    )

    localization_map = (
        localization_map.squeeze()
        .detach()
        .cpu()
        .numpy()
    )

    return (
        attack_probability,
        localization_map
    )


# ============================================================
# 14. HEATMAP NORMALIZATION
# ============================================================

def normalize_heatmap(heatmap):

    heatmap = np.asarray(
        heatmap,
        dtype=np.float32
    )

    heatmap = np.nan_to_num(
        heatmap
    )

    minimum = heatmap.min()

    maximum = heatmap.max()

    if maximum - minimum < 1e-8:

        return np.zeros_like(
            heatmap,
            dtype=np.float32
        )

    return (
        heatmap - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 15. COLORED HEATMAP
# ============================================================

def create_colored_heatmap(heatmap):

    heatmap = normalize_heatmap(
        heatmap
    )

    red = np.clip(
        1.5 - np.abs(4 * heatmap - 3),
        0,
        1
    )

    green = np.clip(
        1.5 - np.abs(4 * heatmap - 2),
        0,
        1
    )

    blue = np.clip(
        1.5 - np.abs(4 * heatmap - 1),
        0,
        1
    )

    colored = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    colored = (
        colored * 255
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        colored
    )


# ============================================================
# 16. HEATMAP OVERLAY
# ============================================================

def make_heatmap_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    original_image = original_image.convert(
        "RGB"
    )

    original_array = np.asarray(
        original_image
    ).astype(
        np.float32
    )

    colored_heatmap = create_colored_heatmap(
        heatmap
    )

    colored_heatmap = colored_heatmap.resize(
        original_image.size
    )

    heatmap_array = np.asarray(
        colored_heatmap
    ).astype(
        np.float32
    )

    overlay = (
        (1 - alpha) * original_array +
        alpha * heatmap_array
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(
        np.uint8
    )

    return (
        colored_heatmap,
        Image.fromarray(
            overlay
        )
    )


# ============================================================
# 17. GRAD-CAM
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

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        self.activations = output

        if output.requires_grad:

            output.register_hook(
                self.gradient_hook
            )

    def gradient_hook(
        self,
        gradient
    ):

        self.gradients = gradient

    def generate(
        self,
        input_tensor,
        target_class=None
    ):

        self.model.zero_grad()

        self.activations = None

        self.gradients = None

        input_tensor = input_tensor.clone().detach()

        input_tensor.requires_grad = True

        output = self.model(
            input_tensor
        )

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        if not isinstance(
            output,
            torch.Tensor
        ):

            raise TypeError(
                "Model output must be a Tensor."
            )

        if output.ndim != 2:

            raise ValueError(
                f"Expected 2D classification output. "
                f"Received shape: {output.shape}"
            )

        if target_class is None:

            target_class = output.argmax(
                dim=1
            ).item()

        score = output[
            0,
            target_class
        ]

        score.backward()

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activation was not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradient was not captured."
            )

        activations = self.activations

        gradients = self.gradients

        if activations.ndim != 4:

            raise ValueError(
                f"Invalid activation dimensions: "
                f"{activations.shape}"
            )

        if gradients.ndim != 4:

            raise ValueError(
                f"Invalid gradient dimensions: "
                f"{gradients.shape}"
            )

        weights = gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        cam = (
            weights * activations
        ).sum(
            dim=1,
            keepdim=True
        )

        cam = F.relu(
            cam
        )

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze()

        cam = cam.detach().cpu().numpy()

        cam = normalize_heatmap(
            cam
        )

        return cam

    def remove_hooks(self):

        self.forward_handle.remove()


def generate_gradcam(image_tensor):

    try:

        target_layer = (
            victim_model.model.layer4[-1]
        )

        gradcam = GradCAM(
            victim_model,
            target_layer
        )

        cam = gradcam.generate(
            image_tensor
        )

        gradcam.remove_hooks()

        return cam, None

    except Exception as e:

        return None, str(e)


# ============================================================
# 18. PERTURBATION MAP
# ============================================================

def calculate_difference(
    clean_image,
    attacked_image
):

    clean_array = np.asarray(
        clean_image
    ).astype(
        np.float32
    )

    attacked_array = np.asarray(
        attacked_image
    ).astype(
        np.float32
    )

    difference = np.abs(
        clean_array -
        attacked_array
    )

    difference = difference.mean(
        axis=2
    )

    return normalize_heatmap(
        difference
    )


# ============================================================
# 19. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">EXA-GUARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Explainable Adversarial Analytics for Facial Biometric Security
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 20. SIDEBAR
# ============================================================

st.sidebar.title(
    "EXA-Guard Settings"
)

st.sidebar.write(
    f"Device: {DEVICE}"
)

st.sidebar.write(
    f"Image Size: {IMAGE_SIZE} × {IMAGE_SIZE}"
)

st.sidebar.divider()

st.sidebar.subheader(
    "Model Status"
)

if victim_status:

    st.sidebar.success(
        "Victim Model Loaded"
    )

else:

    st.sidebar.error(
        "Victim Model Not Loaded"
    )

    st.sidebar.code(
        victim_message
    )

if exa_status:

    st.sidebar.success(
        "EXA-Guard Model Loaded"
    )

else:

    st.sidebar.error(
        "EXA-Guard Model Not Loaded"
    )

    st.sidebar.code(
        exa_message
    )

    st.sidebar.write(
        "Checkpoint path:"
    )

    st.sidebar.code(
        str(EXA_CHECKPOINT)
    )

    st.sidebar.write(
        "File exists:",
        EXA_CHECKPOINT.exists()
    )


# ============================================================
# 21. RESEARCH NOTICE
# ============================================================

st.info(
    """
    Research prototype demonstrating adversarial attack generation,
    attack detection, spatial localization and explainability.

    Outputs are dependent on the availability of trained checkpoints.
    """
)


# ============================================================
# 22. IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Facial Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 23. ATTACK SELECTOR
# ============================================================

attack_name = st.selectbox(
    "Select Attack Scenario",
    [
        "Clean",
        "FGSM",
        "PGD",
        "Patch Attack"
    ]
)


# ============================================================
# 24. MAIN PROCESSING
# ============================================================

if uploaded_file is not None:

    try:

        original_image = Image.open(
            uploaded_file
        ).convert(
            "RGB"
        )

        image_tensor = image_to_tensor(
            original_image
        )

        st.markdown(
            '<div class="section-title">Original Image</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            width=350
        )

        # Generate attack

        attacked_tensor = apply_attack(
            image_tensor,
            attack_name
        )

        attacked_image = tensor_to_image(
            attacked_tensor
        )

        # ----------------------------------------------------
        # CLEAN VS ATTACKED
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Clean vs Attacked Image'
            '</div>',
            unsafe_allow_html=True
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Clean Image"
            )

            st.image(
                tensor_to_image(
                    image_tensor
                ),
                use_container_width=True
            )

        with col2:

            st.subheader(
                attack_name
            )

            st.image(
                attacked_image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # VICTIM MODEL
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Victim Model Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        if victim_status:

            clean_prediction, clean_confidence = (
                get_victim_prediction(
                    image_tensor
                )
            )

            attacked_prediction, attacked_confidence = (
                get_victim_prediction(
                    attacked_tensor
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.metric(
                    "Clean Prediction",
                    clean_prediction
                )

            with col2:

                st.metric(
                    "Attacked Prediction",
                    attacked_prediction
                )

            with col3:

                st.metric(
                    "Attacked Confidence",
                    f"{attacked_confidence * 100:.2f}%"
                )

            if clean_prediction != attacked_prediction:

                st.error(
                    "Prediction changed after attack."
                )

            else:

                st.success(
                    "Prediction remained unchanged."
                )

        else:

            st.warning(
                "Victim checkpoint unavailable. "
                "Prediction results cannot be displayed."
            )

        # ----------------------------------------------------
        # EXA-GUARD
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'EXA-Guard Detection'
            '</div>',
            unsafe_allow_html=True
        )

        if exa_status:

            attack_probability, localization_map = (
                get_exa_guard_prediction(
                    attacked_tensor
                )
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Attack Probability",
                    f"{attack_probability * 100:.2f}%"
                )

            with col2:

                if attack_probability >= 0.5:

                    st.error(
                        "Adversarial Attack Detected"
                    )

                else:

                    st.success(
                        "Clean Image Detected"
                    )

            # ------------------------------------------------
            # SPATIAL LOCALIZATION
            # ------------------------------------------------

            st.markdown(
                '<div class="section-title">'
                'Spatial Attack Localization'
                '</div>',
                unsafe_allow_html=True
            )

            localization_map = normalize_heatmap(
                localization_map
            )

            colored_heatmap, localization_overlay = (
                make_heatmap_overlay(
                    attacked_image,
                    localization_map,
                    alpha=0.45
                )
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                st.subheader(
                    "Attacked Image"
                )

                st.image(
                    attacked_image,
                    use_container_width=True
                )

            with col2:

                st.subheader(
                    "Colored Heatmap"
                )

                st.image(
                    colored_heatmap,
                    use_container_width=True
                )

            with col3:

                st.subheader(
                    "Localization Overlay"
                )

                st.image(
                    localization_overlay,
                    use_container_width=True
                )

            st.caption(
                "Blue: lower activation | Green/Yellow: "
                "intermediate activation | Red: higher activation"
            )

        else:

            st.error(
                "EXA-Guard checkpoint is unavailable. "
                "Spatial localization results cannot be "
                "considered trained model outputs."
            )

        # ----------------------------------------------------
        # GRAD-CAM
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Grad-CAM Explainability'
            '</div>',
            unsafe_allow_html=True
        )

        if victim_status:

            cam, cam_error = generate_gradcam(
                attacked_tensor
            )

            if cam is not None:

                colored_cam, cam_overlay = (
                    make_heatmap_overlay(
                        attacked_image,
                        cam,
                        alpha=0.45
                    )
                )

                col1, col2, col3 = st.columns(3)

                with col1:

                    st.subheader(
                        "Attacked Image"
                    )

                    st.image(
                        attacked_image,
                        use_container_width=True
                    )

                with col2:

                    st.subheader(
                        "Grad-CAM Heatmap"
                    )

                    st.image(
                        colored_cam,
                        use_container_width=True
                    )

                with col3:

                    st.subheader(
                        "Grad-CAM Overlay"
                    )

                    st.image(
                        cam_overlay,
                        use_container_width=True
                    )

            else:

                st.error(
                    f"Grad-CAM could not be generated: "
                    f"{cam_error}"
                )

        else:

            st.warning(
                "Grad-CAM requires a loaded victim model."
            )

        # ----------------------------------------------------
        # PERTURBATION ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Attack Perturbation Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        difference_map = calculate_difference(
            tensor_to_image(
                image_tensor
            ),
            attacked_image
        )

        difference_heatmap = create_colored_heatmap(
            difference_map
        )

        difference_overlay = make_heatmap_overlay(
            tensor_to_image(
                image_tensor
            ),
            difference_map,
            alpha=0.5
        )[1]

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Perturbation Heatmap"
            )

            st.image(
                difference_heatmap,
                use_container_width=True
            )

        with col2:

            st.subheader(
                "Perturbation Overlay"
            )

            st.image(
                difference_overlay,
                use_container_width=True
            )

        # ----------------------------------------------------
        # ATTACK CONFIGURATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Attack Configuration'
            '</div>',
            unsafe_allow_html=True
        )

        if attack_name == "FGSM":

            st.write(
                f"Epsilon: {FGSM_EPSILON * 255:.0f}/255"
            )

        elif attack_name == "PGD":

            st.write(
                f"Epsilon: {PGD_EPSILON * 255:.0f}/255"
            )

            st.write(
                f"Alpha: {PGD_ALPHA * 255:.0f}/255"
            )

            st.write(
                f"Iterations: {PGD_STEPS}"
            )

        elif attack_name == "Patch Attack":

            st.write(
                f"Patch Size: {PATCH_SIZE} × {PATCH_SIZE}"
            )

            st.write(
                "Attack Type: Fixed simulated patch"
            )

        else:

            st.write(
                "No attack applied."
            )

    except Exception as e:

        st.error(
            f"Application Error: {str(e)}"
        )

        st.code(
            traceback.format_exc()
        )


# ============================================================
# 25. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center;color:#64748b">

    EXA-Guard | Research Prototype

    Explainable Adversarial Analytics for Facial Biometric Systems

    </div>
    """,
    unsafe_allow_html=True
)
