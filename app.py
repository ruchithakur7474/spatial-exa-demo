# ============================================================
# EXA-GUARD: Explainable Adversarial Analytics
# Streamlit Application
# ============================================================

import os
import io
import traceback
import warnings

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
    layout="wide",
    initial_sidebar_state="expanded"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGE_SIZE = 112

NUM_CLASSES = 200

FGSM_EPSILON = 8 / 255

PGD_EPSILON = 8 / 255
PGD_ALPHA = 2 / 255
PGD_STEPS = 10

PATCH_SIZE = 20

VICTIM_CHECKPOINT = "victim_model_fp16.pth"

EXA_CHECKPOINT = "exa_guard_model.pth"


# ============================================================
# 2. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 35px;
        font-weight: 800;
        color: #2563eb;
        text-align: center;
    }

    .subtitle {
        text-align: center;
        font-size: 16px;
        color: #64748b;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 700;
        margin-top: 20px;
        margin-bottom: 10px;
    }

    .info-box {
        background-color: #f1f5f9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #2563eb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. IMAGE TRANSFORMATION
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


def tensor_to_numpy(tensor):

    tensor = tensor.detach().cpu()

    tensor = tensor.squeeze(0)

    tensor = tensor * 0.5 + 0.5

    tensor = torch.clamp(tensor, 0, 1)

    return tensor.permute(1, 2, 0).numpy()


# ============================================================
# 4. VICTIM MODEL
# ============================================================

class VictimModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.model = resnet18(weights=None)

        self.model.fc = nn.Linear(
            self.model.fc.in_features,
            num_classes
        )

    def forward(self, x):

        return self.model(x)


# ============================================================
# 5. EXA-GUARD MODEL
# ============================================================

class EXAGuardModel(nn.Module):

    def __init__(self):

        super().__init__()

        # Feature extraction
        self.features = nn.Sequential(

            nn.Conv2d(
                3, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Conv2d(
                64, 128,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 256,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True)
        )

        # Attack classification
        self.attack_classifier = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),

            nn.Flatten(),

            nn.Linear(256, 128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(128, 2)
        )

        # Spatial localization
        self.localization_head = nn.Sequential(

            nn.Conv2d(
                256, 128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                128, 64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64, 1,
                kernel_size=1
            ),

            nn.Sigmoid()
        )

    def forward(self, x):

        features = self.features(x)

        attack_logits = self.attack_classifier(features)

        localization_map = self.localization_head(features)

        localization_map = F.interpolate(
            localization_map,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        return attack_logits, localization_map


# ============================================================
# 6. CHECKPOINT LOADING
# ============================================================

def load_checkpoint(model, checkpoint_path):

    if not os.path.exists(checkpoint_path):

        return False, f"Checkpoint not found: {checkpoint_path}"

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False
        )

        # Support different checkpoint formats

        if isinstance(checkpoint, dict):

            if "state_dict" in checkpoint:

                state_dict = checkpoint["state_dict"]

            elif "model_state_dict" in checkpoint:

                state_dict = checkpoint["model_state_dict"]

            elif "model" in checkpoint:

                state_dict = checkpoint["model"]

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        # Remove common prefixes

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):

                new_key = new_key.replace(
                    "module.",
                    "",
                    1
                )

            cleaned_state_dict[new_key] = value

        result = model.load_state_dict(
            cleaned_state_dict,
            strict=False
        )

        model.to(DEVICE)

        model.eval()

        return True, (
            f"Checkpoint loaded successfully.\n"
            f"Missing keys: {len(result.missing_keys)}\n"
            f"Unexpected keys: {len(result.unexpected_keys)}"
        )

    except Exception as e:

        return False, str(e)


# ============================================================
# 7. MODEL INITIALIZATION
# ============================================================

@st.cache_resource
def initialize_models():

    victim_model = VictimModel(
        num_classes=NUM_CLASSES
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
# 8. FGSM ATTACK
# ============================================================

def generate_fgsm_attack(image_tensor):

    image_tensor = image_tensor.clone().detach()

    image_tensor.requires_grad = True

    output = victim_model(image_tensor)

    predicted_class = output.argmax(dim=1)

    loss = F.cross_entropy(
        output,
        predicted_class
    )

    victim_model.zero_grad()

    loss.backward()

    gradient = image_tensor.grad.data

    perturbed_image = (
        image_tensor +
        FGSM_EPSILON * gradient.sign()
    )

    perturbed_image = torch.clamp(
        perturbed_image,
        -1,
        1
    )

    return perturbed_image.detach()


# ============================================================
# 9. PGD ATTACK
# ============================================================

def generate_pgd_attack(image_tensor):

    original_image = image_tensor.clone().detach()

    perturbed_image = original_image.clone().detach()

    perturbed_image = perturbed_image + torch.empty_like(
        perturbed_image
    ).uniform_(
        -PGD_EPSILON,
        PGD_EPSILON
    )

    perturbed_image = torch.clamp(
        perturbed_image,
        -1,
        1
    )

    predicted_class = victim_model(
        original_image
    ).argmax(dim=1)

    for step in range(PGD_STEPS):

        perturbed_image.requires_grad = True

        output = victim_model(
            perturbed_image
        )

        loss = F.cross_entropy(
            output,
            predicted_class
        )

        victim_model.zero_grad()

        loss.backward()

        gradient = perturbed_image.grad.data

        perturbed_image = (
            perturbed_image.detach() +
            PGD_ALPHA * gradient.sign()
        )

        perturbation = (
            perturbed_image -
            original_image
        )

        perturbation = torch.clamp(
            perturbation,
            -PGD_EPSILON,
            PGD_EPSILON
        )

        perturbed_image = (
            original_image +
            perturbation
        )

        perturbed_image = torch.clamp(
            perturbed_image,
            -1,
            1
        )

    return perturbed_image.detach()


# ============================================================
# 10. SIMULATED PATCH ATTACK
# ============================================================

def generate_patch_attack(image_tensor):

    image_tensor = image_tensor.clone().detach()

    _, _, height, width = image_tensor.shape

    patch_size = min(
        PATCH_SIZE,
        height,
        width
    )

    start_x = (width - patch_size) // 2

    start_y = (height - patch_size) // 2

    # Simulated fixed adversarial patch

    patch = torch.zeros(
        1,
        3,
        patch_size,
        patch_size,
        device=DEVICE
    )

    patch[:, 0, :, :] = 1.0

    patch[:, 1, :, :] = -1.0

    patch[:, 2, :, :] = 1.0

    image_tensor[
        :,
        :,
        start_y:start_y + patch_size,
        start_x:start_x + patch_size
    ] = patch

    return image_tensor.detach()


# ============================================================
# 11. ATTACK SELECTION
# ============================================================

def apply_attack(image_tensor, attack_name):

    if attack_name == "Clean":

        return image_tensor.clone().detach()

    elif attack_name == "FGSM":

        return generate_fgsm_attack(
            image_tensor
        )

    elif attack_name == "PGD":

        return generate_pgd_attack(
            image_tensor
        )

    elif attack_name == "Patch Attack":

        return generate_patch_attack(
            image_tensor
        )

    else:

        return image_tensor.clone().detach()


# ============================================================
# 12. VICTIM MODEL PREDICTION
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
        confidence.item(),
        probabilities
    )


# ============================================================
# 13. EXA-GUARD PREDICTION
# ============================================================

def get_exa_guard_prediction(image_tensor):

    with torch.no_grad():

        attack_logits, localization_map = (
            exa_guard_model(image_tensor)
        )

        probabilities = F.softmax(
            attack_logits,
            dim=1
        )

        attack_probability = probabilities[
            0, 1
        ].item()

        predicted_attack = probabilities.argmax(
            dim=1
        ).item()

        localization_map = localization_map[
            0, 0
        ]

    return (
        attack_probability,
        predicted_attack,
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
        heatmap,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    minimum = heatmap.min()

    maximum = heatmap.max()

    if maximum - minimum < 1e-8:

        return np.zeros_like(
            heatmap,
            dtype=np.float32
        )

    heatmap = (
        heatmap - minimum
    ) / (
        maximum - minimum
    )

    return heatmap


# ============================================================
# 15. COLORED HEATMAP
# ============================================================

def create_colored_heatmap(heatmap):

    heatmap = normalize_heatmap(
        heatmap
    )

    # JET-LIKE COLOR MAPPING
    # Low activation = blue
    # Medium activation = green/yellow
    # High activation = red

    red = np.clip(
        1.5 - np.abs(
            4 * heatmap - 3
        ),
        0,
        1
    )

    green = np.clip(
        1.5 - np.abs(
            4 * heatmap - 2
        ),
        0,
        1
    )

    blue = np.clip(
        1.5 - np.abs(
            4 * heatmap - 1
        ),
        0,
        1
    )

    colored_heatmap = np.stack(
        [
            red,
            green,
            blue
        ],
        axis=-1
    )

    colored_heatmap = (
        colored_heatmap * 255
    ).astype(np.uint8)

    return Image.fromarray(
        colored_heatmap
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
    ).astype(np.float32)

    heatmap = normalize_heatmap(
        heatmap
    )

    colored_heatmap = create_colored_heatmap(
        heatmap
    )

    colored_heatmap = colored_heatmap.resize(
        original_image.size
    )

    heatmap_array = np.asarray(
        colored_heatmap
    ).astype(np.float32)

    overlay = (
        (1 - alpha) * original_array +
        alpha * heatmap_array
    )

    overlay = np.clip(
        overlay,
        0,
        255
    ).astype(np.uint8)

    overlay_image = Image.fromarray(
        overlay
    )

    return (
        colored_heatmap,
        overlay_image
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
            self.target_layer.register_forward_hook(
                self.forward_hook
            )
        )

    def forward_hook(
        self,
        module,
        input,
        output
    ):

        # Handle tuple/list outputs safely

        if isinstance(
            output,
            (tuple, list)
        ):

            output = output[0]

        self.activations = output

        # Capture gradients directly from tensor

        if isinstance(
            output,
            torch.Tensor
        ):

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

        # Handle model output tuple/list

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
                "Model output is not a Tensor."
            )

        if output.ndim != 2:

            raise ValueError(
                "Grad-CAM requires classification output "
                "with shape [batch, classes]."
            )

        if target_class is None:

            target_class = output.argmax(
                dim=1
            ).item()

        score = output[
            0,
            target_class
        ]

        score.backward(
            retain_graph=False
        )

        if self.activations is None:

            raise RuntimeError(
                "Grad-CAM activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Grad-CAM gradients were not captured."
            )

        activations = self.activations

        gradients = self.gradients

        # Both must be 4D feature maps

        if activations.ndim != 4:

            raise ValueError(
                f"Activation shape incorrect: "
                f"{activations.shape}"
            )

        if gradients.ndim != 4:

            raise ValueError(
                f"Gradient shape incorrect: "
                f"{gradients.shape}"
            )

        # Channel-wise gradient importance

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


# ============================================================
# 18. GRAD-CAM EXECUTION
# ============================================================

def generate_gradcam(
    image_tensor,
    target_class=None
):

    try:

        target_layer = (
            victim_model.model.layer4[-1]
        )

        gradcam = GradCAM(
            victim_model,
            target_layer
        )

        cam = gradcam.generate(
            image_tensor,
            target_class
        )

        gradcam.remove_hooks()

        return cam, None

    except Exception as e:

        return None, str(e)


# ============================================================
# 19. IMAGE DIFFERENCE
# ============================================================

def calculate_difference(
    clean_image,
    attacked_image
):

    clean_array = np.asarray(
        clean_image
    ).astype(np.float32)

    attacked_array = np.asarray(
        attacked_image
    ).astype(np.float32)

    difference = np.abs(
        attacked_array -
        clean_array
    )

    difference = difference.mean(
        axis=2
    )

    difference = normalize_heatmap(
        difference
    )

    return difference


# ============================================================
# 20. APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">EXA-GUARD</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
    Explainable Adversarial Analytics for
    Facial Biometric Security
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 21. SIDEBAR
# ============================================================

st.sidebar.title(
    "EXA-Guard Settings"
)

st.sidebar.write(
    f"Device: **{DEVICE}**"
)

st.sidebar.write(
    f"Image Size: **{IMAGE_SIZE} × {IMAGE_SIZE}**"
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

    st.sidebar.warning(
        "Victim Model Not Loaded"
    )

    st.sidebar.caption(
        victim_message
    )

if exa_status:

    st.sidebar.success(
        "EXA-Guard Model Loaded"
    )

else:

    st.sidebar.warning(
        "EXA-Guard Model Not Loaded"
    )

    st.sidebar.caption(
        exa_message
    )


# ============================================================
# 22. MAIN INFORMATION
# ============================================================

st.markdown(
    """
    <div class="info-box">

    <b>Research Demonstration:</b>

    This application demonstrates adversarial attack generation,
    attack detection, spatial localization and explainability.

    Model outputs are meaningful only when trained checkpoints
    are available and correctly matched to the architecture.

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 23. IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-title">Upload Facial Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# 24. ATTACK SELECTION
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
# 25. PROCESS IMAGE
# ============================================================

if uploaded_file is not None:

    try:

        original_image = Image.open(
            uploaded_file
        ).convert("RGB")

        st.markdown(
            '<div class="section-title">'
            'Original Image'
            '</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            width=350
        )

        image_tensor = image_to_tensor(
            original_image
        )

        # Generate attacked image

        attacked_tensor = apply_attack(
            image_tensor,
            attack_name
        )

        attacked_image = tensor_to_image(
            attacked_tensor
        )

        # ====================================================
        # 26. CLEAN VS ATTACKED COMPARISON
        # ====================================================

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
                f"{attack_name} Image"
            )

            st.image(
                attacked_image,
                use_container_width=True
            )

        # ====================================================
        # 27. VICTIM MODEL
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Victim Model Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        if victim_status:

            clean_prediction, clean_confidence, _ = (
                get_victim_prediction(
                    image_tensor
                )
            )

            attacked_prediction, attacked_confidence, _ = (
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
                    "Victim model prediction changed "
                    "after attack."
                )

            else:

                st.info(
                    "Victim model prediction remained unchanged."
                )

        else:

            st.warning(
                "Victim checkpoint is unavailable. "
                "Predictions are not displayed because "
                "the model may be untrained."
            )

        # ====================================================
        # 28. EXA-GUARD ANALYSIS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'EXA-Guard Detection'
            '</div>',
            unsafe_allow_html=True
        )

        if exa_status:

            attack_probability, predicted_attack, localization_map = (
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

                if predicted_attack == 1:

                    st.error(
                        "Adversarial Attack Detected"
                    )

                else:

                    st.success(
                        "Clean Image Detected"
                    )

            # =================================================
            # 29. SPATIAL ATTACK LOCALIZATION
            # =================================================

            st.markdown(
                '<div class="section-title">'
                'Spatial Attack Localization'
                '</div>',
                unsafe_allow_html=True
            )

            localization_map = (
                localization_map.detach()
                .cpu()
                .numpy()
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
                    "Colored Localization Heatmap"
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
                "Blue indicates lower activation, "
                "green/yellow indicates intermediate activation, "
                "and red indicates higher activation."
            )

        else:

            st.warning(
                "EXA-Guard checkpoint is unavailable. "
                "Spatial localization results cannot "
                "be considered trained model outputs."
            )

        # ====================================================
        # 30. GRAD-CAM
        # ====================================================

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
                "Grad-CAM cannot be generated without "
                "a valid victim model checkpoint."
            )

        # ====================================================
        # 31. IMAGE DIFFERENCE
        # ====================================================

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

        difference_heatmap = (
            create_colored_heatmap(
                difference_map
            )
        )

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

            difference_overlay = (
                make_heatmap_overlay(
                    tensor_to_image(
                        image_tensor
                    ),
                    difference_map,
                    alpha=0.5
                )[1]
            )

            st.subheader(
                "Perturbation Overlay"
            )

            st.image(
                difference_overlay,
                use_container_width=True
            )

        # ====================================================
        # 32. TECHNICAL INFORMATION
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Attack Configuration'
            '</div>',
            unsafe_allow_html=True
        )

        if attack_name == "FGSM":

            st.write(
                f"FGSM Epsilon: {FGSM_EPSILON:.5f}"
            )

            st.write(
                f"FGSM Epsilon: {FGSM_EPSILON * 255:.0f}/255"
            )

        elif attack_name == "PGD":

            st.write(
                f"PGD Epsilon: {PGD_EPSILON * 255:.0f}/255"
            )

            st.write(
                f"PGD Alpha: {PGD_ALPHA * 255:.0f}/255"
            )

            st.write(
                f"PGD Steps: {PGD_STEPS}"
            )

        elif attack_name == "Patch Attack":

            st.write(
                f"Patch Size: {PATCH_SIZE} × {PATCH_SIZE}"
            )

            st.write(
                "Patch Type: Fixed simulated patch"
            )

        else:

            st.write(
                "No adversarial perturbation applied."
            )

    except Exception as e:

        st.error(
            f"Application Error: {str(e)}"
        )

        st.code(
            traceback.format_exc()
        )


# ============================================================
# 33. FOOTER
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
