# ============================================================
# EXA-GUARD
# Explainable Adversarial Analytics for Facial Biometrics
# ============================================================

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
    page_icon="👤",
    layout="wide"
)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BASE_DIR = Path(__file__).resolve().parent

IMAGE_SIZE = 112
NUM_CLASSES = 200

# Epsilon and alpha are specified in original [0,1] pixel units.
# Since input is normalized using std=0.5, divide by 0.5.
FGSM_EPSILON = (8 / 255) / 0.5

PGD_EPSILON = (8 / 255) / 0.5
PGD_ALPHA = (2 / 255) / 0.5
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
# 3. IMAGE TRANSFORM
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

    array = tensor.permute(1, 2, 0).numpy()
    array = (array * 255).round().astype(np.uint8)

    return Image.fromarray(array)


# ============================================================
# 4. VICTIM MODEL
# ============================================================

class VictimModel(nn.Module):

    def __init__(self, num_classes=NUM_CLASSES):

        super().__init__()

        self.backbone = resnet18(weights=None)

        self.backbone.fc = nn.Linear(
            self.backbone.fc.in_features,
            num_classes
        )

    def forward(self, x):

        return self.backbone(x)


# ============================================================
# 5. EXA-GUARD MODEL
# CHECKPOINT-MATCHED ARCHITECTURE
# ============================================================

class EXAGuardModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        self.detector = nn.Sequential(

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),

            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
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

            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

    def forward(self, x):

        features = self.features(x)

        attack_logits = self.detector(features)

        attack_probability = torch.sigmoid(
            attack_logits
        )

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

        return attack_probability, localization_map


# ============================================================
# 6. CHECKPOINT UTILITIES
# ============================================================

def extract_state_dict(checkpoint):

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()

    if isinstance(checkpoint, dict):

        for key in (
            "state_dict",
            "model_state_dict",
            "model"
        ):

            if key in checkpoint:

                value = checkpoint[key]

                if isinstance(value, nn.Module):
                    return value.state_dict()

                if isinstance(value, dict):
                    return value

        return checkpoint

    raise TypeError(
        "Unsupported checkpoint format."
    )


def clean_state_dict(state_dict):

    cleaned = {}

    for key, value in state_dict.items():

        new_key = key

        # Remove DataParallel prefix.
        if new_key.startswith("module."):
            new_key = new_key[len("module."):]

        # Remove wrapper prefix if present.
        if new_key.startswith("model."):
            new_key = new_key[len("model."):]

        cleaned[new_key] = value

    return cleaned


def prepare_victim_state_dict(state_dict):

    """
    Supports checkpoints saved with:
    backbone.*
    model.backbone.*
    bare ResNet18 keys such as conv1.weight
    """

    state_dict = clean_state_dict(state_dict)

    # Check whether keys are bare ResNet18 keys.
    if "conv1.weight" in state_dict:

        state_dict = {
            "backbone." + key: value
            for key, value in state_dict.items()
        }

    return state_dict


def safe_torch_load(path):

    try:

        return torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

    except TypeError:

        # Compatibility with older PyTorch versions.
        return torch.load(
            path,
            map_location="cpu"
        )


def load_model_checkpoint(model, checkpoint_path):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():

        return False, (
            f"Checkpoint file not found:\n"
            f"{checkpoint_path}"
        )

    try:

        checkpoint = safe_torch_load(
            checkpoint_path
        )

        state_dict = extract_state_dict(
            checkpoint
        )

        state_dict = clean_state_dict(
            state_dict
        )

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

    except Exception:

        return False, traceback.format_exc()


def initialize_victim_model():

    if not VICTIM_CHECKPOINT.exists():

        return (
            None,
            False,
            f"Victim checkpoint not found:\n{VICTIM_CHECKPOINT}",
            None
        )

    try:

        checkpoint = safe_torch_load(
            VICTIM_CHECKPOINT
        )

        state_dict = extract_state_dict(
            checkpoint
        )

        state_dict = prepare_victim_state_dict(
            state_dict
        )

        # Infer the classifier size from checkpoint weights.
        fc_weight = state_dict.get(
            "backbone.fc.weight"
        )

        if fc_weight is None:

            raise KeyError(
                "Could not find backbone.fc.weight "
                "in the Victim checkpoint."
            )

        if fc_weight.ndim != 2:

            raise ValueError(
                "Invalid Victim classifier weight dimensions."
            )

        checkpoint_num_classes = int(
            fc_weight.shape[0]
        )

        model = VictimModel(
            num_classes=checkpoint_num_classes
        )

        result = model.load_state_dict(
            state_dict,
            strict=True
        )

        model.to(DEVICE)
        model.eval()

        message = (
            "Victim checkpoint loaded successfully.\n"
            f"Classifier outputs: {checkpoint_num_classes}\n"
            f"Missing keys: {len(result.missing_keys)}\n"
            f"Unexpected keys: {len(result.unexpected_keys)}"
        )

        return (
            model,
            True,
            message,
            checkpoint_num_classes
        )

    except Exception:

        return (
            None,
            False,
            traceback.format_exc(),
            None
        )


# ============================================================
# 7. INITIALIZE MODELS
# ============================================================

@st.cache_resource
def initialize_models():

    victim_model, victim_status, victim_message, victim_classes = (
        initialize_victim_model()
    )

    exa_guard_model = EXAGuardModel()

    exa_status, exa_message = load_model_checkpoint(
        exa_guard_model,
        EXA_CHECKPOINT
    )

    if not exa_status:
        exa_guard_model = None

    return (
        victim_model,
        exa_guard_model,
        victim_status,
        victim_message,
        victim_classes,
        exa_status,
        exa_message
    )


(
    victim_model,
    exa_guard_model,
    victim_status,
    victim_message,
    victim_classes,
    exa_status,
    exa_message
) = initialize_models()


# ============================================================
# 8. FGSM ATTACK
# ============================================================

def generate_fgsm_attack(image_tensor):

    if not victim_status or victim_model is None:
        raise RuntimeError("Victim Model is not loaded.")

    original = image_tensor.clone().detach()
    original.requires_grad_(True)

    output = victim_model(original)

    predicted_class = output.argmax(dim=1)

    loss = F.cross_entropy(
        output,
        predicted_class
    )

    victim_model.zero_grad(set_to_none=True)

    loss.backward()

    gradient = original.grad

    if gradient is None:
        raise RuntimeError("FGSM gradient could not be calculated.")

    attacked = (
        original.detach() +
        FGSM_EPSILON * gradient.sign()
    )

    attacked = torch.clamp(
        attacked,
        -1,
        1
    )

    return attacked.detach()


# ============================================================
# 9. PGD ATTACK
# ============================================================

def generate_pgd_attack(image_tensor):

    if not victim_status or victim_model is None:
        raise RuntimeError("Victim Model is not loaded.")

    original = image_tensor.clone().detach()

    attacked = original.clone().detach()

    random_noise = torch.empty_like(
        attacked
    ).uniform_(
        -PGD_EPSILON,
        PGD_EPSILON
    )

    attacked = torch.clamp(
        attacked + random_noise,
        -1,
        1
    )

    with torch.no_grad():

        target_class = victim_model(
            original
        ).argmax(dim=1)

    for _ in range(PGD_STEPS):

        attacked = attacked.detach()
        attacked.requires_grad_(True)

        output = victim_model(attacked)

        loss = F.cross_entropy(
            output,
            target_class
        )

        victim_model.zero_grad(set_to_none=True)

        loss.backward()

        if attacked.grad is None:
            raise RuntimeError("PGD gradient could not be calculated.")

        gradient = attacked.grad.detach()

        attacked = (
            attacked.detach() +
            PGD_ALPHA * gradient.sign()
        )

        perturbation = attacked - original

        perturbation = torch.clamp(
            perturbation,
            -PGD_EPSILON,
            PGD_EPSILON
        )

        attacked = original + perturbation

        attacked = torch.clamp(
            attacked,
            -1,
            1
        )

    return attacked.detach()


# ============================================================
# 10. SIMULATED PATCH ATTACK
# ============================================================

def generate_patch_attack(image_tensor):

    attacked = image_tensor.clone().detach()

    _, _, height, width = attacked.shape

    patch_size = min(
        PATCH_SIZE,
        height,
        width
    )

    start_x = (width - patch_size) // 2
    start_y = (height - patch_size) // 2

    patch = torch.zeros(
        1,
        3,
        patch_size,
        patch_size,
        device=DEVICE
    )

    # Fixed magenta patch in normalized input space.
    patch[:, 0] = 1.0
    patch[:, 1] = -1.0
    patch[:, 2] = 1.0

    attacked[
        :,
        :,
        start_y:start_y + patch_size,
        start_x:start_x + patch_size
    ] = patch

    return attacked.detach()


# ============================================================
# 11. ATTACK SELECTION
# ============================================================

def apply_attack(image_tensor, attack_name):

    if attack_name == "Clean":

        return image_tensor.clone().detach()

    if attack_name == "FGSM":

        return generate_fgsm_attack(image_tensor)

    if attack_name == "PGD":

        return generate_pgd_attack(image_tensor)

    if attack_name == "Patch Attack":

        return generate_patch_attack(image_tensor)

    raise ValueError(
        f"Unsupported attack: {attack_name}"
    )


# ============================================================
# 12. VICTIM PREDICTION
# ============================================================

def get_victim_prediction(image_tensor):

    if not victim_status or victim_model is None:
        raise RuntimeError("Victim Model is not loaded.")

    with torch.no_grad():

        output = victim_model(image_tensor)

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

    if not exa_status or exa_guard_model is None:
        raise RuntimeError("EXA-Guard Model is not loaded.")

    with torch.no_grad():

        attack_probability, localization_map = (
            exa_guard_model(image_tensor)
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

    return (
        heatmap - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 15. COLORED HEATMAP
# ============================================================

def create_colored_heatmap(heatmap):

    heatmap = normalize_heatmap(heatmap)

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
        [red, green, blue],
        axis=-1
    )

    colored = (
        colored * 255
    ).astype(np.uint8)

    return Image.fromarray(colored)


# ============================================================
# 16. HEATMAP OVERLAY
# ============================================================

def make_heatmap_overlay(
    original_image,
    heatmap,
    alpha=0.45
):

    original_image = original_image.convert("RGB")

    original_array = np.asarray(
        original_image
    ).astype(np.float32)

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

    return (
        colored_heatmap,
        Image.fromarray(overlay)
    )


# ============================================================
# 17. GRAD-CAM
# ============================================================

class GradCAM:

    def __init__(self, model, target_layer):

        self.model = model
        self.target_layer = target_layer

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

    def forward_hook(self, module, inputs, output):

        if isinstance(output, (tuple, list)):
            output = output[0]

        self.activations = output

        if output.requires_grad:

            output.register_hook(
                self.gradient_hook
            )

    def gradient_hook(self, gradient):

        self.gradients = gradient

    def generate(self, input_tensor, target_class=None):

        self.model.zero_grad(set_to_none=True)

        self.activations = None
        self.gradients = None

        input_tensor = input_tensor.clone().detach()
        input_tensor.requires_grad_(True)

        output = self.model(input_tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        if not isinstance(output, torch.Tensor):
            raise TypeError("Model output must be a Tensor.")

        if output.ndim != 2:
            raise ValueError(
                f"Expected 2D output, received {output.shape}"
            )

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        score = output[0, target_class]

        score.backward()

        if self.activations is None:
            raise RuntimeError("Grad-CAM activation was not captured.")

        if self.gradients is None:
            raise RuntimeError("Grad-CAM gradient was not captured.")

        activations = self.activations
        gradients = self.gradients

        if activations.ndim != 4 or gradients.ndim != 4:
            raise ValueError("Invalid Grad-CAM tensor dimensions.")

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

        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False
        )

        cam = cam.squeeze().detach().cpu().numpy()

        return normalize_heatmap(cam)

    def remove_hooks(self):

        self.forward_handle.remove()


def generate_gradcam(image_tensor):

    if not victim_status or victim_model is None:
        return None, "Victim Model is not loaded."

    gradcam = None

    try:

        target_layer = victim_model.backbone.layer4[-1]

        gradcam = GradCAM(
            victim_model,
            target_layer
        )

        cam = gradcam.generate(
            image_tensor
        )

        return cam, None

    except Exception as error:

        return None, str(error)

    finally:

        if gradcam is not None:
            gradcam.remove_hooks()

        victim_model.zero_grad(set_to_none=True)


# ============================================================
# 18. PERTURBATION ANALYSIS
# ============================================================

def calculate_difference(clean_image, attacked_image):

    clean_array = np.asarray(
        clean_image
    ).astype(np.float32)

    attacked_array = np.asarray(
        attacked_image
    ).astype(np.float32)

    difference = np.abs(
        clean_array - attacked_array
    )

    difference = difference.mean(axis=2)

    return normalize_heatmap(difference)


# ============================================================
# 19. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">👤EXA-GUARD</div>',
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

st.sidebar.title("EXA-Guard Settings")

st.sidebar.write(f"Device: {DEVICE}")

st.sidebar.write(
    f"Image Size: {IMAGE_SIZE} × {IMAGE_SIZE}"
)

st.sidebar.divider()

st.sidebar.subheader("Model Status")

if victim_status:

    st.sidebar.success("Victim Model Loaded")

    st.sidebar.caption(
        f"Classifier outputs: {victim_classes}"
    )

else:

    st.sidebar.error("Victim Model Not Loaded")

    with st.sidebar.expander(
        "Victim Model Loading Details"
    ):

        st.code(
            victim_message,
            language="text"
        )

        st.write("Checkpoint path:")
        st.code(str(VICTIM_CHECKPOINT))

        st.write(
            "Checkpoint exists:",
            VICTIM_CHECKPOINT.exists()
        )


if exa_status:

    st.sidebar.success("EXA-Guard Model Loaded")

else:

    st.sidebar.error("EXA-Guard Model Not Loaded")

    with st.sidebar.expander(
        "EXA-Guard Loading Details"
    ):

        st.code(
            exa_message,
            language="text"
        )

        st.write("Checkpoint path:")
        st.code(str(EXA_CHECKPOINT))

        st.write(
            "Checkpoint exists:",
            EXA_CHECKPOINT.exists()
        )


if st.sidebar.button("Reload Models"):

    initialize_models.clear()

    st.rerun()


# ============================================================
# 21. RESEARCH NOTICE
# ============================================================

st.info(
    """
    EXA-Guard is a research prototype for adversarial attack
    generation, attack detection, spatial localization and
    explainability.

    Model outputs depend on successfully loaded trained checkpoints.
    A successful checkpoint load does not independently verify
    training quality, dataset provenance or preprocessing compatibility.
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
    type=["jpg", "jpeg", "png"]
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
        ).convert("RGB")

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

        # Generate selected attack.
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

            st.subheader("Clean Image")

            st.image(
                tensor_to_image(image_tensor),
                use_container_width=True
            )

        with col2:

            st.subheader(attack_name)

            st.image(
                attacked_image,
                use_container_width=True
            )

        # ----------------------------------------------------
        # VICTIM MODEL ANALYSIS
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Victim Model Analysis'
            '</div>',
            unsafe_allow_html=True
        )

        if victim_status:

            clean_prediction, clean_confidence = (
                get_victim_prediction(image_tensor)
            )

            attacked_prediction, attacked_confidence = (
                get_victim_prediction(attacked_tensor)
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

            st.caption(
                "Predictions are classifier output indices, "
                "not verified person identities."
            )

        else:

            st.warning(
                "Victim checkpoint unavailable. "
                "Prediction and gradient-based attack results "
                "cannot be displayed."
            )

        # ----------------------------------------------------
        # EXA-GUARD DETECTION
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

                    st.error("Adversarial Attack Detected")

                else:

                    st.success("Clean Image Detected")

            # -----------------------------------------------
            # SPATIAL LOCALIZATION
            # -----------------------------------------------

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

                st.subheader("Attacked Image")

                st.image(
                    attacked_image,
                    use_container_width=True
                )

            with col2:

                st.subheader("Colored Heatmap")

                st.image(
                    colored_heatmap,
                    use_container_width=True
                )

            with col3:

                st.subheader("Localization Overlay")

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
                "Spatial localization cannot be reported as "
                "trained model output."
            )

        # ----------------------------------------------------
        # GRAD-CAM EXPLAINABILITY
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

                    st.subheader("Attacked Image")

                    st.image(
                        attacked_image,
                        use_container_width=True
                    )

                with col2:

                    st.subheader("Grad-CAM Heatmap")

                    st.image(
                        colored_cam,
                        use_container_width=True
                    )

                with col3:

                    st.subheader("Grad-CAM Overlay")

                    st.image(
                        cam_overlay,
                        use_container_width=True
                    )

            else:

                st.error(
                    f"Grad-CAM could not be generated: {cam_error}"
                )

        else:

            st.warning(
                "Grad-CAM requires a loaded Victim Model."
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

        clean_display_image = tensor_to_image(
            image_tensor
        )

        difference_map = calculate_difference(
            clean_display_image,
            attacked_image
        )

        difference_heatmap = create_colored_heatmap(
            difference_map
        )

        _, difference_overlay = make_heatmap_overlay(
            clean_display_image,
            difference_map,
            alpha=0.5
        )

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Perturbation Heatmap")

            st.image(
                difference_heatmap,
                use_container_width=True
            )

        with col2:

            st.subheader("Perturbation Overlay")

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

            st.write("FGSM epsilon: 8/255 (original pixel scale)")

        elif attack_name == "PGD":

            st.write("PGD epsilon: 8/255 (original pixel scale)")
            st.write("PGD alpha: 2/255 (original pixel scale)")
            st.write(f"Iterations: {PGD_STEPS}")

        elif attack_name == "Patch Attack":

            st.write(
                f"Patch Size: {PATCH_SIZE} × {PATCH_SIZE}"
            )

            st.write(
                "Attack Type: Fixed simulated patch"
            )

        else:

            st.write("No attack applied.")

    except Exception as error:

        st.error(
            f"Application Error: {str(error)}"
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
