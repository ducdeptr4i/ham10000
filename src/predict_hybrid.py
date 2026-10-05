"""Inference helpers for the HAM10000 Streamlit demo."""

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet50, vit_b_16


BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
CNN_PATH = OUTPUT_DIR / "best_resnet50.pth"
VIT_PATH = OUTPUT_DIR / "best_vit_b16.pth"
HYBRID_PATH = OUTPUT_DIR / "best_hybrid_weighted.pth"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_FULL_NAMES = {
    "akiec": "Dày sừng ánh sáng / ung thư biểu mô nội biểu mô",
    "bcc": "Ung thư biểu mô tế bào đáy",
    "bkl": "Tổn thương dạng dày sừng lành tính",
    "df": "U xơ da",
    "mel": "U hắc tố",
    "nv": "Nốt ruồi sắc tố",
    "vasc": "Tổn thương mạch máu",
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class HybridClassifier(nn.Module):
    """Classifier architecture matching train_hybrid_weighted.py."""

    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(2816, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

    def forward(self, x):
        return self.classifier(x)


def load_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for checkpoint in (CNN_PATH, VIT_PATH, HYBRID_PATH):
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Không tìm thấy checkpoint: {checkpoint}")

    cnn = resnet50(weights=None)
    cnn.fc = nn.Linear(cnn.fc.in_features, len(CLASS_NAMES))
    cnn.load_state_dict(torch.load(CNN_PATH, map_location=device))
    cnn_classifier = cnn.fc.to(device).eval()
    cnn.fc = nn.Identity()
    cnn = cnn.to(device).eval()

    vit = vit_b_16(weights=None)
    vit.heads.head = nn.Linear(vit.heads.head.in_features, len(CLASS_NAMES))
    vit.load_state_dict(torch.load(VIT_PATH, map_location=device))
    vit_classifier = vit.heads.to(device).eval()
    vit.heads = nn.Identity()
    vit = vit.to(device).eval()

    hybrid = HybridClassifier()
    hybrid.load_state_dict(torch.load(HYBRID_PATH, map_location=device))
    hybrid = hybrid.to(device).eval()
    return cnn, vit, hybrid, device, cnn_classifier, vit_classifier


def inspect_image(image):
    """Report likely image-quality issues without altering or cropping it."""
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    issues = []
    suggestions = []

    if min(width, height) < 224:
        issues.append("Ảnh có cạnh dưới 224 px; chi tiết nhỏ có thể không rõ.")
    if max(width, height) / max(1, min(width, height)) > 2.5:
        issues.append("Ảnh có tỷ lệ khung hình rất dài; vùng tổn thương có thể nhỏ.")

    # Look only at narrow outer bands. A warning is advisory; dark lesions can
    # also trigger this heuristic, so the image is never cropped automatically.
    band_y = max(1, int(height * 0.05))
    band_x = max(1, int(width * 0.05))
    bands = {
        "mép trên": rgb[:band_y, :, :],
        "mép dưới": rgb[-band_y:, :, :],
        "mép trái": rgb[:, :band_x, :],
        "mép phải": rgb[:, -band_x:, :],
    }
    dark_edges = []
    blank_edges = []
    for name, band in bands.items():
        gray = cv2.cvtColor(band, cv2.COLOR_RGB2GRAY)
        if float(np.mean(gray < 18)) > 0.72:
            dark_edges.append(name)
        if float(np.std(gray)) < 3.0 and (
            float(np.mean(gray)) < 24 or float(np.mean(gray)) > 238
        ):
            blank_edges.append(name)

    if dark_edges:
        issues.append("Có vùng gần đen ở " + ", ".join(dark_edges) + ".")
        suggestions.append("Kiểm tra viền đen có chiếm diện tích đáng kể không.")
    if blank_edges:
        issues.append("Có viền gần như đồng màu ở " + ", ".join(blank_edges) + ".")
        suggestions.append("Xem lại ảnh gốc; hệ thống không tự cắt viền.")

    return {
        "width": width,
        "height": height,
        "issues": issues,
        "suggestions": suggestions,
        "has_warnings": bool(issues),
    }


def _make_overlay(image, heatmap):
    colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    colored = cv2.resize(colored, image.size, interpolation=cv2.INTER_LINEAR)
    original = np.asarray(image.convert("RGB"), dtype=np.float32)
    alpha = cv2.resize(heatmap, image.size, interpolation=cv2.INTER_LINEAR)
    alpha = (0.5 * alpha[..., None]).astype(np.float32)
    blended = original * (1.0 - alpha) + colored.astype(np.float32) * alpha
    return Image.fromarray(np.uint8(np.clip(blended, 0, 255)))


def _rank_logits(logits, top_k=2):
    probabilities = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probabilities, k=top_k)
    ranked = []
    for score, index in zip(values.detach().cpu().tolist(), indices.detach().cpu().tolist()):
        code = CLASS_NAMES[index]
        ranked.append({
            "class_code": code,
            "class_name": CLASS_FULL_NAMES[code],
            "score": float(score),
        })
    return probabilities, ranked


def _make_gradcam(activation, gradients, image):
    heatmap = torch.relu(
        (gradients.mean(dim=(2, 3), keepdim=True) * activation).sum(dim=1)
    )
    heatmap = F.interpolate(
        heatmap.unsqueeze(1), size=(224, 224), mode="bilinear", align_corners=False
    )[0, 0].detach().cpu().numpy()
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return _make_overlay(image, heatmap)


def predict_image(image, cnn, vit, hybrid, device, cnn_classifier, vit_classifier):
    """Compare standalone and hybrid rankings and create CNN-branch Grad-CAM maps."""
    image = image.convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)
    captured = {}

    def capture_layer(_module, _inputs, output):
        captured["activation"] = output

    hook = cnn.layer4.register_forward_hook(capture_layer)
    try:
        # Gradients are needed only for the visualization. Model weights stay fixed.
        with torch.enable_grad():
            cnn_feature = cnn(input_tensor)
            cnn_logits = cnn_classifier(cnn_feature)
            cnn_probabilities, cnn_ranked = _rank_logits(cnn_logits)
            with torch.no_grad():
                vit_feature = vit(input_tensor)
                vit_logits = vit_classifier(vit_feature)
                vit_probabilities, vit_ranked = _rank_logits(vit_logits)
            logits = hybrid(torch.cat((cnn_feature, vit_feature), dim=1))
            probabilities, hybrid_ranked = _rank_logits(logits)
            activation = captured.get("activation")
            if activation is None:
                raise RuntimeError("Không lấy được đặc trưng ResNet50 để tạo Grad-CAM.")
            cnn_gradients = torch.autograd.grad(
                cnn_logits[0, int(cnn_probabilities.argmax().item())],
                activation,
                retain_graph=True,
            )[0]
            hybrid_gradients = torch.autograd.grad(
                logits[0, int(probabilities.argmax().item())], activation
            )[0]
    finally:
        hook.remove()

    all_model_probabilities = {
        "ResNet50": cnn_probabilities,
        "ViT-B/16": vit_probabilities,
        "Hybrid": probabilities,
    }
    all_model_rankings = {
        "ResNet50": cnn_ranked,
        "ViT-B/16": vit_ranked,
        "Hybrid": hybrid_ranked,
    }

    return {
        "class_code": hybrid_ranked[0]["class_code"],
        "class_name": hybrid_ranked[0]["class_name"],
        "score": hybrid_ranked[0]["score"],
        "ranked_classes": hybrid_ranked,
        "model_rankings": all_model_rankings,
        "all_scores": {
            model_name: {
                CLASS_NAMES[i]: float(model_probabilities[i].detach().item())
                for i in range(len(CLASS_NAMES))
            }
            for model_name, model_probabilities in all_model_probabilities.items()
        },
        "gradcam_resnet": _make_gradcam(activation, cnn_gradients, image),
        "gradcam_hybrid": _make_gradcam(activation, hybrid_gradients, image),
    }
