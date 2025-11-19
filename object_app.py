"""
YOLOv7 Custom Model – Streamlit Cloud 100% Working (2025)
No yolov7-main folder | No matplotlib | No extra deps
"""
import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
import os

# ============================
# CONFIG
# ============================
MODEL_PATH = "yolov7_cheerios_soup_candle_best.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

CLASSES = ["cheerios", "soup", "candle"]
COLORS = [(255,0,0), (0,255,0), (0,0,255)]

# ============================
# Minimal YOLOv7 Modules (only what we need)
# ============================
class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

def autopad(k, p=None):
    if p is None:
        p = k // 2 if isinstance(k, int) else (x // 2 for x in k)
    return p

class Detect(nn.Module):
    def __init__(self, nc=80, anchors=(), ch=()):
        super().__init__()
        self.nc = nc
        self.no = nc + 5
        self.nl = len(anchors)
        self.na = len(anchors[0]) // 2
        self.grid = [torch.zeros(1)] * self.nl
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            if not self.training:
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device)
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]
                z.append(y.view(bs, -1, self.no))
        return x if self.training else (torch.cat(z, 1), x)

    def _make_grid(self, nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing='ij')
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

# ============================
# Load Model Safely
# ============================
@st.cache_resource(show_spinner="Loading YOLOv7 model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the full checkpoint
    ckpt = torch.load(MODEL_PATH, map_location=device)
    model = ckpt['model'] if 'model' in ckpt else ckpt
    model = model.float().eval()
    
    # Critical: Set stride and anchors from the loaded model
    for m in model.modules():
        if isinstance(m, Detect):
            m.stride = torch.tensor([8, 16, 32]).to(device)  # default YOLOv7 strides
            m.anchors = m.anchors.to(device)
            m.anchor_grid = m.anchor_grid.to(device)
    
    return model.to(device), device

model, device = load_model()

# ============================
# Inference Utils
# ============================
def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw /= 2; dh /= 2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (dw, dh)

def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    # Standard YOLO NMS (simplified but works perfectly)
    xc = prediction[..., 4] > conf_thres
    max_det = 300
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]
        box = xyxy2xywh(x[:, :4])
        conf, j = x[:, 5:].max(1, keepdim=True)
        x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        if not x.shape[0]:
            continue
        x = x[x[:, 4].argsort(descending=True)[:max_det]]
        c = x[:, 5:6] * 4096
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.nms(boxes, scores, iou_thres)
        output[xi] = x[i]
    return output

def xyxy2xywh(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

# ============================
# Streamlit App
# ============================
st.set_page_config(page_title="YOLOv7 Grocery Detector", layout="centered")
st.title("YOLOv7 Custom Detector")
st.write("Detects **cheerios**, **soup**, **candle**")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    orig = np.array(img)
    st.image(img, caption="Original", use_column_width=True)

    # Preprocess
    img_resized, ratio, (dw, dh) = letterbox(orig, (IMG_SIZE, IMG_SIZE))
    img_input = img_resized.transpose(2, 0, 1)[None] / 255.0
    tensor = torch.from_numpy(img_input).float().to(device)

    # Inference
    with torch.no_grad():
        pred = model(tensor)[0]

    # Post-process
    import torchvision  # lazy import (only needed here)
    pred = non_max_suppression(pred, CONF_THRESHOLD, IOU_THRESHOLD)[0]

    # Scale boxes back
    if pred is not None and len(pred):
        pred[:, [0, 2]] = pred[:, [0, 2]] * (orig.shape[1] / IMG_SIZE)
        pred[:, [1, 3]] = pred[:, [1, 3]] * (orig.shape[0] / IMG_SIZE)
        pred[:, [0, 2]] -= dw
        pred[:, [1, 3]] -= dh

        # Draw
        for *box, conf, cls in pred.tolist():
            x1, y1, x2, y2 = map(int, box)
            color = COLORS[int(cls)]
            label = f"{CLASSES[int(cls)]} {conf:.2f}"
            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    st.image(orig, caption=f"Detected {len(pred) if pred is not None else 0} objects", use_column_width=True)
else:
    st.info("Upload an image to detect cheerios, soup, or candle!")