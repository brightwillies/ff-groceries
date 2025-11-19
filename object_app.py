"""
Streamlit YOLOv7 Custom Detector – DEPLOYABLE VERSION (2025)
"""
import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os
import sys

# ============================
# CONFIG
# ============================
MODEL_PATH = "yolov7_cheerios_soup_candle_best.pt"
CFG_PATH = "yolov7-main/cfg/training/yolov7.yaml"   # change if you used a custom yaml
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

CLASSES = ["cheerios", "soup", "candle"]
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


# ============================
# LOAD MODEL (SAFE WAY – NO PICKLE DANGER)
# ============================
@st.cache_resource(show_spinner="Loading YOLOv7 model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Model not found: {MODEL_PATH}")
        st.stop()

    if not os.path.exists(CFG_PATH):
        st.error(f"YOLOv7 config not found: {CFG_PATH}\nMake sure the full yolov7-main folder is uploaded.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Add yolov7-main to path
    sys.path.insert(0, "yolov7-main")

    try:
        from models.yolo import Model
    except ImportError as e:
        st.error("Failed to import YOLOv7 classes. Check that yolov7-main folder is complete.")
        raise e

    # Re-create exact model architecture
    model = Model(cfg=CFG_PATH, nc=len(CLASSES)).to(device)

    # Load weights safely (only state_dict)
    ckpt = torch.load(MODEL_PATH, map_location=device)
    state_dict = ckpt['model'].float().state_dict() if 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)

    model.eval()
    return model, device


# ============================
# PREPROCESS
# ============================
def preprocess(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_norm = img_norm.transpose(2, 0, 1)[np.newaxis, ...]  # (1,3,640,640)
    tensor = torch.from_numpy(img_norm).to(device)
    return tensor, img_bgr


# ============================
# POSTPROCESS (YOLOv7 raw output → boxes)
# ============================
def postprocess(pred, orig_shape):
    detections = []
    pred = pred[0].cpu().numpy()  # (num_dets, 6) → x1,y1,x2,y2,conf,cls

    h, w = orig_shape[:2]
    scale_w = w / IMG_SIZE
    scale_h = h / IMG_SIZE

    for det in pred:
        x1, y1, x2, y2, conf, cls_id = det
        if conf < CONF_THRESHOLD:
            continue
        # Scale back to original image size
        x1 = int(x1 * scale_w)
        y1 = int(y1 * scale_h)
        x2 = int(x2 * scale_w)
        y2 = int(y2 * scale_h)
        detections.append([x1, y1, x2, y2, float(conf), int(cls_id)])
    return detections


# ============================
# DRAW BOXES
# ============================
def draw_boxes(img, detections):
    img = img.copy()
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASSES[cls_id]} {conf:.2f}"
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return img


# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="YOLOv7 Grocery Detector", layout="centered")
st.title("YOLOv7 Custom Detector")
st.write("Detects **cheerios**, **soup**, **candle**")

model, device = load_model()

uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tensor, orig_bgr = preprocess(image)

    with torch.no_grad():
        raw_pred = model(tensor)[0]  # YOLOv7 returns (pred, train_out)

    detections = postprocess(raw_pred, image.size[::-1])  # (w,h)

    result_bgr = draw_boxes(orig_bgr, detections)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    st.image(result_rgb, caption=f"Found {len(detections)} object(s)", use_column_width=True)

    if detections:
        st.success(f"Detected: {', '.join([CLASSES[d[5]] for d in detections])}")
    else:
        st.info("No objects detected above confidence threshold.")

else:
    st.info("Please upload an image to start detection.")