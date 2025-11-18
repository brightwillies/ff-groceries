"""
Streamlit YOLOv7 Detector (FINAL WORKING VERSION)
"""
import sys
sys.path.append(r"C:\Users\harshika\Downloads\yolov7-main\yolov7-main")  # <-- IMPORTANT

import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image
import os


# ============================
# CONFIG
# ============================
MODEL_PATH = r"C:\Users\harshika\Downloads\Streamlit\yolov7_cheerios_soup_candle_best.pt"
IMG_SIZE = 640
CONF_THRESHOLD = 0.25

CLASSES = ["cheerios", "soup", "candle"]
COLORS = [(255,0,0),(0,255,0),(0,0,255)]


# ============================
# LOAD YOLOv7 CHECKPOINT
# ============================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model file not found.")
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Must use weights_only=False for YOLOv7 checkpoints
    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=False
    )

    # YOLOv7 saves model as checkpoint["model"]
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model = checkpoint["model"]
    else:
        model = checkpoint

    model.to(device).eval()
    return model, device


# ============================
# PREPROCESS
# ============================
def preprocess(image):
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img_resized = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE))

    img_norm = img_resized / 255.0
    img_norm = img_norm.transpose(2, 0, 1)
    tensor = torch.from_numpy(img_norm).float().unsqueeze(0)

    return tensor, img_bgr


# ============================
# POSTPROCESS
# ============================
def yolo_postprocess(raw_output):
    if isinstance(raw_output, torch.Tensor):
        raw_output = raw_output.cpu().numpy()

    detections = []
    for det in raw_output:
        x1, y1, x2, y2, conf, cls = det
        if conf >= CONF_THRESHOLD:
            detections.append([x1, y1, x2, y2, float(conf), int(cls)])

    return detections


# ============================
# DRAW BOXES
# ============================
def draw_boxes(img, detections):
    img = img.copy()
    for (x1, y1, x2, y2, conf, cls_id) in detections:
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        color = COLORS[cls_id % len(COLORS)]
        label = f"{CLASSES[cls_id]} {conf:.2f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            img, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
        )

    return img


# ============================
# STREAMLIT UI
# ============================
st.set_page_config(page_title="YOLOv7 Detector", layout="centered")
st.title("YOLOv7 Grocery Item Detector")
st.write("Detects **cheerios**, **soup**, **candle**")

with st.spinner("Loading YOLOv7 model..."):
    model, device = load_model()

uploaded = st.file_uploader("Upload an Image", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded Image")

    tensor, original_img = preprocess(image)
    tensor = tensor.to(device)

    with torch.no_grad():
        raw_output = model(tensor)[0]

    detections = yolo_postprocess(raw_output)

    result_img = draw_boxes(original_img, detections)
    result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))

    st.image(result_pil, caption="Detection Result")

    st.success(f"Detected {len(detections)} objects.")

else:
    st.info("Upload an image to start detection.")
