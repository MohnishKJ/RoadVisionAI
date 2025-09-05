import os
import tempfile
import cv2
import torch
import numpy as np
import streamlit as st
import pandas as pd
from ultralytics import YOLO
from PIL import Image

# ----------------- Load MiDaS -----------------
midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large")
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device).eval()

# ----------------- Load YOLO -----------------
yolo_model = YOLO("best.pt")

# ----------------- Helper Functions -----------------
def run_yolo(image):
    return yolo_model(image)

def run_midas(image):
    if isinstance(image, Image.Image):
        img_np = np.array(image)
    else:
        img_np = image
    img_np = img_np.astype(np.float32) / 255.0
    input_batch = transform(img_np).to(device)
    with torch.no_grad():
        prediction = midas(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_np.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    return prediction.cpu().numpy()

def colorize_depth(depth_map):
    depth_norm = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_uint8 = depth_norm.astype(np.uint8)
    return cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

def calculate_pothole_metrics(bbox, depth_map, calib_factor=1.0):
    x1, y1, x2, y2 = map(int, bbox)
    pothole_region = depth_map[y1:y2, x1:x2]
    if pothole_region.size == 0:
        return None, None
    avg_depth = float(np.mean(pothole_region) * calib_factor)
    diameter = float(max(x2 - x1, y2 - y1) * calib_factor)
    return avg_depth, diameter

def process_results(image, results, depth_map, calib_factor=1.0, unit="px"):
    image_draw = np.array(image.copy())
    pothole_data = []
    pothole_id = 1
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            depth, diameter = calculate_pothole_metrics([x1, y1, x2, y2], depth_map, calib_factor)
            if depth is None:
                continue

            cv2.rectangle(image_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"Pothole {pothole_id} | Depth: {depth:.1f}{unit} | Dia: {diameter:.1f}{unit}"
            cv2.putText(image_draw, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            pothole_data.append({
                "Pothole ID": pothole_id,
                "Confidence (%)": f"{conf*100:.1f}",
                f"Depth ({unit})": f"{depth:.1f}",
                f"Diameter ({unit})": f"{diameter:.1f}"
            })
            pothole_id += 1
    return image_draw, pothole_data

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="Road Vision AI", layout="wide")

st.title("Road Vision AI")
st.markdown(
    "Upload an image or video to detect potholes and estimate their **depth & diameter** in real units."
)

calib_factor = st.number_input(
    "Calibration factor (pixels → cm). Default=1 (pixel units). Example: if 10 pixels = 1 cm, enter 0.1",
    min_value=0.0, value=1.0
)
unit = "px" if calib_factor == 1.0 else "cm"

uploaded_file = st.file_uploader("Upload Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file:
    file_type = uploaded_file.type.split("/")[0]
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if file_type == "image":
        image = Image.open(tfile.name).convert("RGB")
        results = run_yolo(image)
        depth_map = run_midas(image)

        yolo_vis, pothole_data = process_results(image, results, depth_map, calib_factor, unit)
        depth_vis = colorize_depth(depth_map)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("YOLO Detections")
            st.image(yolo_vis, channels="RGB", use_container_width=True)
        with col2:
            st.subheader("Depth Map (MiDaS)")
            st.image(depth_vis, channels="BGR", use_container_width=True)

        if pothole_data:
            st.subheader("📊 Pothole Measurements")
            df = pd.DataFrame(pothole_data)
            st.dataframe(df, use_container_width=True, height=300)

    elif file_type == "video":
        stframe1 = st.empty()
        stframe2 = st.empty()
        table_placeholder = st.empty()

        cap = cv2.VideoCapture(tfile.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            results = run_yolo(image)
            depth_map = run_midas(image)

            yolo_vis, pothole_data = process_results(image, results, depth_map, calib_factor, unit)
            depth_vis = colorize_depth(depth_map)

            stframe1.image(yolo_vis, channels="RGB", caption="YOLO Detections", use_container_width=True)
            stframe2.image(depth_vis, channels="BGR", caption="Depth Map", use_container_width=True)

            if pothole_data:
                df = pd.DataFrame(pothole_data)
                table_placeholder.dataframe(df, use_container_width=True, height=300)

        cap.release()
