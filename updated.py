import streamlit as st
import cv2
import torch
import tempfile
from ultralytics import YOLO
import numpy as np
import time

# Cache the YOLO model to avoid reloading on every run
@st.cache_resource
def load_model():
    return YOLO('best.pt')

# Load YOLOv8 model
model = load_model()

# Function to detect animals in video
def detect_animals(video_path):
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps

    st.write(f"Video Duration: {duration:.2f} seconds")
    st.write(f"Total Frames: {total_frames}")
    st.write(f"FPS: {fps}")

    # Initialize counters for detected animals
    animal_counts = {'tiger': 0, 'elephant': 0, 'rhino': 0}

    frame_count = 0
    start_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Run YOLOv8 model
        results = model.predict(frame)

        # Process results
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                label = model.names[cls]
                if label in animal_counts:  # Only count specific animals
                    animal_counts[label] += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = box.conf[0].item()

                    # Draw bounding box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Convert to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display frame
        stframe.image(frame, channels="RGB")

        # Update progress bar
        progress = frame_count / total_frames
        progress_bar.progress(progress)

        # Display status
        elapsed_time = time.time() - start_time
        status_text.text(f"Processed {frame_count}/{total_frames} frames | Elapsed Time: {elapsed_time:.2f}s")

    cap.release()

    # Display detection summary
    st.write("### Detection Summary")
    for animal, count in animal_counts.items():
        st.write(f"{animal.capitalize()}: {count}")

# Streamlit UI
st.title("Animal Detection in Video using YOLOv8")
uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov", "mkv"])

if uploaded_video:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tfile.write(uploaded_video.read())

    # Detect animals in the video
    detect_animals(tfile.name)