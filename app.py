"""
Canny Edge Detector - Streamlit Web App
-----------------------------------------
Upload an image, tune the thresholds, and see the Canny edge-detected
result live in the browser.

Run locally:
    streamlit run app.py

Deploy for free:
    Push this file + requirements.txt to a GitHub repo, then deploy
    on Streamlit Community Cloud (share.streamlit.io) — see the
    deployment guide provided alongside this file.
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Canny Edge Detector", page_icon="🖼️", layout="wide")

st.title("🖼️ Canny Edge Detector")
st.caption("Computer Vision Project — built with Python & OpenCV")

st.markdown(
    """
    Upload an image and the app will detect edges using the **Canny Edge
    Detection** algorithm. Adjust the sliders to see how the thresholds
    affect which edges get picked up.
    """
)

with st.sidebar:
    st.header("Settings")
    low_threshold = st.slider("Low Threshold", 0, 255, 50)
    high_threshold = st.slider("High Threshold", 0, 255, 150)
    blur_amount = st.slider("Blur Kernel Size (odd number)", 1, 15, 5, step=2)
    st.markdown("---")
    st.markdown(
        "**How it works:**\n"
        "1. Convert to grayscale\n"
        "2. Gaussian blur (reduce noise)\n"
        "3. Compute intensity gradients\n"
        "4. Thin edges (non-max suppression)\n"
        "5. Double threshold + hysteresis"
    )

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

use_sample = False
if uploaded_file is None:
    use_sample = st.checkbox("No image? Use a built-in sample instead", value=True)

def load_sample():
    img = np.full((400, 600, 3), 245, dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (250, 200), (200, 130, 60), -1)
    cv2.circle(img, (450, 130), 90, (60, 180, 200), -1)
    cv2.ellipse(img, (300, 300), (120, 60), 30, 0, 360, (100, 200, 100), -1)
    cv2.putText(img, "Sample Image", (180, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, (30, 30, 30), 2)
    return img

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file).convert("RGB")
    image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
elif use_sample:
    image = load_sample()
else:
    image = None

if image is not None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    k = blur_amount if blur_amount % 2 == 1 else blur_amount + 1
    blurred = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(blurred, low_threshold, high_threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Original")
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("Grayscale + Blur")
        st.image(blurred, use_container_width=True)
    with col3:
        st.subheader("Canny Edges")
        st.image(edges, use_container_width=True)

    edges_pil = Image.fromarray(edges)
    import io
    buf = io.BytesIO()
    edges_pil.save(buf, format="PNG")
    st.download_button("⬇️ Download edge-detected image", buf.getvalue(),
                        file_name="canny_edges.png", mime="image/png")
else:
    st.info("Upload an image above, or check the sample box in the sidebar to try it out.")
