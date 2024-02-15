import streamlit as st
import cv2
import numpy as np

def mean_filter(image, n):
    kernel = np.ones((n, n), np.float32) / (n*n)
    return cv2.filter2D(image, -1, kernel)

def apply_image_processing(gray_image, blue_image, green_image,
                           pump, white, posx, posy,
                           use_mean_filter, n_mean):
    M = np.float32([[1, 0, posx], [0, 1, posy]])
    shifted = cv2.warpAffine(gray_image, M, (gray_image.shape[1], gray_image.shape[0]))

    b = cv2.addWeighted(shifted, 1, blue_image, pump, 0)
    g = cv2.addWeighted(shifted, 1, green_image, white, 0)
    result_image = cv2.merge([b, g, shifted])

    if use_mean_filter:
        result_image = mean_filter(result_image, n_mean)

    return result_image

def load_image(uploaded_file):
    if uploaded_file is not None:
        content = uploaded_file.read()
        nparr = np.frombuffer(content, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return None

# Streamlit App
st.title("Image Processing App")

# File Selectors on Sidebar
uploaded_gray_image = st.sidebar.file_uploader("Choose Gray Image", type=["jpg", "jpeg", "png", "bmp"])
uploaded_pump_image = st.sidebar.file_uploader("Choose Pump Image", type=["jpg", "jpeg", "png", "bmp"])
uploaded_whitelight_image = st.sidebar.file_uploader("Choose Whitelight Image", type=["jpg", "jpeg", "png", "bmp"])

# Load images
gray_image = load_image(uploaded_gray_image)
pump_image = load_image(uploaded_pump_image)
whitelight_image = load_image(uploaded_whitelight_image)

# Check shapes
if gray_image is not None and pump_image is not None and whitelight_image is not None:
    assert gray_image.shape == pump_image.shape == whitelight_image.shape
    shp = gray_image.shape

    # Layout
    col1, col2 = st.columns([2, 1])  # Image and Controls

    # Controls
    with col2:
        # Sliders
        with st.expander("Adjust Parameters"):
            pump = st.slider("Pump", 0.0, 1.0, 0.5)
            white = st.slider("Whitelight", 0.0, 1.0, 0.5)
            posx = st.slider("Translate X", -shp[0], shp[0], 0, 1)
            posy = st.slider("Translate Y", -shp[1], shp[1], 0, 1)

            # Checkbox for mean filter
            use_mean_filter = st.checkbox("Use Mean Filter")

            # Neighbours slider
            n_mean = st.slider("Neighbours for Mean Filter", 1, 20, 4, 1)

        st.image(gray_image, caption='Gray Image', use_column_width=True)

    # Image
    with col1:
        if gray_image is not None:
            result_image = apply_image_processing(gray_image, pump_image, whitelight_image,
                                                  pump, white, posx, posy,
                                                  use_mean_filter, n_mean)
            # Display processed image
            st.image(result_image, caption='Processed Image', use_column_width=True)
else:
    st.warning("Please upload all three images.")
