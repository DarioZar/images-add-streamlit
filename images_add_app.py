import streamlit as st
import cv2
import numpy as np

def translate_image(image, x, y):
    # Create the translation matrix
    M = np.float32([[1, 0, x], [0, 1, y]])
    # Apply the affine transformation
    return cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

def mean_filter(image, n):
    # Create a kernel for the mean of n neighbours
    kernel = np.ones((n, n), np.float32) / (n*n)
    # Apply the mean filter
    return cv2.filter2D(image, -1, kernel)

def apply_image_processing(gray_image, blue_image, green_image,
                           blue, green, use_mean_filter, n_mean):
    # Apply the translations on blue and green images
    shiftedblue = translate_image(blue_image, blue["x"], blue["y"])
    shiftedgreen = translate_image(green_image, green["x"], green["y"])
    # Apply the intensity scaling
    b = cv2.addWeighted(gray_image, 1, shiftedblue, blue["int"], 0)
    g = cv2.addWeighted(gray_image, 1, shiftedgreen, green["int"], 0)
    # Merge the channels
    result_image = cv2.merge([gray_image, g, b])

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
    col1, col2 = st.columns([3, 1])  # Image and Controls

    # Controls
    with col2:
        # Sliders
        with st.expander("Pump Parameters"):
            pump = {}
            pump["int"] = st.slider("Pump Intensity", 0.0, 1.5, 0.5)
            pump["x"] = st.slider("Pump Translate X", -shp[0], shp[0], 0, 1)
            pump["y"] = st.slider("Pump Translate Y", -shp[1], shp[1], 0, 1)
        
        with st.expander("Whitelight Parameters"):
            white = {}
            white["int"] = st.slider("Whitelight Intensity", 0.0, 1.5, 0.5)
            white["x"] = st.slider("Whitelight Translate X", -shp[0], shp[0], 0, 1)
            white["y"] = st.slider("Whitelight Translate Y", -shp[1], shp[1], 0, 1)

        # Checkbox for mean filter
        use_mean_filter = st.checkbox("Use Mean Filter")
        # Neighbours slider
        n_mean = st.slider("Neighbours for Mean Filter", 1, 20, 4, 1)

        st.image(gray_image, caption='Gray Image', use_column_width=True)

    # Image
    with col1:
        if gray_image is not None:
            result_image = apply_image_processing(gray_image, pump_image, whitelight_image,
                                                  pump, white, use_mean_filter, n_mean)
            # Display processed image
            st.image(result_image, caption='Processed Image', use_column_width=True)
else:
    st.warning("Please upload all three images.")
