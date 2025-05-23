import streamlit as st

# Set page config must be FIRST
st.set_page_config(page_title="Alzheimer's Stage Predictor", layout="centered")

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# Load model once
@st.cache_resource
def load_my_model():
    return load_model("model/Augmented_Alzheimer_Model_95.h5")

model = load_my_model()

# Define class names
class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

# Preprocess image
def preprocess_image(img):
    img = img.resize((244, 244))  # Resize to match model input
    img_array = np.array(img)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]  # Remove alpha channel if present
    img_array = preprocess_input(img_array.astype(np.float32))  # Use Xception preprocessing
    return np.expand_dims(img_array, axis=0)

# Explain with LIME
def explain_with_lime(img, model):
    def predict(images):
        processed_images = []
        for img in images:
            pil_img = Image.fromarray(img.astype('uint8')).convert("RGB")
            processed_img = preprocess_image(pil_img)
            processed_images.append(processed_img[0])  # shape: (244,244,3)
        processed_images = np.array(processed_images)
        return model.predict(processed_images)

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        np.array(img),
        predict,
        top_labels=1,
        hide_color=0,
        num_samples=1000
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5,
        hide_rest=False
    )
    return mark_boundaries(temp, mask)

# Streamlit UI
st.title("🧠 Alzheimer's Disease Stage Predictor")

uploaded_file = st.file_uploader("Upload a brain MRI image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and Predict
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)
    predicted_class = class_names[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.success(f"🧠 Detected stage: **{predicted_class}**")

    # LIME Explanation
    st.subheader("LIME Explainability:")

    lime_result = explain_with_lime(image.resize((244, 244)), model)

    fig, ax = plt.subplots()
    ax.imshow(lime_result)
    ax.axis('off')
    st.pyplot(fig)
