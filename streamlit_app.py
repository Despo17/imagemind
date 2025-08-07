# streamlit_app.py (Styled for IMAGEMIND)

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="IMAGEMIND – CNN-Based Multi-Class Classifier",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Class Names
# ------------------------------
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# ------------------------------
# Load the Trained Model
# ------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("visionnet_tl.h5", compile=False)
    return model

model = load_model()

# ------------------------------
# Sidebar Info
# ------------------------------
st.sidebar.title("📚 About IMAGEMIND")
st.sidebar.markdown("""
**IMAGEMIND** is a deep learning-based multi-class image classifier powered by **MobileNetV2** and trained on the CIFAR-10 dataset.

🎯 It predicts the object in an uploaded image across 10 common categories.

👨‍💻 Developed by: Abilash  
📂 Model: `visionnet_tl.h5`
""")

# ------------------------------
# Main Interface
# ------------------------------
st.title("🧠 IMAGEMIND – CNN-Based Multi-Class Classifier")
st.markdown("Upload an image and let the AI predict its class. Make sure the image is clear and centered.")
st.markdown("---")

# ------------------------------
# Upload Image
# ------------------------------
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((96, 96))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Predict
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Display prediction
    st.markdown("---")
    st.subheader(f"🎯 Predicted Class: `{predicted_class}`")
    st.progress(float(confidence) / 100)
    st.write(f"🔍 Confidence: **{confidence:.2f}%**")

    # Show bar chart of all class probabilities
    st.markdown("---")
    st.subheader("📊 Prediction Probabilities")
    prob_dict = {class_names[i]: float(predictions[0][i]) for i in range(10)}
    st.bar_chart(prob_dict)
else:
    st.info("👈 Upload a `.jpg` or `.png` image to get started.")

# ------------------------------
# Footer
# ------------------------------
st.markdown("""
<p style='text-align: center;'>Made with ❤️ by Abilash — Powered by Streamlit & TensorFlow | Project: IMAGEMIND</p>
""", unsafe_allow_html=True)