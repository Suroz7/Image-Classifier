import streamlit as st
from transformers import AutoModelForImageClassification, AutoFeatureExtractor
from PIL import Image
import torch
import requests
import json
import time
import os

# Page configuration
st.set_page_config(
    page_title="Mimesis",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# CSS for fade-in animation
st.markdown("""
    <style>
    @keyframes fadeIn {
        from { opacity: 0; }
        to { opacity: 1; }
    }
    .fade-in {
        animation: fadeIn 2s ease-in;
    }
    .loading-container {
        text-align: center;
        padding: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# Update logo path to use local file
LOGO_PATH = os.path.join("data", "asd.png")

if 'initial_load_complete' not in st.session_state:
    # Initial loading screen
    initial_container = st.empty()
    with initial_container.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            try:
                logo = Image.open(LOGO_PATH)
                st.markdown('<div class="fade-in loading-container">', unsafe_allow_html=True)
                st.image(logo, width=200)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Could not load logo: {str(e)}")
            
            st.markdown('<div class="fade-in"><h1>Welcome to Mimesis Primitive Image Classifier</h1></div>', unsafe_allow_html=True)
            time.sleep(2)
    
    initial_container.empty()
    st.session_state.initial_load_complete = True
    st.session_state.logo = logo  # Store logo in session state

# Main interface
st.title("Image Classification with ResNet-50")

def load_model():
    with st.spinner('Loading model...'):
        model_name = "microsoft/resnet-50"
        model = AutoModelForImageClassification.from_pretrained(model_name)
        feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
        imagenet_labels = requests.get("https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json").json()
    return model, feature_extractor, imagenet_labels

@st.cache_resource
def get_model():
    return load_model()

# Load model and resources
model, feature_extractor, imagenet_labels = get_model()

image_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if image_file is not None:
    image = Image.open(image_file)
    st.image(image, caption="Uploaded Image", width=300)  # Changed to 300px width
    
    # Processing loading container with shorter delay
    processing_container = st.empty()
    with processing_container.container():
        col1, col2, col3 = st.columns([1,2,1])
        with col2:
            st.markdown('<div class="fade-in loading-container">', unsafe_allow_html=True)
            st.image(st.session_state.logo, width=100)  # Reduced logo size
            st.markdown('<h3>Processing Your Image...</h3>', unsafe_allow_html=True)  # Changed to h3
            st.markdown('</div>', unsafe_allow_html=True)
            time.sleep(2)  # Reduced delay time
    
    processing_container.empty()
    
    with st.spinner('Analyzing image...'):
        inputs = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = logits.argmax(-1).item()

        predicted_class_label = imagenet_labels[str(predicted_class_id)][1]
        st.header(f"Predicted class: {predicted_class_label}")
