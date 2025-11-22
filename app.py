import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
import os
import requests
import numpy as np

# --- CONFIGURATION ---
# The URL to your model file on Hugging Face
MODEL_URL = "https://huggingface.co/Skindoc/streamlit9/resolve/main/best_model_20251122_103707.pth"
# Local path to save the model after download
LOCAL_MODEL_PATH = "best_model_20251122_103707.pth"

# Class names must match your training exactly
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

# SAFETY THRESHOLDS (Calibrated)
THRESHOLDS = {
    'mel': 0.1362,  # >13.6% probability -> Flag as Melanoma
    'scc': 0.0003   # >0.03% probability -> Flag as SCC
}

# --- HELPER FUNCTIONS ---

def download_model(url, save_path):
    """Downloads the model file if it doesn't exist locally (Crucial for Deployment)."""
    if not os.path.exists(save_path):
        with st.spinner(f"Downloading model from Hugging Face... (This happens only once)"):
            try:
                # Use a temporary directory for robustness on cloud platforms
                if 'STREAMLIT_SERVER_SESSION_ID' in os.environ:
                    # Use the current directory if it's Streamlit Cloud
                    pass
                
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(save_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                st.success("Model downloaded successfully!")
            except Exception as e:
                st.error(f"Failed to download model: {e}")
                st.stop()
    return save_path

@st.cache_resource
def load_model(path):
    """Loads the model into memory and caches it."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the empty architecture
    model = timm.create_model("tf_efficientnet_b4", pretrained=False, num_classes=len(CLASS_NAMES))
    
    # Load weights
    try:
        checkpoint = torch.load(path, map_location=device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model weights: {e}")
        st.stop()
        
    model.to(device)
    model.eval()
    return model, device

def predict(model, device, image):
    """Runs the inference logic with safety thresholds."""
    # Preprocessing (Same as validation)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    img_t = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(img_t)
        probs = F.softmax(outputs, dim=1)[0].cpu().numpy()
        
    results = {name: float(p) for name, p in zip(CLASS_NAMES, probs)}
    
    # --- SAFETY LOGIC ---
    
    # Priority 1: Melanoma Check
    if results['mel'] >= THRESHOLDS['mel']:
        return {
            "diagnosis": "Melanoma (High Risk)",
            "risk": "HIGH",
            "probabilities": results,
            "reason": f"Melanoma probability ({results['mel']:.1%}) exceeds safety threshold ({THRESHOLDS['mel']:.1%})"
        }
        
    # Priority 2: SCC Check
    if results['scc'] >= THRESHOLDS['scc']:
        return {
            "diagnosis": "Squamous Cell Carcinoma (Risk)",
            "risk": "HIGH",
            "probabilities": results,
            "reason": f"SCC probability ({results['scc']:.4%}) exceeds safety threshold ({THRESHOLDS['scc']:.4%})"
        }

    # Priority 3: Standard Winner
    winner_idx = probs.argmax()
    winner_class = CLASS_NAMES[winner_idx]
    
    # Map to friendly names
    friendly_names = {
        'nv': 'Melanocytic Nevus (Common Mole)',
        'bkl': 'Benign Keratosis',
        'bcc': 'Basal Cell Carcinoma',
        'vasc': 'Vascular Lesion',
        'df': 'Dermatofibroma',
        'akiec': 'Actinic Keratosis',
        'mel': 'Melanoma',
        'scc': 'Squamous Cell Carcinoma'
    }
    
    diagnosis = friendly_names.get(winner_class, winner_class.upper())
    
    if winner_class in ['bcc', 'akiec']:
        risk = "Medium"
    else:
        risk = "Low"
        
    return {
        "diagnosis": diagnosis,
        "risk": risk,
        "probabilities": results,
        "reason": "Most likely classification"
    }

# --- APP UI ---
st.set_page_config(page_title="DermScan AI", page_icon="🩺")

st.title("🩺 DermScan AI: Professional Skin Lesion Analysis")
st.markdown("""
**Medical Safety Notice:** This tool uses an AI model calibrated for **High Sensitivity (97% Recall)**. 
It is designed to flag *potential* risks for professional review. It is not a replacement for a doctor.
""")

# Load Model Logic
# 1. Download model (if necessary)
model_path = download_model(MODEL_URL, LOCAL_MODEL_PATH)
# 2. Load model (into cache)
model, device = load_model(model_path)

# File Uploader
uploaded_file = st.file_uploader("Upload a dermoscopic image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Analyzed Image", use_column_width=True)
    
    with st.spinner("Analyzing..."):
        result = predict(model, device, image)
        
    st.divider()
    
    # Display Badge
    if result["risk"] == "HIGH":
        st.error(f"⚠️ RISK LEVEL: {result['risk']}")
    elif result["risk"] == "Medium":
        st.warning(f"🔸 RISK LEVEL: {result['risk']}")
    else:
        st.success(f"💚 RISK LEVEL: {result['risk']}")
        
    st.subheader(f"Diagnosis: {result['diagnosis']}")
    st.info(f"Analysis: {result['reason']}")
    
    with st.expander("View Detailed Probabilities"):
        # Sort and display
        sorted_probs = dict(sorted(result['probabilities'].items(), key=lambda item: item[1], reverse=True))
        for cls, prob in sorted_probs.items():
            st.progress(prob, text=f"{cls}: {prob:.2%}")
