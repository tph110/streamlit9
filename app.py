import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
import timm
from torchvision import transforms
import os
import requests

# --- CONFIGURATION ---
MODEL_URL = "https://huggingface.co/Skindoc/streamlit9/resolve/main/best_model_20251122_103707.pth"
LOCAL_MODEL_PATH = "best_model_20251122_103707.pth"

CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'scc', 'vasc']

# Friendly display names for each class
FRIENDLY_NAMES = {
    'akiec': 'Actinic Keratosis (Pre-cancerous)',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevus (Common Mole)',
    'scc': 'Squamous Cell Carcinoma',
    'vasc': 'Vascular Lesion'
}

# Risk categories for each class
RISK_CATEGORIES = {
    'mel': 'HIGH',
    'scc': 'HIGH', 
    'bcc': 'MEDIUM',
    'akiec': 'MEDIUM',
    'bkl': 'LOW',
    'df': 'LOW',
    'nv': 'LOW',
    'vasc': 'LOW'
}

# Safety thresholds - flag these even if not the top prediction
SAFETY_THRESHOLDS = {
    'mel': 0.1362,   # Flag melanoma if probability > 13.6%
    'scc': 0.0003    # Flag SCC if probability > 0.03%
}


@st.cache_resource
def load_model():
    """Downloads model if needed and loads it into memory."""
    device = torch.device("cpu")
    
    # Download model if not present locally
    if not os.path.exists(LOCAL_MODEL_PATH):
        st.info("📥 Downloading model (first time only)...")
        try:
            response = requests.get(MODEL_URL, stream=True, timeout=300)
            response.raise_for_status()
            
            with open(LOCAL_MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except requests.RequestException as e:
            st.error(f"Failed to download model: {e}")
            return None, None
    
    # Load the model
    try:
        model = timm.create_model(
            "tf_efficientnet_b4", 
            pretrained=False, 
            num_classes=len(CLASS_NAMES)
        )
        
        checkpoint = torch.load(LOCAL_MODEL_PATH, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model, device
        
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None


def preprocess_image(image):
    """Prepare image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])
    return transform(image).unsqueeze(0)


def analyze_image(model, device, image):
    """Run inference and apply safety logic."""
    img_tensor = preprocess_image(image).to(device)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = F.softmax(outputs, dim=1)[0].cpu().numpy()
    
    # Create results dictionary
    results = {name: float(prob) for name, prob in zip(CLASS_NAMES, probabilities)}
    
    # Safety check: Melanoma
    if results['mel'] >= SAFETY_THRESHOLDS['mel']:
        return {
            "prediction": "mel",
            "diagnosis": FRIENDLY_NAMES['mel'],
            "risk": "HIGH",
            "probabilities": results,
            "flagged": True,
            "flag_reason": f"Melanoma probability ({results['mel']:.1%}) exceeds safety threshold ({SAFETY_THRESHOLDS['mel']:.1%})"
        }
    
    # Safety check: Squamous Cell Carcinoma
    if results['scc'] >= SAFETY_THRESHOLDS['scc']:
        return {
            "prediction": "scc",
            "diagnosis": FRIENDLY_NAMES['scc'],
            "risk": "HIGH",
            "probabilities": results,
            "flagged": True,
            "flag_reason": f"SCC probability ({results['scc']:.4%}) exceeds safety threshold ({SAFETY_THRESHOLDS['scc']:.4%})"
        }
    
    # Standard prediction: highest probability class
    top_class = CLASS_NAMES[probabilities.argmax()]
    
    return {
        "prediction": top_class,
        "diagnosis": FRIENDLY_NAMES[top_class],
        "risk": RISK_CATEGORIES[top_class],
        "probabilities": results,
        "flagged": False,
        "flag_reason": None
    }


def display_risk_badge(risk_level):
    """Display a colored risk indicator."""
    if risk_level == "HIGH":
        st.error(f"⚠️ **Risk Level: {risk_level}** - Urgent dermatologist review recommended")
    elif risk_level == "MEDIUM":
        st.warning(f"🔸 **Risk Level: {risk_level}** - Dermatologist consultation advised")
    else:
        st.success(f"✅ **Risk Level: {risk_level}** - Likely benign, monitor for changes")


def display_probabilities(probabilities):
    """Display class probabilities as a bar chart."""
    # Sort by probability (descending)
    sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    
    for class_name, prob in sorted_probs:
        friendly_name = FRIENDLY_NAMES[class_name]
        risk = RISK_CATEGORIES[class_name]
        
        # Color coding based on risk
        if risk == "HIGH":
            bar_color = "🔴"
        elif risk == "MEDIUM":
            bar_color = "🟠"
        else:
            bar_color = "🟢"
        
        # Create visual bar
        bar_length = int(prob * 30)
        bar = "█" * bar_length + "░" * (30 - bar_length)
        
        st.text(f"{bar_color} {class_name.upper():5} │{bar}│ {prob:6.2%}  {friendly_name}")


# --- STREAMLIT APP ---
st.set_page_config(
    page_title="DermScan AI",
    page_icon="🩺",
    layout="centered"
)

st.title("🩺 DermScan AI")
st.subheader("Professional Skin Lesion Analysis")

st.markdown("""
---
**⚕️ Medical Disclaimer**  
This AI tool is designed for **screening purposes only** and is calibrated for high sensitivity 
(97% recall for melanoma detection). It is **not a substitute for professional medical diagnosis**. 
Always consult a qualified dermatologist for clinical evaluation.

**Training Data:** HAM10000 & ISIC2019 datasets  
**Classifications:** 8 lesion types including melanoma, BCC, SCC, and benign conditions
""")

st.divider()

# Load model
model, device = load_model()

if model is None:
    st.error("❌ Model failed to load. Please refresh the page or check the model URL.")
    st.stop()

st.success("✅ AI model loaded and ready", icon="🤖")

# File upload
uploaded_file = st.file_uploader(
    "Upload a dermoscopic image",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a high-quality dermoscopic image"
)

if uploaded_file is not None:
    # Load and display image
    image = Image.open(uploaded_file).convert('RGB')
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Run analysis
    with st.spinner("🔬 Analyzing image..."):
        result = analyze_image(model, device, image)
    
    with col2:
        st.markdown("### Analysis Result")
        
        # Risk badge
        display_risk_badge(result["risk"])
        
        # Diagnosis
        st.markdown(f"**Prediction:** {result['diagnosis']}")
        
        # Confidence
        confidence = result["probabilities"][result["prediction"]]
        st.markdown(f"**Confidence:** {confidence:.1%}")
        
        # Safety flag note
        if result["flagged"]:
            st.warning(f"🚨 **Safety Flag:** {result['flag_reason']}")
    
    st.divider()
    
    # Detailed probabilities
    with st.expander("📊 View All Class Probabilities"):
        display_probabilities(result["probabilities"])
    
    # Recommendations based on risk
    with st.expander("📋 Recommended Actions"):
        if result["risk"] == "HIGH":
            st.markdown("""
            **Urgent Actions:**
            - Schedule an appointment with a dermatologist as soon as possible
            - Do not attempt to treat or remove the lesion yourself
            - Document any changes in size, color, or shape
            - Bring this analysis to your appointment for reference
            """)
        elif result["risk"] == "MEDIUM":
            st.markdown("""
            **Recommended Actions:**
            - Schedule a dermatologist consultation within 2-4 weeks
            - Monitor the lesion for any changes
            - Take photos to track progression
            - Note any symptoms like itching, bleeding, or pain
            """)
        else:
            st.markdown("""
            **General Guidance:**
            - Continue regular skin self-examinations
            - Monitor for changes using the ABCDE criteria
            - Annual skin checks with a dermatologist are recommended
            - Protect skin from excessive sun exposure
            """)

st.divider()
st.caption("DermScan AI v1.0 | For research and screening purposes only | © 2024")
