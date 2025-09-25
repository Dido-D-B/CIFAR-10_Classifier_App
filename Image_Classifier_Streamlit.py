import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import hashlib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
import io
import seaborn as sns
from sklearn.metrics import confusion_matrix
from keras.applications.resnet import preprocess_input
import keras

from pathlib import Path

# PATHS
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parent if (THIS_FILE.parent / "models").exists() else THIS_FILE.parent
if (PROJECT_ROOT.name == "app") and (PROJECT_ROOT.parent / "models").exists():
    PROJECT_ROOT = PROJECT_ROOT.parent
PATHS = {
    "model": PROJECT_ROOT / "models" / "resnet50_cifar10.keras",
    "cifar_samples": PROJECT_ROOT / "data" / "cifar_samples.npz",
    "fig_training": PROJECT_ROOT / "reports" / "figures" / "training_metrics.png",
    "fig_confusion": PROJECT_ROOT / "reports" / "figures" / "confusion_matrix_test.png",
    "fig_cifar": PROJECT_ROOT / "reports" / "figures" / "cifar.png",
    "fig_class_dist": PROJECT_ROOT / "reports" / "figures" / "class_distribution.png",
    # Logs (keep at repo root)
    "feedback_csv": PROJECT_ROOT / "feedback_log.csv",
    "usage_csv": PROJECT_ROOT / "usage_metrics.csv",
}

# CONFIGURATION
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']
FEEDBACK_FILE = str(PATHS["feedback_csv"])
USAGE_METRICS_FILE = str(PATHS["usage_csv"])
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'bmp']
DISPLAY_UPSCALE_FACTOR = 4
DISPLAY_SHARPEN_STRENGTH = 1.5
DISPLAY_UNSHARP_RADIUS = 2
DISPLAY_UNSHARP_PERCENT = 150

# PAGE CONFIG
st.set_page_config(
    page_title="CIFAR-10 AI Image Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CUSTOM CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1rem;
        margin-top: 1rem;
        color: #555;
    }
    .small-text {
        font-size: 0.8rem;
        color: #666;
    }
    .prediction-success {
        background: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .prediction-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
}

body {
    font-family: 'Arial', sans-serif;
}

img {
    max-width: 100%;
    height: auto;
}               
            
</style>
""", unsafe_allow_html=True)

if 'session_id' not in st.session_state:
    st.session_state.session_id = hashlib.md5(str(datetime.now()).encode()).hexdigest()[:8]
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'sample_idx' not in st.session_state:
    st.session_state.sample_idx = 0

@st.cache_resource
def load_model():
    with st.spinner("Loading AI model... This may take a moment."):
        try:
            model = tf.keras.models.load_model(str(PATHS["model"]))
            return model
        except Exception as e:
            st.error(f"❌ Could not load model: {e}")
            return None 

@st.cache_data
def load_cifar_samples():
    try:
        data = np.load(str(PATHS["cifar_samples"]))
        images = data["images"]
        labels = data["labels"]
        return images, labels
    except Exception as e:
        st.error(f"❌ Could not load CIFAR samples: {e}")
        return np.array([]), np.array([])

def validate_image(uploaded_file):
    if uploaded_file is None:
        return False, "No file uploaded"
    if uploaded_file.size > MAX_FILE_SIZE:
        return False, f"File too large (> {MAX_FILE_SIZE/1024/1024:.1f}MB)"
    if uploaded_file.name.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
    try:
        Image.open(uploaded_file).verify()
        return True, "Valid image"
    except:
        return False, "Invalid image file"
    
def preprocess_image_array(image_array, blur_radius=None):
    # Image is a PIL.Image or numpy array with shape (H, W, 3)
    if isinstance(image_array, Image.Image):
        # Resize to 32x32
        image_array = image_array.resize((32, 32))
        # Optionally apply blur
        if blur_radius and blur_radius > 0:
            image_array = image_array.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        # Convert to np.array
        image_array = np.array(image_array)

    # Ensure dtype and shape
    if image_array.dtype != np.uint8:
        image_array = (image_array * 255).astype(np.uint8)

    # ResNet50 preprocessing: scale to [-1, 1]
    preprocessed = preprocess_input(image_array)
    return np.expand_dims(preprocessed, axis=0)
def get_image_hash(image_bytes):
    return hashlib.md5(image_bytes).hexdigest()

def make_prediction(image_array):
    if model is None:
        st.error("❌ Model not loaded. Please restart the app.")
        return None, None, None, None, None, None
    try:
        start_time = time.time()
        preds = model.predict(image_array, verbose=0)
        prediction_time = time.time() - start_time
        class_idx = np.argmax(preds)
        confidence = np.max(preds)
        top_3_idx = np.argsort(preds[0])[-3:][::-1]
        top_3_probs = preds[0][top_3_idx]
        return preds, class_idx, confidence, top_3_idx, top_3_probs, prediction_time
    except Exception as e:
        st.error(f"❌ Prediction failed: {str(e)}")
        return None, None, None, None, None, None

def log_usage_metrics(image_type, prediction_time, confidence, image_size, blur_radius=None):
    metrics_data = {
        'timestamp': [datetime.now().isoformat()],
        'session_id': [st.session_state.session_id],
        'image_type': [image_type],
        'prediction_time': [prediction_time],
        'confidence': [confidence],
        'image_size': [str(image_size)],
        'blur_radius': [blur_radius]
    }
    new_row = pd.DataFrame(metrics_data)
    if os.path.exists(USAGE_METRICS_FILE):
        existing = pd.read_csv(USAGE_METRICS_FILE)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(USAGE_METRICS_FILE, index=False)

def log_feedback(image_type, prediction, correct, confidence, image_hash, sample_idx=None):
    feedback_data = {
    'timestamp': [datetime.now().isoformat()],
    'session_id': [st.session_state.session_id],
    'image_type': [image_type],
    'prediction': [prediction],
    'confidence': [confidence],
    'correct': [correct],
    'image_hash': [image_hash],
    'sample_idx': [sample_idx]
    }

    new_row = pd.DataFrame(feedback_data)
    if os.path.exists(FEEDBACK_FILE):
        existing = pd.read_csv(FEEDBACK_FILE)
        updated = pd.concat([existing, new_row], ignore_index=True)
    else:
        updated = new_row
    updated.to_csv(FEEDBACK_FILE, index=False)

def load_feedback_log():
    if os.path.exists(FEEDBACK_FILE):
        return pd.read_csv(FEEDBACK_FILE)
    else:
        return pd.DataFrame(columns=['timestamp','session_id','image_type','prediction','confidence','correct','image_hash','sample_idx'])
    
def create_enhanced_probability_chart(preds, top_3_idx, top_3_probs):
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(
            '<b>All Class Probabilities</b>',
            '<b>Top 3 Predictions</b>'
        ),
        row_heights=[0.55, 0.45],
        vertical_spacing=0.15
    )

    # Improved color scheme
    main_colors = ['#FF6361' if i == np.argmax(preds[0]) else '#58508D' for i in range(len(CLASSES))]
    top_colors = ['#FFA600', '#FF6361', '#BC5090']

    # ALL CLASSES
    fig.add_trace(
        go.Bar(
            x=CLASSES,
            y=preds[0],
            marker_color=main_colors,
            text=[f'{p:.1%}' for p in preds[0]],
            textposition='outside',
            opacity=0.85,
            hovertemplate='%{x}: %{y:.2%}<extra></extra>'
        ),
        row=1, col=1
    )

    # TOP 3 ONLY
    fig.add_trace(
        go.Bar(
            x=[CLASSES[i] for i in top_3_idx],
            y=top_3_probs,
            marker_color=top_colors,
            text=[f'{p:.1%}' for p in top_3_probs],
            textposition='outside',
            opacity=0.9,
            hovertemplate='%{x}: %{y:.2%}<extra></extra>'
        ),
        row=2, col=1
    )

    # Global layout
    fig.update_layout(
        height=750,
        margin=dict(t=80, b=120, l=50, r=50),
        showlegend=False,
        plot_bgcolor='white',
        font=dict(size=14)
)

    # Axes tweaks
    fig.update_xaxes(
        tickangle=45,
        showgrid=False,
        zeroline=False
    )
    fig.update_yaxes(
        tickformat=".0%",
        showgrid=True,
        gridcolor='rgba(200,200,200,0.3)',
        zeroline=False
    )

    return fig

def display_confidence_meter(confidence):
    if confidence > 0.8:
        color, level, emoji = "#28a745", "High", "🟢"
    elif confidence > 0.5:
        color, level, emoji = "#ffc107", "Medium", "🟡"
    else:
        color, level, emoji = "#dc3545", "Low", "🔴"
    st.markdown(f"""
    <div class="metric-container">
        <h4>{emoji} Confidence Level: {level}</h4>
        <div style="background-color: #e9ecef; border-radius: 10px; padding: 5px;">
            <div style="background-color: {color}; width: {confidence*100:.1f}%; height: 20px; border-radius: 5px; text-align: center; line-height: 20px; color: white; font-weight: bold;">
                {confidence:.1%}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# LOAD MODEL AND DATA
model = load_model()
cifar_images, cifar_labels = load_cifar_samples()

# PAGE TITLE
st.markdown("""
<div class="main-header">
    <h1>CIFAR-10 AI Image Classifier</h1>
    <p>
    Powered by ResNet50 Deep Learning | Computer Vision Project by 
    <a href="https://www.linkedin.com/in/dido-de-boodt/" target="_blank" style="color: #ffffff; text-decoration: underline;">
    <strong>Dido De Boodt</strong>
    </a>
    </p>
</div>
""", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to:",
    ("About this Project", "Model Insights", "Upload Your Image", "CIFAR-10 Samples", "Analytics Dashboard")
)

st.sidebar.header("Session Stats")
st.sidebar.markdown(f"<p class='small-text'>Total Predictions: {st.session_state.total_predictions}</p>", unsafe_allow_html=True)
st.sidebar.markdown(f"<p class='small-text'>Session ID: {st.session_state.session_id}</p>", unsafe_allow_html=True)

# FEEDBACK LOG MANAGEMENT
if st.sidebar.button("🗑️ Clear Feedback Logs"):
    if os.path.exists(FEEDBACK_FILE):
        os.remove(FEEDBACK_FILE)
    st.sidebar.success("Feedback log cleared!")

with st.sidebar.expander("Advanced Settings"):
    blur_radius = st.slider("Blur Strength (for uploaded images)", 0.0, 5.0, 1.5, 0.1)
    show_probabilities = st.checkbox("Show detailed probabilities", value=True)
    confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

    if confidence_threshold < 0.2 or confidence_threshold > 0.9:
        st.warning("⚠️ Extreme thresholds may reduce useful predictions!")

# MAIN CONTENT
if section == "About this Project":
    st.header("About this Project")
    st.markdown("""
The goal of this project is to build an **image classification model** for the CIFAR-10 dataset using transfer learning with **ResNet50**.

**CIFAR-10** is a well-known computer vision benchmark containing 60,000 32×32 color images across 10 mutually exclusive classes. 
""")
    # Display dataset images and class distribution
    st.subheader("CIFAR-10 Dataset Overview")
    # Format images for display

    col1, col2 = st.columns(2)

    with col1:
        st.image(str(PATHS["fig_cifar"]), caption="CIFAR-10 Dataset", use_container_width=True)

    with col2:
        st.image(str(PATHS["fig_class_dist"]), caption="CIFAR-10 Class Distribution", use_container_width=True)
    
    st.subheader("Project Overview")
    st.markdown("""
 For this project, we constrain ourselves to 10,000 training samples to simulate limited-data scenarios and reduce training time. Our approach leverages a **pretrained ResNet50 model** (trained on ImageNet) as a feature extractor. We first freeze the pretrained layers and train only the custom classification head, then unfreeze the entire network for fine-tuning.

* Importing and exploring the CIFAR-10 dataset
* Preprocessing images for ResNet50
* Building a transfer learning pipeline with a frozen base model
* Training a custom classification head
* Fine-tuning the entire model
* Evaluating model performance on the test set

This project aims to demonstrate the power of transfer learning for small-scale datasets, highlight practical techniques for fine-tuning deep networks, and develop hands-on experience with modern computer vision workflows.
""")
    
    st.subheader("Challenges")
    st.markdown("""
Throughout this project, several practical challenges emerged that shaped both the modeling strategy and the final results:
                
* **Small Training Set:** With only 10,000 training images (a subset of CIFAR-10), the model was prone to overfitting and struggled to generalize to the test set. More data would likely yield significant gains.
* **Low Resolution Images:** ResNet50 is designed for larger inputs (224x224), while CIFAR-10 images are 32x32. Upsampling helps but cannot fully recover lost detail, limiting feature extraction quality.
* **Confusing Classes:** Classes like cat, dog, and bird are visually similar and small in frame. Even with augmentation and label smoothing, precision and recall for these classes remained lower.
* **Long Training Times:** Fine-tuning deeper layers (e.g. top 20) significantly increased training time per epoch. Iterating on hyperparameters required patience and careful planning.

These challenges highlight the real-world complexity of adapting cutting-edge deep learning models to constrained datasets. An invaluable learning experience for deploying computer vision pipelines thoughtfully.
""")
    
    st.subheader("Key Takeaways")
    st.markdown("""
✅  Even with limited data, transfer learning and careful fine-tuning can deliver solid results.

✅ Layer-wise unfreezing with tuned learning rates is effective for adapting large models.

✅ Visualization (accuracy/loss curves, confusion matrices) is essential for diagnosing model behavior and improving training strategy.

✅ Data augmentation remains vital when working with small or imbalanced datasets.                
""")
    
if section == "Upload Your Image":
    st.header("Upload Your Own Image")
    st.markdown("""
Classes: **airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck**
                """)

    uploaded_file = st.file_uploader("Choose an image...", type=ALLOWED_EXTENSIONS)

    if uploaded_file is not None:
        is_valid, message = validate_image(uploaded_file)
        if not is_valid:
            st.error(f"❌ {message}")
        else:
            uploaded_file.seek(0)
            image_bytes = uploaded_file.read()
            image_hash = get_image_hash(image_bytes)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)

            resized_image = image.resize((32, 32))
            if blur_radius > 0:
                processed_image = resized_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            else:
                processed_image = resized_image

            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, use_container_width=True)

            img_array = preprocess_image_array(processed_image, blur_radius)
            result = make_prediction(img_array)

            if result[0] is not None:
                preds, class_idx, confidence, top_3_idx, top_3_probs, prediction_time = result
                st.session_state.total_predictions += 1
                log_usage_metrics("uploaded", prediction_time, confidence, image.size, blur_radius)

                st.markdown("---")
                st.subheader("Prediction Results")
                if confidence >= confidence_threshold:
                    st.markdown(f"<div class='prediction-success'><h3>✅ Prediction: {CLASSES[class_idx]}</h3><p>Confidence: {confidence:.1%} | Time: {prediction_time:.2f}s</p></div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='prediction-warning'><h3>⚠️ Uncertain Prediction: {CLASSES[class_idx]}</h3><p>Low confidence: {confidence:.1%}</p></div>", unsafe_allow_html=True)
                
                display_confidence_meter(confidence)
                st.subheader("Top 3 Predictions")
                for i, (idx, prob) in enumerate(zip(top_3_idx, top_3_probs)):
                    st.write(f"{i+1}. **{CLASSES[idx]}** — {prob:.1%}")

                if show_probabilities:
                    st.subheader("Detailed Analysis")
                    st.plotly_chart(create_enhanced_probability_chart(preds, top_3_idx, top_3_probs), use_container_width=True)

                st.subheader("Was this prediction correct?")
                c1, c2, c3 = st.columns(3)
                with c1:
                    if st.button("✅ Yes"):
                        log_feedback("uploaded", CLASSES[class_idx], True, confidence, image_hash)
                        st.success("Thanks for your feedback!")
                with c2:
                    if st.button("❌ No"):
                        log_feedback("uploaded", CLASSES[class_idx], False, confidence, image_hash)
                        st.warning("Thanks! We'll try to improve.")
                with c3:
                    if st.button("🤔 Unsure"):
                        st.info("No worries! Try another image.")
        
elif section == "CIFAR-10 Samples":
    st.header("CIFAR-10 Sample Classification")

    if len(cifar_images) == 0:
        st.error("❌ No sample images available. Check your 'cifar_samples.npz' file.")
    else:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("Choose a Sample")
            sample_idx = st.slider("Sample Index", 0, len(cifar_images)-1, st.session_state.sample_idx)
            st.session_state.sample_idx = sample_idx

            if st.button("Random Sample"):
                st.session_state.sample_idx = np.random.randint(0, len(cifar_images))
                st.rerun()

        with col2:
            st.info("**Tip:** Random sampling helps explore different categories!")        

        # Get the sample image (keep the raw numpy array unchanged for prediction)
        sample_image = cifar_images[st.session_state.sample_idx]

        # Create a *separate* enhanced DISPLAY IMAGE ONLY (don't modify sample_image itself)
        display_image = Image.fromarray(sample_image)

        # Upscale for clearer human viewing with LANCZOS resampling
        upscale_factor = 8
        upscale_size = (32 * upscale_factor, 32 * upscale_factor)
        display_image = display_image.resize(upscale_size, Image.Resampling.LANCZOS)

        # Apply enhancements only for display
        display_image = ImageEnhance.Contrast(display_image).enhance(1.3)
        display_image = ImageEnhance.Sharpness(display_image).enhance(2.0)
        display_image = ImageEnhance.Color(display_image).enhance(1.2)
        display_image = display_image.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=200, threshold=2)
        )
            
        # Display images side by side
        col_img1, col_img2 = st.columns(2)
        
        with col_img1:
            st.markdown("**Original (32×32)**")
            # Show original at a reasonable size without blur
            original_display = Image.fromarray(sample_image)
            original_display = original_display.resize((128, 128), Image.Resampling.NEAREST)
            st.image(original_display, caption=f"Sample #{st.session_state.sample_idx}")
        
        with col_img2:
            st.markdown(f"**Enhanced Display ({upscale_size[0]}×{upscale_size[1]})**")
            st.image(display_image, caption="Enhanced for Clarity", use_container_width=True)

        # Use original image for prediction (not the enhanced one)
        input_array = preprocess_image_array(sample_image)
        result = make_prediction(input_array)

        if result[0] is not None:
            preds, class_idx, confidence, top_3_idx, top_3_probs, prediction_time = result
            st.session_state.total_predictions += 1
            log_usage_metrics("cifar_sample", prediction_time, confidence, sample_image.shape[:2])

            # Compare prediction to ground truth
            true_label_idx = np.argmax(cifar_labels[st.session_state.sample_idx])
            is_correct = (class_idx == true_label_idx)
            log_feedback("cifar_sample", CLASSES[class_idx], is_correct, confidence, "sample_image", sample_idx=st.session_state.sample_idx)

            st.subheader("Classification Results")
            
            # Show true label vs prediction
            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.success(f"**Prediction:** {CLASSES[class_idx]}")
                st.write(f"Confidence: {confidence:.1%}")
            
            with col_result2:
                st.info(f"**True Label:** {CLASSES[true_label_idx]}")
                if is_correct:
                    st.write("✅ **Correct Prediction!**")
                else:
                    st.write("❌ **Incorrect Prediction**")
            
            st.write(f"Processing Time: {prediction_time:.2f}s")
            display_confidence_meter(confidence)

            if show_probabilities:
                st.subheader("Probability Distribution")
                st.plotly_chart(create_enhanced_probability_chart(preds, top_3_idx, top_3_probs), use_container_width=True)
            
# ANALYTICS DASHBOARD                
elif section == "Analytics Dashboard":
    st.header("Analytics Dashboard")

    feedback_df = load_feedback_log()
    if 'sample_idx' in feedback_df.columns and len(cifar_labels) > 0:
        mask = (feedback_df['image_type'] == 'cifar_sample') & feedback_df['sample_idx'].notnull()
        if mask.any():
            sample_indices = feedback_df.loc[mask, 'sample_idx'].astype(int)
            true_labels = np.argmax(cifar_labels[sample_indices], axis=1)
            true_class_names = [CLASSES[i] for i in true_labels]
            feedback_df.loc[mask, 'true_label'] = true_class_names
            feedback_df.loc[mask, 'correct'] = feedback_df.loc[mask, 'prediction'] == feedback_df.loc[mask, 'true_label']
    if feedback_df.empty:
        st.info("No feedback data yet. Make some predictions to see analytics!")
    else:
        # Recompute correct flag for cifar_sample entries using stored sample_idx
        if 'sample_idx' in feedback_df.columns and len(cifar_labels) > 0:
            mask = (feedback_df['image_type'] == 'cifar_sample') & feedback_df['sample_idx'].notnull()
            sample_indices = feedback_df.loc[mask, 'sample_idx'].astype(int)
            true_labels = np.argmax(cifar_labels[sample_indices], axis=1)
            true_class_names = [CLASSES[i] for i in true_labels]
            feedback_df.loc[mask, 'true_label'] = true_class_names
            feedback_df.loc[mask, 'correct'] = feedback_df.loc[mask, 'prediction'] == feedback_df.loc[mask, 'true_label']
        
        # Metrics
        st.subheader("Overview Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Feedback", len(feedback_df))
        with col2:
            sample_feedback = feedback_df[feedback_df['image_type'] == 'cifar_sample']
            acc = sample_feedback['correct'].mean() if not sample_feedback.empty else 0
            st.metric("Accuracy", f"{acc:.1%}")
        with col3:
            avg_conf = feedback_df['confidence'].mean() if 'confidence' in feedback_df else 0
            st.metric("Avg Confidence", f"{avg_conf:.1%}")
        with col4:
            st.metric("Unique Sessions", feedback_df['session_id'].nunique())

        # Charts
        if 'prediction' in feedback_df.columns:
            st.subheader("Accuracy by Class")
            sample_feedback = feedback_df[feedback_df['image_type'] == 'cifar_sample']
            if not sample_feedback.empty:
                class_acc = sample_feedback.groupby('prediction')['correct'].mean()
                fig = px.bar(class_acc, labels={'value':'Accuracy'})
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No CIFAR-10 sample feedback yet to calculate class-wise accuracy.")

        if 'confidence' in feedback_df.columns:
            st.subheader("Confidence Score Distribution")
            fig = px.histogram(feedback_df, x='confidence', nbins=20)
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Raw Feedback Data"):
            st.dataframe(feedback_df.sort_values('timestamp', ascending=False))

elif section == "Model Insights":
    with st.container():
        st.subheader("Model Training Visualizations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Training Metrics**")
            st.image(str(PATHS["fig_training"]), use_container_width=True)

        with col2:
            st.markdown("**Confusion Matrix (Test Set 10,000 images)**")
            st.image(str(PATHS["fig_confusion"]), use_container_width=True)

        st.markdown("""
        These charts provide insight into model training:

        - **Training Metrics**: Accuracy and loss curves over all phases.
        - **Confusion Matrix**: Test set performance showing class-wise predictions.
        """)
        
        # Notes, markdown, metrics 
        st.subheader("Performance Metrics")
        st.markdown("""
        - Model: ResNet50 Fine-tuned on CIFAR-10            
        - Validation Accuracy: **~68%**
        - Test Accuracy: **~65%**
        - Training Accuracy: **~67%**
        - Optimizer: AdamW
        - Learning Rate Scheduling over 4 phases
        - Data Augmentation applied
        """)       

# FOOTER            
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p><strong>Enhanced CIFAR-10 Classifier</strong> | Built with Streamlit & TensorFlow</p>
</div>
""", unsafe_allow_html=True)
