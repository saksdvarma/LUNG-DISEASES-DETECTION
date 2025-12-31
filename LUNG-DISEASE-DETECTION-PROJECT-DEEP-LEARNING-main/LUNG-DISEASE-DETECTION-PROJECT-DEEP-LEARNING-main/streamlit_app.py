#!/usr/bin/env python3
"""
Streamlit App for Lung Disease Detection
Uses trained models to predict COVID-19, Normal, or Viral Pneumonia from chest X-rays
"""

import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
import timm
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json
import time

# Import model classes from project.py
import sys
sys.path.append('.')
from project import TraditionalCNN, create_resnet50, create_efficientnet, Config

# Page configuration
st.set_page_config(
    page_title="Lung Disease Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
    }
    .prediction-box {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .model-name {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    .confidence-high {
        color: #27ae60;
        font-weight: bold;
    }
    .confidence-medium {
        color: #f39c12;
        font-weight: bold;
    }
    .confidence-low {
        color: #e74c3c;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# Class names
CLASS_NAMES = ['COVID', 'Normal', 'Viral Pneumonia']
CLASS_COLORS = ['#e74c3c', '#27ae60', '#3498db']

# Device configuration
@st.cache_resource
def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Model loading functions
@st.cache_resource
def load_traditional_cnn(model_path, device):
    """Load Traditional CNN model"""
    model = TraditionalCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_resnet50(model_path, device):
    """Load ResNet50 model"""
    model = create_resnet50(num_classes=3, pretrained=False)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Error loading ResNet50: {str(e)}")
        raise
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_efficientnet(model_path, device):
    """Load EfficientNetB0 model"""
    model = create_efficientnet(num_classes=3, pretrained=False)
    try:
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
    except RuntimeError as e:
        st.error(f"Error loading EfficientNetB0: {str(e)}")
        raise
    model.to(device)
    model.eval()
    return model

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model inference"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    if isinstance(image, Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
    else:
        image = Image.fromarray(image)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor

# Prediction function
def predict(model, image_tensor, device, model_name):
    """Run prediction on a single model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        # Get probabilities for all classes
        probs = probabilities[0].cpu().numpy()
        pred_class = predicted.item()
        confidence_score = confidence.item()
    
    return {
        'predicted_class': pred_class,
        'class_name': CLASS_NAMES[pred_class],
        'confidence': confidence_score,
        'probabilities': probs,
        'model_name': model_name
    }

# Load model performance data
@st.cache_data
def load_model_performance():
    """Load model performance metrics"""
    try:
        with open('outputs/results_summary.json', 'r') as f:
            return json.load(f)
    except:
        return {
            "models": {
                "Traditional CNN": {
                    "accuracy": 0.9310,
                    "precision": 0.9331,
                    "recall": 0.9310,
                    "f1_score": 0.9291
                },
                "ResNet50": {
                    "accuracy": 0.9837,
                    "precision": 0.9838,
                    "recall": 0.9837,
                    "f1_score": 0.9837
                },
                "EfficientNetB0": {
                    "accuracy": 0.9886,
                    "precision": 0.9886,
                    "recall": 0.9886,
                    "f1_score": 0.9886
                }
            },
            "best_model": "EfficientNetB0"
        }

def main():
    # Header
    st.markdown('<h1 class="main-header">🫁 Lung Disease Detection System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Model Information")
        st.markdown("### Trained Models")
        st.markdown("""
        - **Traditional CNN** (Baseline)
        - **ResNet50** (Transfer Learning)
        - **EfficientNetB0** (State-of-the-art)
        """)
        
        st.markdown("### Model Performance")
        perf_data = load_model_performance()
        
        for model_name, metrics in perf_data['models'].items():
            st.markdown(f"**{model_name}**")
            st.metric("Accuracy", f"{metrics['accuracy']*100:.2f}%")
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}")
            st.markdown("---")
        
        st.markdown(f"**🏆 Best Model:** {perf_data['best_model']}")
        
        st.markdown("---")
        st.markdown("### 📝 Instructions")
        st.markdown("""
        1. Upload a chest X-ray image
        2. View predictions from all models
        3. Compare model confidence scores
        4. Analyze model performance metrics
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Chest X-ray Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a chest X-ray image (PNG, JPG, or JPEG format)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            # Display image - compatible with all Streamlit versions
            try:
                st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
            except TypeError:
                # Fallback for older Streamlit versions
                st.image(image, caption="Uploaded X-ray Image")
            
            # Image info
            st.info(f"**Image Size:** {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        st.header("🔍 Prediction Results")
        
        if uploaded_file is not None:
            # Initialize device
            device = get_device()
            
            # Load models
            model_paths = {
                'Traditional CNN': Path('outputs/traditional_cnn_best.pth'),
                'ResNet50': Path('outputs/resnet50_finetuned.pth'),
                'EfficientNetB0': Path('outputs/efficientnet_finetuned.pth')
            }
            
            # Check if models exist
            missing_models = [name for name, path in model_paths.items() if not path.exists()]
            if missing_models:
                st.error(f"❌ Missing model files: {', '.join(missing_models)}")
                st.info("Please ensure all model files are in the 'outputs' directory.")
                return
            
            # Preprocess image
            image_tensor = preprocess_image(image)
            
            # Run predictions
            predictions = {}
            
            with st.spinner("🔄 Running predictions on all models..."):
                # Traditional CNN
                try:
                    cnn_model = load_traditional_cnn(model_paths['Traditional CNN'], device)
                    predictions['Traditional CNN'] = predict(cnn_model, image_tensor, device, 'Traditional CNN')
                except Exception as e:
                    st.error(f"Error loading Traditional CNN: {str(e)}")
                
                # ResNet50
                try:
                    resnet_model = load_resnet50(model_paths['ResNet50'], device)
                    predictions['ResNet50'] = predict(resnet_model, image_tensor, device, 'ResNet50')
                except Exception as e:
                    st.error(f"Error loading ResNet50: {str(e)}")
                
                # EfficientNetB0
                try:
                    eff_model = load_efficientnet(model_paths['EfficientNetB0'], device)
                    predictions['EfficientNetB0'] = predict(eff_model, image_tensor, device, 'EfficientNetB0')
                except Exception as e:
                    st.error(f"Error loading EfficientNetB0: {str(e)}")
            
            # Display predictions
            if predictions:
                # Consensus prediction
                class_votes = {}
                for pred in predictions.values():
                    class_name = pred['class_name']
                    if class_name not in class_votes:
                        class_votes[class_name] = 0
                    class_votes[class_name] += 1
                
                consensus_class = max(class_votes, key=class_votes.get)
                consensus_count = class_votes[consensus_class]
                
                st.success(f"🎯 **Consensus Prediction:** {consensus_class} ({consensus_count}/3 models agree)")
                st.markdown("---")
                
                # Individual model predictions
                for model_name, pred in predictions.items():
                    with st.container():
                        st.markdown(f'<div class="prediction-box">', unsafe_allow_html=True)
                        st.markdown(f'<p class="model-name">{model_name}</p>', unsafe_allow_html=True)
                        
                        # Prediction and confidence
                        col_pred, col_conf = st.columns(2)
                        with col_pred:
                            st.markdown(f"**Prediction:** {pred['class_name']}")
                        with col_conf:
                            conf_class = "confidence-high" if pred['confidence'] > 0.8 else "confidence-medium" if pred['confidence'] > 0.6 else "confidence-low"
                            st.markdown(f'<p class="{conf_class}">Confidence: {pred["confidence"]*100:.2f}%</p>', unsafe_allow_html=True)
                        
                        # Probability bars
                        prob_df = pd.DataFrame({
                            'Class': CLASS_NAMES,
                            'Probability': pred['probabilities']
                        })
                        
                        fig = px.bar(
                            prob_df,
                            x='Class',
                            y='Probability',
                            color='Class',
                            color_discrete_sequence=CLASS_COLORS,
                            range_y=[0, 1],
                            title=f"{model_name} - Class Probabilities"
                        )
                        fig.update_layout(showlegend=False, height=250)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.markdown("---")
        else:
            st.info("👆 Please upload an image to see predictions")
    
    # Analytics Section
    if uploaded_file is not None and predictions:
        st.markdown("---")
        st.header("📊 Model Analytics & Comparison")
        
        # Create tabs for different analytics
        tab1, tab2, tab3, tab4 = st.tabs([
            "📈 Confidence Comparison",
            "🎯 Prediction Agreement",
            "📊 Model Performance",
            "🔬 Detailed Analysis"
        ])
        
        with tab1:
            st.subheader("Confidence Scores Comparison")
            
            # Confidence comparison chart
            conf_data = {
                'Model': list(predictions.keys()),
                'Confidence': [p['confidence']*100 for p in predictions.values()],
                'Predicted Class': [p['class_name'] for p in predictions.values()]
            }
            conf_df = pd.DataFrame(conf_data)
            
            fig = px.bar(
                conf_df,
                x='Model',
                y='Confidence',
                color='Predicted Class',
                color_discrete_sequence=CLASS_COLORS,
                text='Confidence',
                title="Model Confidence Scores"
            )
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(yaxis_title="Confidence (%)", height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confidence statistics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Confidence", f"{np.mean([p['confidence']*100 for p in predictions.values()]):.2f}%")
            with col2:
                st.metric("Highest Confidence", f"{max([p['confidence']*100 for p in predictions.values()]):.2f}%")
            with col3:
                st.metric("Lowest Confidence", f"{min([p['confidence']*100 for p in predictions.values()]):.2f}%")
        
        with tab2:
            st.subheader("Model Agreement Analysis")
            
            # Agreement matrix
            agreement_data = []
            for i, (model1, pred1) in enumerate(predictions.items()):
                for j, (model2, pred2) in enumerate(predictions.items()):
                    if i < j:
                        agreement = pred1['class_name'] == pred2['class_name']
                        agreement_data.append({
                            'Model 1': model1,
                            'Model 2': model2,
                            'Agreement': 'Yes' if agreement else 'No',
                            'Class 1': pred1['class_name'],
                            'Class 2': pred2['class_name']
                        })
            
            agreement_df = pd.DataFrame(agreement_data)
            
            if not agreement_df.empty:
                # Agreement visualization
                fig = px.scatter(
                    agreement_df,
                    x='Model 1',
                    y='Model 2',
                    color='Agreement',
                    size=[1]*len(agreement_df),
                    title="Model Prediction Agreement Matrix",
                    color_discrete_map={'Yes': '#27ae60', 'No': '#e74c3c'}
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Agreement statistics
                total_pairs = len(agreement_df)
                agreeing_pairs = len(agreement_df[agreement_df['Agreement'] == 'Yes'])
                agreement_rate = (agreeing_pairs / total_pairs) * 100 if total_pairs > 0 else 0
                
                st.metric("Model Agreement Rate", f"{agreement_rate:.1f}%")
                
                # Display agreement details
                st.dataframe(agreement_df, use_container_width=True)
        
        with tab3:
            st.subheader("Historical Model Performance")
            
            # Load performance data
            perf_data = load_model_performance()
            
            # Performance metrics comparison
            metrics_data = []
            for model_name, metrics in perf_data['models'].items():
                metrics_data.append({
                    'Model': model_name,
                    'Accuracy': metrics['accuracy'] * 100,
                    'Precision': metrics['precision'] * 100,
                    'Recall': metrics['recall'] * 100,
                    'F1-Score': metrics['f1_score'] * 100
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            
            # Radar chart for performance metrics
            fig = go.Figure()
            
            for idx, row in metrics_df.iterrows():
                fig.add_trace(go.Scatterpolar(
                    r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']],
                    theta=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                    fill='toself',
                    name=row['Model']
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[90, 100]
                    )),
                showlegend=True,
                title="Model Performance Metrics (Radar Chart)",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Bar chart comparison
            metrics_melted = metrics_df.melt(
                id_vars=['Model'],
                value_vars=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                var_name='Metric',
                value_name='Score (%)'
            )
            
            fig2 = px.bar(
                metrics_melted,
                x='Model',
                y='Score (%)',
                color='Metric',
                barmode='group',
                title="Performance Metrics Comparison",
                height=400
            )
            st.plotly_chart(fig2, use_container_width=True)
            
            # Performance table
            st.dataframe(metrics_df, use_container_width=True)
        
        with tab4:
            st.subheader("Detailed Probability Analysis")
            
            # Probability heatmap
            prob_matrix = np.array([p['probabilities'] for p in predictions.values()])
            prob_df = pd.DataFrame(
                prob_matrix,
                index=list(predictions.keys()),
                columns=CLASS_NAMES
            )
            
            fig = px.imshow(
                prob_matrix,
                labels=dict(x="Class", y="Model", color="Probability"),
                x=CLASS_NAMES,
                y=list(predictions.keys()),
                color_continuous_scale='RdYlGn',
                aspect="auto",
                title="Probability Heatmap Across Models and Classes"
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detailed probability table
            st.subheader("Detailed Probabilities")
            prob_df_display = prob_df.copy()
            prob_df_display = prob_df_display.applymap(lambda x: f"{x*100:.2f}%")
            st.dataframe(prob_df_display, use_container_width=True)
            
            # Probability distribution
            st.subheader("Probability Distribution")
            for model_name, pred in predictions.items():
                fig = px.pie(
                    values=pred['probabilities'],
                    names=CLASS_NAMES,
                    title=f"{model_name} - Probability Distribution",
                    color_discrete_sequence=CLASS_COLORS
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
        <p><strong>Lung Disease Detection System</strong></p>
        <p>Medical Image Classification using Deep Learning</p>
        <p>⚠️ <em>This tool is for research purposes only. Not intended for clinical diagnosis.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

