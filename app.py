import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from model_trainer import DiabetesModelTrainer
import warnings
warnings.filterwarnings('ignore')

# Medical-themed color scheme
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'success': '#06A77D',
    'warning': '#F18F01',
    'background': '#F8F9FA',
    'text': '#2C3E50'
}

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Prediction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for medical-grade interface
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Source+Sans+Pro:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Inter', 'Source Sans Pro', sans-serif;
        color: {COLORS['text']};
    }}
    
    .main {{
        background-color: {COLORS['background']};
    }}
    
    .stButton>button {{
        background-color: {COLORS['primary']};
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 12px 24px;
        border: none;
        width: 100%;
        transition: all 0.3s;
    }}
    
    .stButton>button:hover {{
        background-color: {COLORS['secondary']};
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .metric-card {{
        background-color: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 10px 0;
        border-left: 4px solid {COLORS['primary']};
    }}
    
    .prediction-card {{
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        margin: 15px 0;
    }}
    
    .success-box {{
        background-color: {COLORS['success']}20;
        border-left: 4px solid {COLORS['success']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    
    .warning-box {{
        background-color: {COLORS['warning']}20;
        border-left: 4px solid {COLORS['warning']};
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }}
    
    h1 {{
        color: {COLORS['primary']};
        font-weight: 700;
        margin-bottom: 10px;
    }}
    
    h2 {{
        color: {COLORS['secondary']};
        font-weight: 600;
        margin-top: 20px;
    }}
    
    h3 {{
        color: {COLORS['primary']};
        font-weight: 600;
    }}
    
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
    }}
    
    .stTabs [data-baseweb="tab"] {{
        background-color: white;
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }}
    
    .stTabs [aria-selected="true"] {{
        background-color: {COLORS['primary']};
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'trainer' not in st.session_state:
    with st.spinner('Loading and training models...'):
        trainer = DiabetesModelTrainer('attached_assets/diabetes_dataset_1761929795067.csv')
        trainer.train_all_models()
        st.session_state.trainer = trainer
        st.session_state.training_complete = True

trainer = st.session_state.trainer

# Header
st.title("🏥 Diabetes Risk Prediction System")
st.markdown(f"<p style='font-size: 18px; color: {COLORS['text']}; margin-bottom: 30px;'>Advanced Machine Learning Models for Healthcare Analytics</p>", unsafe_allow_html=True)

# Sidebar for navigation
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=80)
    st.title("Navigation")
    page = st.radio("Select Page", ["Patient Prediction", "Model Comparison", "About Models"])
    
    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    st.metric("Training Samples", len(trainer.X_train))
    st.metric("Test Samples", len(trainer.X_test))
    st.metric("Features", 5)

# PAGE 1: Patient Prediction
if page == "Patient Prediction":
    st.header("Patient Diabetes Risk Assessment")
    st.markdown("Enter patient health data to receive real-time predictions from both ML models")
    
    # Input form
    st.markdown("### Patient Information")
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=45, step=1)
        bmi = st.number_input("BMI (Body Mass Index)", min_value=15.0, max_value=60.0, value=25.0, step=0.1)
        glucose = st.number_input("Glucose Level (mg/dL)", min_value=50, max_value=250, value=120, step=1)
    
    with col2:
        blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=40, max_value=150, value=80, step=1)
        family_history = st.selectbox("Family History of Diabetes", ["No", "Yes"])
        st.markdown("")
        predict_button = st.button("🔍 Predict Diabetes Risk", use_container_width=True)
    
    if predict_button:
        with st.spinner('Analyzing patient data...'):
            predictions = trainer.predict_patient(age, bmi, glucose, blood_pressure, family_history)
            
            st.markdown("---")
            st.markdown("## 🎯 Prediction Results")
            
            # Side-by-side model predictions
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: {COLORS['primary']}; margin-top: 0;'>📈 Logistic Regression Model</h3>", unsafe_allow_html=True)
                
                lr_result = predictions['logistic_regression']
                lr_pred = lr_result['prediction']
                lr_prob = lr_result['probability_positive']
                
                if lr_pred == 1:
                    st.markdown(f"""
                    <div class='warning-box'>
                        <h2 style='color: {COLORS['warning']}; margin: 0;'>⚠️ POSITIVE</h2>
                        <p style='font-size: 16px; margin: 10px 0 0 0;'>High risk of diabetes detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='color: {COLORS['success']}; margin: 0;'>✅ NEGATIVE</h2>
                        <p style='font-size: 16px; margin: 10px 0 0 0;'>Low risk of diabetes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Confidence Score")
                st.progress(lr_prob / 100)
                st.markdown(f"**Diabetes Risk: {lr_prob:.1f}%**")
                st.markdown(f"**No Diabetes: {lr_result['probability_negative']:.1f}%**")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown(f"<h3 style='color: {COLORS['secondary']}; margin-top: 0;'>🌳 Decision Tree Model</h3>", unsafe_allow_html=True)
                
                dt_result = predictions['decision_tree']
                dt_pred = dt_result['prediction']
                dt_prob = dt_result['probability_positive']
                
                if dt_pred == 1:
                    st.markdown(f"""
                    <div class='warning-box'>
                        <h2 style='color: {COLORS['warning']}; margin: 0;'>⚠️ POSITIVE</h2>
                        <p style='font-size: 16px; margin: 10px 0 0 0;'>High risk of diabetes detected</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class='success-box'>
                        <h2 style='color: {COLORS['success']}; margin: 0;'>✅ NEGATIVE</h2>
                        <p style='font-size: 16px; margin: 10px 0 0 0;'>Low risk of diabetes</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("#### Confidence Score")
                st.progress(dt_prob / 100)
                st.markdown(f"**Diabetes Risk: {dt_prob:.1f}%**")
                st.markdown(f"**No Diabetes: {dt_result['probability_negative']:.1f}%**")
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Model Agreement Analysis
            st.markdown("---")
            st.markdown("### 🤝 Model Agreement Analysis")
            
            if lr_pred == dt_pred:
                st.markdown(f"""
                <div class='success-box'>
                    <h4 style='margin: 0; color: {COLORS['success']};'>✅ Both Models Agree</h4>
                    <p style='margin: 10px 0 0 0;'>Both Logistic Regression and Decision Tree models predict the same outcome, indicating higher confidence in the result.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class='warning-box'>
                    <h4 style='margin: 0; color: {COLORS['warning']};'>⚠️ Models Disagree</h4>
                    <p style='margin: 10px 0 0 0;'>The models have different predictions. Consider reviewing the confidence scores and consulting with a healthcare professional for further evaluation.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Clinical Recommendation
            st.markdown("---")
            st.markdown("### 🩺 Clinical Recommendation")
            avg_risk = (lr_prob + dt_prob) / 2
            
            if avg_risk > 70:
                st.error(f"**High Risk ({avg_risk:.1f}%)**: Immediate medical consultation recommended. Schedule appointment with endocrinologist.")
            elif avg_risk > 40:
                st.warning(f"**Moderate Risk ({avg_risk:.1f}%)**: Lifestyle modifications recommended. Regular monitoring advised.")
            else:
                st.success(f"**Low Risk ({avg_risk:.1f}%)**: Continue healthy lifestyle. Annual screening recommended.")

# PAGE 2: Model Comparison
elif page == "Model Comparison":
    st.header("Model Performance Comparison")
    st.markdown("Comprehensive comparison of Logistic Regression vs Decision Tree models")
    
    # Performance metrics comparison
    st.markdown("### 📊 Model Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    lr_metrics = trainer.lr_metrics
    dt_metrics = trainer.dt_metrics
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy", 
                  f"{lr_metrics['accuracy']*100:.2f}%", 
                  f"{(lr_metrics['accuracy'] - dt_metrics['accuracy'])*100:.2f}%",
                  help="Logistic Regression")
        st.caption("Logistic Regression")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Accuracy", 
                  f"{dt_metrics['accuracy']*100:.2f}%",
                  help="Decision Tree")
        st.caption("Decision Tree")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Precision", 
                  f"{lr_metrics['precision']*100:.2f}%",
                  f"{(lr_metrics['precision'] - dt_metrics['precision'])*100:.2f}%")
        st.caption("Logistic Regression")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Precision", 
                  f"{dt_metrics['precision']*100:.2f}%")
        st.caption("Decision Tree")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Recall", 
                  f"{lr_metrics['recall']*100:.2f}%",
                  f"{(lr_metrics['recall'] - dt_metrics['recall'])*100:.2f}%")
        st.caption("Logistic Regression")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Recall", 
                  f"{dt_metrics['recall']*100:.2f}%")
        st.caption("Decision Tree")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("F1-Score", 
                  f"{lr_metrics['f1']*100:.2f}%",
                  f"{(lr_metrics['f1'] - dt_metrics['f1'])*100:.2f}%")
        st.caption("Logistic Regression")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("F1-Score", 
                  f"{dt_metrics['f1']*100:.2f}%")
        st.caption("Decision Tree")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Confusion Matrices
    st.markdown("---")
    st.markdown("### 🔢 Confusion Matrices")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Logistic Regression")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(lr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'],
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    with col2:
        st.markdown("#### Decision Tree")
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(dt_metrics['confusion_matrix'], annot=True, fmt='d', cmap='RdPu',
                    xticklabels=['No Diabetes', 'Diabetes'],
                    yticklabels=['No Diabetes', 'Diabetes'],
                    cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # ROC Curves
    st.markdown("---")
    st.markdown("### 📈 ROC Curves Comparison")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Logistic Regression ROC
    fpr_lr, tpr_lr, _ = lr_metrics['roc_curve']
    ax.plot(fpr_lr, tpr_lr, color=COLORS['primary'], linewidth=2.5, 
            label=f'Logistic Regression (AUC = {lr_metrics["auc"]:.3f})')
    
    # Decision Tree ROC
    fpr_dt, tpr_dt, _ = dt_metrics['roc_curve']
    ax.plot(fpr_dt, tpr_dt, color=COLORS['secondary'], linewidth=2.5,
            label=f'Decision Tree (AUC = {dt_metrics["auc"]:.3f})')
    
    # Diagonal line
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')
    
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold', color=COLORS['text'])
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
    
    # Cross-Validation Scores
    st.markdown("---")
    st.markdown("### 🔄 Cross-Validation Performance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### Logistic Regression")
        if 'cv_score' in lr_metrics:
            st.metric("CV Accuracy (Mean)", f"{lr_metrics['cv_score']*100:.2f}%")
            if 'best_params' in lr_metrics and lr_metrics['best_params']:
                st.markdown("**Best Parameters:**")
                for param, value in lr_metrics['best_params'].items():
                    st.text(f"• {param}: {value}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.markdown("#### Decision Tree")
        if 'cv_score' in dt_metrics:
            st.metric("CV Accuracy (Mean)", f"{dt_metrics['cv_score']*100:.2f}%")
            if 'best_params' in dt_metrics and dt_metrics['best_params']:
                st.markdown("**Best Parameters:**")
                for param, value in dt_metrics['best_params'].items():
                    st.text(f"• {param}: {value}")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Feature Importance
    st.markdown("---")
    st.markdown("### 🎯 Feature Importance (Decision Tree)")
    
    feature_importance = trainer.get_feature_importance()
    if feature_importance:
        fig, ax = plt.subplots(figsize=(10, 5))
        features = list(feature_importance.keys())
        importances = list(feature_importance.values())
        
        colors_bars = [COLORS['primary'], COLORS['secondary'], COLORS['success'], 
                      COLORS['warning'], COLORS['text']]
        ax.barh(features, importances, color=colors_bars)
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance in Decision Tree Model', fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    
    # Model Recommendations
    st.markdown("---")
    st.markdown("### 💡 Model Selection Recommendation")
    
    if lr_metrics['accuracy'] > dt_metrics['accuracy']:
        better_model = "Logistic Regression"
        better_acc = lr_metrics['accuracy']
    else:
        better_model = "Decision Tree"
        better_acc = dt_metrics['accuracy']
    
    st.info(f"""
    **Recommended Model: {better_model}**
    
    Based on the performance metrics:
    - **Accuracy**: {better_acc*100:.2f}%
    - This model shows better overall performance on the test dataset
    - However, using both models together provides more comprehensive diagnosis
    """)

# PAGE 3: About Models
else:
    st.header("About the Machine Learning Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='prediction-card'>
        <h3 style='color: {COLORS['primary']};'>📈 Logistic Regression</h3>
        
        **Type**: Linear Classification Model
        
        **How it works**:
        - Uses sigmoid function to predict probability
        - Linear decision boundary
        - Outputs probability scores between 0 and 1
        
        **Strengths**:
        - Fast training and prediction
        - Interpretable coefficients
        - Works well with linearly separable data
        - Provides probability estimates
        
        **Use Cases**:
        - Binary classification problems
        - When interpretability is important
        - Large datasets with many features
        
        **In Healthcare**:
        - Risk assessment and screening
        - Disease diagnosis
        - Patient outcome prediction
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='prediction-card'>
        <h3 style='color: {COLORS['secondary']};'>🌳 Decision Tree</h3>
        
        **Type**: Non-linear Classification Model
        
        **How it works**:
        - Creates tree-like model of decisions
        - Splits data based on feature values
        - Non-linear decision boundaries
        
        **Strengths**:
        - Handles non-linear relationships
        - Easy to visualize and understand
        - No feature scaling required
        - Provides feature importance
        
        **Use Cases**:
        - Complex decision-making scenarios
        - When feature interactions matter
        - Mixed data types (numerical & categorical)
        
        **In Healthcare**:
        - Diagnostic decision support
        - Treatment pathway selection
        - Patient stratification
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 🔬 Model Optimization Techniques")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
        <h4 style='color: {COLORS['primary']};'>Cross-Validation</h4>
        
        **What it is**:
        - 5-fold cross-validation technique
        - Divides training data into 5 equal parts
        - Trains model 5 times, each time using different part for validation
        
        **Purpose**:
        - Prevents overfitting
        - Provides robust performance estimate
        - Ensures model generalizes well to unseen data
        
        **Benefits**:
        - More reliable accuracy estimates
        - Better model selection
        - Reduces variance in performance metrics
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
        <h4 style='color: {COLORS['secondary']};'>Hyperparameter Tuning</h4>
        
        **What it is**:
        - Grid Search optimization
        - Systematically tests different parameter combinations
        - Finds best configuration for each model
        
        **Parameters Tuned**:
        - **Logistic Regression**: C (regularization), solver
        - **Decision Tree**: max_depth, min_samples_split, criterion
        
        **Benefits**:
        - Optimized model performance
        - Automatic parameter selection
        - Improved accuracy and generalization
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📋 Dataset Information")
    
    st.markdown(f"""
    <div class='metric-card'>
    
    **Features Used for Prediction**:
    1. **Age**: Patient age in years (range: 18-100)
    2. **BMI**: Body Mass Index in kg/m² (range: 15-60)
    3. **Glucose**: Blood glucose level in mg/dL (range: 50-250)
    4. **Blood Pressure**: Systolic blood pressure in mmHg (range: 40-150)
    5. **Family History**: Genetic predisposition to diabetes (Yes/No)
    
    **Target Variable**: Diabetic (0 = No Diabetes, 1 = Has Diabetes)
    
    **Data Preprocessing**:
    - Missing values filled with median values for numerical features
    - Feature scaling applied using StandardScaler for Logistic Regression
    - Family history converted to binary encoding (Yes=1, No=0)
    - Train-test split: 80% training (160 samples), 20% testing (40 samples)
    - Stratified sampling to maintain class balance
    
    **Dataset Statistics**:
    - Total samples: {len(trainer.df)} patients
    - Training samples: {len(trainer.X_train)}
    - Testing samples: {len(trainer.X_test)}
    - Features: 5 predictive features
    
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### 📊 Understanding Model Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        <div class='metric-card'>
        <h4 style='color: {COLORS['primary']};'>Performance Metrics</h4>
        
        **Accuracy**: Percentage of correct predictions
        - Formula: (TP + TN) / (TP + TN + FP + FN)
        - Higher is better (0-100%)
        
        **Precision**: Of predicted diabetics, how many actually have diabetes
        - Formula: TP / (TP + FP)
        - Measures false positive rate
        
        **Recall (Sensitivity)**: Of actual diabetics, how many were identified
        - Formula: TP / (TP + FN)
        - Measures false negative rate
        
        **F1-Score**: Harmonic mean of precision and recall
        - Formula: 2 × (Precision × Recall) / (Precision + Recall)
        - Balances precision and recall
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class='metric-card'>
        <h4 style='color: {COLORS['secondary']};'>Advanced Metrics</h4>
        
        **AUC (Area Under Curve)**: Measures overall model performance
        - Range: 0.5 to 1.0
        - 0.5 = Random guessing
        - 1.0 = Perfect classification
        - Higher values indicate better discrimination
        
        **Confusion Matrix**: Shows prediction breakdown
        - True Positives (TP): Correctly identified diabetics
        - True Negatives (TN): Correctly identified non-diabetics
        - False Positives (FP): Incorrectly predicted as diabetic
        - False Negatives (FN): Missed diabetic cases
        
        **ROC Curve**: Visualizes trade-off between true positive and false positive rates
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### ⚕️ Clinical Disclaimer")
    st.warning("""
    **Important**: This tool is for educational and screening purposes only. 
    It should NOT replace professional medical diagnosis or treatment. 
    Always consult with qualified healthcare professionals for medical advice, 
    diagnosis, and treatment decisions.
    """)
