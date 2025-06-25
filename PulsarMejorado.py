# Import packages and set the random seed
import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go

np.random.seed(874)

# Page configuration
st.set_page_config(
    page_title="Pulsar Prediction App",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-result {
        font-size: 1.1rem;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .pulsar-positive {
        background-color: #d4edda;
        border-left: 4px solid #28a745;
    }
    .pulsar-negative {
        background-color: #f8d7da;
        border-left: 4px solid #dc3545;
    }
    .sidebar-info {
        background-color: #e3f2fd;
        padding: 0.8rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the pulsar data"""
    return pd.read_csv('Pulsar.csv')

@st.cache_resource
def train_models(predictors, outcome_pulsar, outcome_outlier):
    """Train and cache the random forest models"""
    rf_pulsar = RandomForestClassifier(n_estimators=100, max_features=0.55, oob_score=True, random_state=874)
    rf_outlier = RandomForestClassifier(n_estimators=100, max_features=0.55, oob_score=True, random_state=874)
    
    rf_pulsar.fit(predictors, outcome_pulsar)
    rf_outlier.fit(predictors, outcome_outlier)
    
    return rf_pulsar, rf_outlier

# Load data
pulsar = load_data()

# Prepare data
predictors = pulsar[['integrated_profile_mean', 'integrated_profile_standard_deviation', 
                    'integrated_profile_excess_kurtosis', 'integrated_profile_skewness', 
                    'dmsnr_curve_mean', 'dmsnr_curve_standard_deviation', 
                    'dmsnr_curve_excess_kurtosis', 'dmsnr_curve_skewness']]

outcome_pulsar = np.ravel(pulsar[['outcome']])
outcome_outlier = np.ravel(pulsar[['outlier']])

# Train models
rf_pulsar, rf_outlier = train_models(predictors, outcome_pulsar, outcome_outlier)

# Main title
st.markdown('<h1 class="main-header">🌌 Pulsar Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">High Time Resolution Universe Survey - Machine Learning Application</p>', unsafe_allow_html=True)

# Create tabs for better organization
tab1, tab2, tab3, tab4 = st.tabs(["🔍 Prediction", "📊 Model Analysis", "📈 Data Exploration", "ℹ️ About"])

# Sidebar for parameters
with st.sidebar:
    st.markdown("### 🎛️ Pulsar Parameters")
    
    # Add info box
    st.markdown("""
    <div class="sidebar-info">
    <strong>💡 How to use:</strong><br>
    Adjust the sliders to set pulsar characteristics and get real-time predictions!
    </div>
    """, unsafe_allow_html=True)
    
    # Organize parameters in collapsible sections
    with st.expander("🌟 Integrated Profile Parameters", expanded=True):
        integrated_profile_mean = st.slider(
            'Mean', 5.82, 192.61, 111.08, 
            help="Average of the integrated pulse profile"
        )
        integrated_profile_standard_deviation = st.slider(
            'Standard Deviation', 24.78, 98.77, 46.55,
            help="Variability in the integrated pulse profile"
        )
        integrated_profile_excess_kurtosis = st.slider(
            'Excess Kurtosis', -1.87, 8.06, 0.48,
            help="Measure of tail heaviness in the distribution"
        )
        integrated_profile_skewness = st.slider(
            'Skewness', -1.78, 68.09, 1.77,
            help="Asymmetry of the distribution"
        )
    
    with st.expander("📡 DM-SNR Curve Parameters", expanded=True):
        dmsnr_curve_mean = st.slider(
            'DM-SNR Mean', 0.22, 223.38, 12.61,
            help="Average of the DM-SNR curve"
        )
        dmsnr_curve_standard_deviation = st.slider(
            'DM-SNR Standard Deviation', 7.38, 110.63, 26.33,
            help="Variability in the DM-SNR curve"
        )
        dmsnr_curve_excess_kurtosis = st.slider(
            'DM-SNR Excess Kurtosis', -3.13, 34.53, 8.30,
            help="Tail heaviness of DM-SNR distribution"
        )
        dmsnr_curve_skewness = st.slider(
            'DM-SNR Skewness', -1.97, 1190.99, 104.86,
            help="Asymmetry of DM-SNR distribution"
        )

# Create parameter dataframe
parameters = pd.DataFrame({
    'integrated_profile_mean': [integrated_profile_mean],
    'integrated_profile_standard_deviation': [integrated_profile_standard_deviation],
    'integrated_profile_excess_kurtosis': [integrated_profile_excess_kurtosis],
    'integrated_profile_skewness': [integrated_profile_skewness],
    'dmsnr_curve_mean': [dmsnr_curve_mean],
    'dmsnr_curve_standard_deviation': [dmsnr_curve_standard_deviation],
    'dmsnr_curve_excess_kurtosis': [dmsnr_curve_excess_kurtosis],
    'dmsnr_curve_skewness': [dmsnr_curve_skewness]
})

# Make predictions
prediction_pulsar = rf_pulsar.predict(parameters)[0]
prob_pulsar = rf_pulsar.predict_proba(parameters)[0]
prediction_outlier = rf_outlier.predict(parameters)[0]
prob_outlier = rf_outlier.predict_proba(parameters)[0]

# Tab 1: Prediction Results
with tab1:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌟 Pulsar Classification")
        
        # Show prediction with confidence
        confidence_pulsar = max(prob_pulsar) * 100
        
        if prediction_pulsar == 'pulsar':
            st.markdown(f"""
            <div class="prediction-result pulsar-positive">
            <strong>✅ PULSAR DETECTED!</strong><br>
            Confidence: {confidence_pulsar:.1f}%<br>
            This object shows characteristics consistent with a pulsar.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result pulsar-negative">
            <strong>❌ Non-Pulsar</strong><br>
            Confidence: {confidence_pulsar:.1f}%<br>
            This object does not appear to be a pulsar.
            </div>
            """, unsafe_allow_html=True)
        
        # Probability bar chart
        fig_pulsar = go.Figure(data=[
            go.Bar(x=['Non-Pulsar', 'Pulsar'], 
                   y=[prob_pulsar[0]*100, prob_pulsar[1]*100],
                   marker_color=['#ff7f7f', '#90EE90'])
        ])
        fig_pulsar.update_layout(
            title="Pulsar Prediction Probabilities",
            yaxis_title="Probability (%)",
            height=300
        )
        st.plotly_chart(fig_pulsar, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Outlier Detection")
        
        confidence_outlier = max(prob_outlier) * 100
        
        if prediction_outlier == 'multivariate outlier':
            st.markdown(f"""
            <div class="prediction-result pulsar-negative">
            <strong>⚠️ Multivariate Outlier</strong><br>
            Confidence: {confidence_outlier:.1f}%<br>
            This object has unusual characteristics.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result pulsar-positive">
            <strong>✅ Normal Object</strong><br>
            Confidence: {confidence_outlier:.1f}%<br>
            This object has typical characteristics.
            </div>
            """, unsafe_allow_html=True)
        
        # Probability bar chart for outlier
        fig_outlier = go.Figure(data=[
            go.Bar(x=['Normal', 'Outlier'], 
                   y=[prob_outlier[1]*100, prob_outlier[0]*100],
                   marker_color=['#90EE90', '#ff7f7f'])
        ])
        fig_outlier.update_layout(
            title="Outlier Detection Probabilities",
            yaxis_title="Probability (%)",
            height=300
        )
        st.plotly_chart(fig_outlier, use_container_width=True)

# Tab 2: Model Analysis
with tab2:
    st.subheader("📊 Model Performance & Feature Importance")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Pulsar Model")
        error_pulsar = 1 - rf_pulsar.oob_score_
        accuracy_pulsar = rf_pulsar.oob_score_
        
        st.metric("Model Accuracy", f"{accuracy_pulsar:.1%}")
        st.metric("Out-of-Bag Error", f"{error_pulsar:.1%}")
        
        # Feature importance chart
        feature_names = ['IP Mean', 'IP Std', 'IP Excess Kurt', 'IP Skew',
                        'DM Mean', 'DM Std', 'DM Excess Kurt', 'DM Skew']
        importances = rf_pulsar.feature_importances_
        
        fig_imp_pulsar = px.bar(
            x=importances, 
            y=feature_names,
            orientation='h',
            title="Feature Importance - Pulsar Model",
            color=importances,
            color_continuous_scale='viridis'
        )
        fig_imp_pulsar.update_layout(height=400)
        st.plotly_chart(fig_imp_pulsar, use_container_width=True)
    
    with col2:
        st.markdown("### Outlier Model")
        error_outlier = 1 - rf_outlier.oob_score_
        accuracy_outlier = rf_outlier.oob_score_
        
        st.metric("Model Accuracy", f"{accuracy_outlier:.1%}")
        st.metric("Out-of-Bag Error", f"{error_outlier:.1%}")
        
        # Feature importance chart
        importances_outlier = rf_outlier.feature_importances_
        
        fig_imp_outlier = px.bar(
            x=importances_outlier, 
            y=feature_names,
            orientation='h',
            title="Feature Importance - Outlier Model",
            color=importances_outlier,
            color_continuous_scale='plasma'
        )
        fig_imp_outlier.update_layout(height=400)
        st.plotly_chart(fig_imp_outlier, use_container_width=True)

# Tab 3: Data Exploration
with tab3:
    st.subheader("📈 Dataset Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Samples", len(pulsar))
    with col2:
        pulsars_count = len(pulsar[pulsar['outcome'] == 'pulsar'])
        st.metric("Pulsars", pulsars_count)
    with col3:
        non_pulsars_count = len(pulsar[pulsar['outcome'] == 'non-pulsar'])
        st.metric("Non-Pulsars", non_pulsars_count)
    
    # Distribution plots
    st.subheader("Parameter Distributions")
    
    selected_feature = st.selectbox(
        "Select a feature to explore:",
        options=list(predictors.columns),
        format_func=lambda x: x.replace('_', ' ').title()
    )
    
    fig_dist = px.histogram(
        pulsar, 
        x=selected_feature, 
        color='outcome',
        title=f"Distribution of {selected_feature.replace('_', ' ').title()}",
        marginal="box"
    )
    st.plotly_chart(fig_dist, use_container_width=True)

# Tab 4: About
with tab4:
    st.subheader("ℹ️ About This Application")
    
    st.markdown("""
    ### 🌌 What are Pulsars?
    Pulsars are highly magnetized rotating neutron stars that emit beams of electromagnetic radiation. 
    They appear to pulse as the star rotates and the beam sweeps across Earth.
    
    ### 🔬 The Dataset
    This application uses data from the **High Time Resolution Universe Survey (HTRU2)**, 
    which contains pulsar candidates collected during the initial survey.
    
    ### 🤖 Machine Learning Approach
    - **Algorithm**: Random Forest Classifier
    - **Features**: 8 statistical measures from integrated pulse profiles and DM-SNR curves
    - **Dual Classification**: Pulsar detection + Outlier identification
    
    ### 📊 Features Explained
    - **Integrated Profile**: Statistical measures of the pulse profile
    - **DM-SNR Curve**: Dispersion measure and signal-to-noise ratio characteristics
    - **Statistical Measures**: Mean, Standard Deviation, Skewness, Excess Kurtosis
    
    ### 🎯 Model Performance
    The Random Forest models achieve high accuracy in distinguishing between pulsars and 
    non-pulsar objects, while also identifying multivariate outliers in the data.
    """)
    
    st.markdown("---")
    st.markdown("*Application developed for astrophysics data mining using Python and Streamlit*")

# Add a footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #666; font-size: 0.8rem;'>"
    "🌟 Pulsar Prediction System | High Time Resolution Universe Survey"
    "</div>", 
    unsafe_allow_html=True
)
