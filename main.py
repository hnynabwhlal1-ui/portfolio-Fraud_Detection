import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go


# 1. Page Configuration
st.set_page_config(
    page_title="AI Credit Card Fraud Detection Portal",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Professional FinTech CSS Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .metric-box {
        background-color: white; padding: 20px; border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05); border-left: 5px solid #1f77b4;
    }
    .fraud-box { border-left: 5px solid #d62728; }
    </style>
""", unsafe_allow_html=True)

# 2. Dynamic Path Resolution for Models & Scalers
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
scaler_path = os.path.join(BASE_DIR, 'robust_scaler.pkl')
model_path = os.path.join(BASE_DIR, 'best_random_forest_model.pkl')


# 3. Secure Asset Loading using Cache
@st.cache_resource
def load_assets():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler


try:
    model, scaler = load_assets()
    assets_loaded = True
except Exception as e:
    assets_loaded = False
    error_message = str(e)

# 4. Sidebar Navigation
st.sidebar.title("🎮 Navigation")
page = st.sidebar.radio("Go to:", ["📖 Project Overview", "⚡ Real-Time Screening", "📊 Analytics & Insights"])

# --- SECTION 1: PROJECT OVERVIEW ---
if page == "📖 Project Overview":
    st.title("💳 AI-Powered Financial Security Portal")
    st.subheader("Interactive Machine Learning System for Credit Card Fraud Detection")

    st.markdown("""
    ### 📖 The Real-World Scenario: What Happens Behind the Screens?
    Imagine a customer named Sarah who is fast asleep at 3:00 AM. Suddenly, a fraudster in another country tries to use her compromised credit card info to buy a $1,500 luxury watch online. 

    Without Artificial Intelligence, this transaction might pass through undetected, causing financial loss and immense stress. 

    With our **AI Fraud Detection Model**, the moment the transaction is requested, the system analyzes 30 unique behavioral features in less than **50 milliseconds**.
    """)

    # Portfolio Metrics Boxes
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            '<div class="metric-box"><h3>Model Precision</h3><p style="font-size:32px; font-weight:bold; color:#2ca02c;">84%</p><p style="font-size:14px; color:gray;">Extremely low false alarm rate to protect genuine user experience.</p></div>',
            unsafe_allow_html=True)
    with col2:
        st.markdown(
            '<div class="metric-box"><h3>Model Recall</h3><p style="font-size:32px; font-weight:bold; color:#1f77b4;">78%</p><p style="font-size:14px; color:gray;">High capability of detecting and isolating true fraudulent activities.</p></div>',
            unsafe_allow_html=True)
    with col3:
        st.markdown(
            '<div class="metric-box fraud-box"><h3>Champion Model</h3><p style="font-size:28px; font-weight:bold; color:#d62728;">Random Forest</p><p style="font-size:14px; color:gray;">Optimized via Grid Search and robust imblearn pipeline logic.</p></div>',
            unsafe_allow_html=True)

# --- SECTION 2: REAL-TIME SCREENING ---
elif page == "⚡ Real-Time Screening":
    st.title("⚡ Real-Time Fraud Predictor Simulation")
    st.markdown("Simulate a new incoming transaction transaction attributes to test live ML model scoring.")

    if not assets_loaded:
        st.error(f"⚠️ Failed to load model or scaler. Error details: {error_message}")
    else:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("📥 Transaction Features")
            amount_input = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=25000.0, value=88.0,
                                           step=10.0)
            hour_input = st.slider("Transaction Hour (0-23)", min_value=0, max_value=23, value=12)

            st.markdown("**🎛️ Anonymized Behavioral Fingerprints (PCA Samples):**")
            v1 = st.slider("Behavioral Component V1", -5.0, 5.0, 0.0)
            v2 = st.slider("Behavioral Component V2", -5.0, 5.0, 0.0)
            v3 = st.slider("Behavioral Component V3", -5.0, 5.0, 0.0)
            v4 = st.slider("Behavioral Component V4", -5.0, 5.0, 0.0)

            # Reconstructing the 28 default dimensions
            features_dict = {f'V{i}': [0.0] for i in range(1, 29)}
            features_dict['V1'] = [v1]
            features_dict['V2'] = [v2]
            features_dict['V3'] = [v3]
            features_dict['V4'] = [v4]
            features_dict['Amount'] = [amount_input]
            features_dict['Hour'] = [hour_input]

            input_df = pd.DataFrame(features_dict)

            # Ensuring strict matching column order (V1-V28, Amount, Hour)
            cols_order = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour']
            input_df = input_df[cols_order]

            # Feature Scaling transformations matching original training data
            input_df['Amount'] = scaler.transform(input_df[['Amount']])

        with col2:
            st.subheader("🔮 Model Analytics & Decision")

            # Execution prediction probabilities
            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]
            fraud_prob = probabilities[1] * 100

            # Dynamic Interactive Gauge Chart
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=fraud_prob,
                title={'text': "Fraud Probability Index (%)", 'font': {'size': 20}},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#d62728" if fraud_prob > 50 else "#1f77b4"},
                    'steps': [
                        {'range': [0, 40], 'color': '#e8f5e9'},
                        {'range': [40, 70], 'color': '#fff3e0'},
                        {'range': [70, 100], 'color': '#ffebee'}
                    ],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'value': 70}
                }
            ))
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)

            # Rule conditional outputs matching calculated probability threshold triggers
            if prediction == 1 or fraud_prob >= 70:
                st.error(
                    "🚨 CRITICAL ALERT: High Probability of Fraudulent Pattern! Recommended Action: Deny transaction & trigger immediate security alert.")
            elif fraud_prob >= 40:
                st.warning(
                    "⚠️ SUSPICIOUS PROFILE: Elevated risk metrics detected. Recommended Action: Prompt user for Multi-Factor Authentication (MFA).")
            else:
                st.success("✅ TRANSACTION APPROVED: Normal behavioral signature confirmed.")

# --- SECTION 3: ANALYTICS & INSIGHTS ---
elif page == "📊 Analytics & Insights":
    st.title("📈 Behavioral Data Insights")
    st.markdown(
        "Interactive visualizations displaying demographic and financial variance between legitimate and malicious actions.")

    # Statistical baseline mocked evaluation distribution
    np.random.seed(42)
    mock_data = pd.DataFrame({
        'Amount': np.append(np.random.exponential(scale=50, size=500), np.random.uniform(500, 2000, size=50)),
        'Class': np.append(np.zeros(500), np.ones(50)),
        'Hour': np.append(np.random.normal(loc=14, scale=4, size=500).astype(int) % 24,
                          np.random.normal(loc=3, scale=2, size=50).astype(int) % 24)
    })
    mock_data['Status'] = mock_data['Class'].map({0: 'Normal', 1: 'Fraud'})

    col1, col2 = st.columns(2)
    with col1:
        fig1 = px.box(mock_data, x='Status', y='Amount', color='Status', log_y=True,
                      title="Financial Volume Profiles (Logarithmic Scale)",
                      color_discrete_map={'Normal': '#1f77b4', 'Fraud': '#d62728'})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.histogram(mock_data, x='Hour', color='Status', barmode='group',
                            title="Hourly Operational Distribution Profiles",
                            color_discrete_map={'Normal': '#1f77b4', 'Fraud': '#d62728'})
        st.plotly_chart(fig2, use_container_width=True)