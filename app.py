import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Set page config
st.set_page_config(
    page_title="Credit Card Default Prediction",
    page_icon="💳",
    layout="wide"
)

# Title
st.title("💳 Credit Card Default Prediction")
st.markdown("---")

# Load data and create scaler
@st.cache_resource
def load_scaler_and_model():
    # Load data
    df = pd.read_csv('default of credit card clients.csv')
    df.rename(columns={'default payment next month': 'DEFAULT'}, inplace=True)
    df.drop(columns=['ID'], inplace=True)

    # Feature Engineering
    df_fe = df.copy()
    for i in range(1, 7):
        df_fe[f'UTIL_{i}'] = df_fe[f'BILL_AMT{i}'] / (df_fe['LIMIT_BAL'] + 1)
    for i in range(1, 7):
        df_fe[f'PAY_RATIO_{i}'] = df_fe[f'PAY_AMT{i}'] / (df_fe[f'BILL_AMT{i}'].abs() + 1)
    pay_status_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    df_fe['TOTAL_DELAY_COUNT'] = (df_fe[pay_status_cols] > 0).sum(axis=1)
    df_fe['MAX_DELAY'] = df_fe[pay_status_cols].max(axis=1)
    df_fe['AVG_DELAY'] = df_fe[pay_status_cols].mean(axis=1)
    df_fe['BILL_TREND'] = df_fe['BILL_AMT1'] - df_fe['BILL_AMT6']
    df_fe['PAY_AMT_TREND'] = df_fe['PAY_AMT1'] - df_fe['PAY_AMT6']
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    df_fe['AVG_BILL'] = df_fe[bill_cols].mean(axis=1)
    df_fe['AVG_PAY_AMT'] = df_fe[pay_amt_cols].mean(axis=1)

    X = df_fe.drop(columns=['DEFAULT']).iloc[:, :27]  # Match saved model features

    # Scaler
    scaler = StandardScaler()
    scaler.fit(X)

    # Model
    class CreditDefaultNet(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(32, 1),
                nn.Sigmoid()
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.network(x).squeeze(1)

    input_dim = 27  # Match saved model
    model = CreditDefaultNet(input_dim)

    # Load weights if exists
    if os.path.exists('best_model.pth'):
        state_dict = torch.load('best_model.pth', map_location=torch.device('cpu'))
        model.load_state_dict(state_dict, strict=False)
        st.success("Model loaded successfully!")
    else:
        st.error("Model file 'best_model.pth' not found!")

    model.eval()

    return scaler, model, X.columns.tolist()

scaler, model, feature_names = load_scaler_and_model()

# Input form
st.header("📝 Input Customer Data")

col1, col2, col3 = st.columns(3)

with col1:
    limit_bal = st.number_input("Limit Balance (NT$)", min_value=10000, max_value=1000000, value=50000, step=10000)
    sex = st.selectbox("Gender", [1, 2], format_func=lambda x: "Male" if x == 1 else "Female")
    education = st.selectbox("Education", [1, 2, 3, 4], format_func=lambda x: {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"}[x])
    marriage = st.selectbox("Marital Status", [1, 2, 3], format_func=lambda x: {1: "Married", 2: "Single", 3: "Others"}[x])
    age = st.number_input("Age", min_value=18, max_value=100, value=30)

with col2:
    st.subheader("Payment Status (Last 6 months)")
    pay_0 = st.number_input("PAY_0 (Sep)", min_value=-2, max_value=9, value=0)
    pay_2 = st.number_input("PAY_2 (Aug)", min_value=-2, max_value=9, value=0)
    pay_3 = st.number_input("PAY_3 (Jul)", min_value=-2, max_value=9, value=0)
    pay_4 = st.number_input("PAY_4 (Jun)", min_value=-2, max_value=9, value=0)
    pay_5 = st.number_input("PAY_5 (May)", min_value=-2, max_value=9, value=0)
    pay_6 = st.number_input("PAY_6 (Apr)", min_value=-2, max_value=9, value=0)

with col3:
    st.subheader("Bill Amounts (NT$)")
    bill_amt1 = st.number_input("BILL_AMT1 (Sep)", value=0)
    bill_amt2 = st.number_input("BILL_AMT2 (Aug)", value=0)
    bill_amt3 = st.number_input("BILL_AMT3 (Jul)", value=0)
    bill_amt4 = st.number_input("BILL_AMT4 (Jun)", value=0)
    bill_amt5 = st.number_input("BILL_AMT5 (May)", value=0)
    bill_amt6 = st.number_input("BILL_AMT6 (Apr)", value=0)

st.header("💰 Payment Amounts (NT$)")
col4, col5, col6 = st.columns(3)

with col4:
    pay_amt1 = st.number_input("PAY_AMT1 (Sep)", value=0)
with col5:
    pay_amt2 = st.number_input("PAY_AMT2 (Aug)", value=0)
with col6:
    pay_amt3 = st.number_input("PAY_AMT3 (Jul)", value=0)

col7, col8, col9 = st.columns(3)
with col7:
    pay_amt4 = st.number_input("PAY_AMT4 (Jun)", value=0)
with col8:
    pay_amt5 = st.number_input("PAY_AMT5 (May)", value=0)
with col9:
    pay_amt6 = st.number_input("PAY_AMT6 (Apr)", value=0)

# Predict button
if st.button("🔮 Predict Default Risk", type="primary"):
    # Create input data
    input_data = {
        'LIMIT_BAL': limit_bal,
        'SEX': sex,
        'EDUCATION': education,
        'MARRIAGE': marriage,
        'AGE': age,
        'PAY_0': pay_0,
        'PAY_2': pay_2,
        'PAY_3': pay_3,
        'PAY_4': pay_4,
        'PAY_5': pay_5,
        'PAY_6': pay_6,
        'BILL_AMT1': bill_amt1,
        'BILL_AMT2': bill_amt2,
        'BILL_AMT3': bill_amt3,
        'BILL_AMT4': bill_amt4,
        'BILL_AMT5': bill_amt5,
        'BILL_AMT6': bill_amt6,
        'PAY_AMT1': pay_amt1,
        'PAY_AMT2': pay_amt2,
        'PAY_AMT3': pay_amt3,
        'PAY_AMT4': pay_amt4,
        'PAY_AMT5': pay_amt5,
        'PAY_AMT6': pay_amt6
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Feature Engineering
    for i in range(1, 7):
        input_df[f'UTIL_{i}'] = input_df[f'BILL_AMT{i}'] / (input_df['LIMIT_BAL'] + 1)
    for i in range(1, 7):
        input_df[f'PAY_RATIO_{i}'] = input_df[f'PAY_AMT{i}'] / (input_df[f'BILL_AMT{i}'].abs() + 1)
    pay_status_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    input_df['TOTAL_DELAY_COUNT'] = (input_df[pay_status_cols] > 0).sum(axis=1)
    input_df['MAX_DELAY'] = input_df[pay_status_cols].max(axis=1)
    input_df['AVG_DELAY'] = input_df[pay_status_cols].mean(axis=1)
    input_df['BILL_TREND'] = input_df['BILL_AMT1'] - input_df['BILL_AMT6']
    input_df['PAY_AMT_TREND'] = input_df['PAY_AMT1'] - input_df['PAY_AMT6']
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']
    input_df['AVG_BILL'] = input_df[bill_cols].mean(axis=1)
    input_df['AVG_PAY_AMT'] = input_df[pay_amt_cols].mean(axis=1)

    # Ensure order matches training
    input_df = input_df[feature_names[:27]]  # Match 27 features

    # Scale
    input_scaled = scaler.transform(input_df)

    # Predict
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled)
        prob = model(input_tensor).item()

    # Display result
    st.header("🎯 Prediction Result")
    col_result1, col_result2 = st.columns(2)

    with col_result1:
        st.metric("Default Probability", f"{prob:.3f}")

    with col_result2:
        if prob > 0.5:
            st.error("⚠️ High Risk of Default")
            st.write("This customer has a high probability of defaulting on their credit card payment.")
        else:
            st.success("✅ Low Risk of Default")
            st.write("This customer has a low probability of defaulting on their credit card payment.")

    # Additional info
    st.markdown("---")
    st.subheader("📊 Risk Assessment")
    if prob < 0.2:
        risk_level = "Very Low"
        color = "green"
    elif prob < 0.4:
        risk_level = "Low"
        color = "blue"
    elif prob < 0.6:
        risk_level = "Medium"
        color = "orange"
    elif prob < 0.8:
        risk_level = "High"
        color = "red"
    else:
        risk_level = "Very High"
        color = "darkred"

    st.markdown(f"<h3 style='color:{color};'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)

    # Show input summary            
    st.subheader("📋 Input Summary")
    summary_df = pd.DataFrame(list(input_data.items()), columns=['Feature', 'Value'])
    st.dataframe(summary_df, use_container_width=True)

st.markdown("---")
st.caption("Built with PyTorch and Streamlit | Dataset: UCI Credit Card Clients")