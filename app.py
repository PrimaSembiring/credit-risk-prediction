import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn 
import joblib
import os 

# page config
st.set_page_config(
    page_title = 'Credit Card Default Prediction',
    layout = 'wide'
)

# title
st.title('Prediksi Risiko Gagal Bayar Kartu Kredit')
st.markdown("----")

# load scaler dan buat model yang udah ada
@st.cache_resource # supaya 1x load aja, sisanya pakai cache
def load_scaler_and_model():
    scaler = joblib.load("scaler.pkl") # load scaler dari training

    # ini harus sesuai feature yang sudah di feature enginering 
    feature_names = [
    'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
    'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
    'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
    'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    'PAY_MEAN','bill_total','pay_total','PAY_RATIO'
]
    
    # Arsitektur model sesuai dgn di notebook
    class CreditModel(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.network = nn.Sequential(
                # Blok pertama
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256), # menormalisas agar model stabil dan cepat 
                nn.LeakyReLU(0.1), # utk pola non linear / gradienya tetap ada 
                nn.Dropout(0.4),

                # blok kedua
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.3),

                # blok ketiga
                nn.Linear(128, 64),
                nn.BatchNorm1d(64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.2),

                # blok keempat 
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.LeakyReLU(),

                # output layer
                nn.Linear(32, 1),
                # nn.Sigmoid()
            )
        
        def forward(self, x):
            return self.network(x).squeeze(1)
    
    input_dim = len(feature_names) # sesuai jumlah feature_name
    model = CreditModel(input_dim)

    # load hasil model dari hasil training
    load_state = torch.load('best_model.pth', map_location='cpu')
    model.load_state_dict(load_state, strict=True) # load harus sesuai 100%
    model.eval()

    return scaler, model, feature_names

# simpan hasil function ke variabel ini
scaler, model, feature_name = load_scaler_and_model()

# ------- Membuat inputan user-------- #
st.header('Input data')
st.write("")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader('Informasi Costumer')
    limit_bal = st.number_input('Credit Limit', min_value=10000, max_value=1000000, step=10000)
    sex = st.selectbox("Jenis kelamin", [1,2], format_func=lambda x: 'Male' if x == 1 else 'Female')
    education = st.selectbox('Pendidikan terakhir', [1,2,3,4], format_func=lambda x: {1:"Graduate School",
                                                                                      2:"University",
                                                                                      3:'High School',
                                                                                      4:'Others'}[x])
    marriage = st.selectbox('Status', [1,2,3], format_func=lambda x: {1:'Menikah',
                                                                      2:'Jomblo',
                                                                      3:'Lainnya'}[x])
    age = st.number_input('Usia', min_value=18, max_value=90, value=25)

with col2:
    st.subheader("Riwayat Pembayaran")
    pay_0 = st.number_input('Bulan Ke-1', min_value=-2, max_value=9, value=0)
    pay_2 = st.number_input('Bulan Ke-2', min_value=-2, max_value=9, value=0)
    pay_3 = st.number_input('Bulan Ke-3', min_value=-2, max_value=9, value=0)
    pay_4 = st.number_input('Bulan Ke-4', min_value=-2, max_value=9, value=0)
    pay_5 = st.number_input('Bulan Ke-5', min_value=-2, max_value=9, value=0)
    pay_6 = st.number_input('Bulan Ke-6', min_value=-2, max_value=9, value=0)

