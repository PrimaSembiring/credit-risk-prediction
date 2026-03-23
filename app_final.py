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
def load_scaler():
    scaler = joblib.load("scaler.pkl") # ambil scaler dari training

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


