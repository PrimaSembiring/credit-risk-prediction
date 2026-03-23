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
scaler, model, feature_names = load_scaler_and_model()

# ------- Membuat inputan user-------- #
st.header('Input data')
st.write("")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.subheader('Informasi Costumer')
    st.caption("Masukkan informasi dasar costumer")
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
    st.caption("-2 sampai 0 = tepat waktu, 1 sampai 9 = keterlambatan (bulan)")
    pay_0 = st.number_input('Bulan Ke-1', min_value=-2, max_value=9, value=0)
    pay_2 = st.number_input('Bulan Ke-2', min_value=-2, max_value=9, value=0)
    pay_3 = st.number_input('Bulan Ke-3', min_value=-2, max_value=9, value=0)
    pay_4 = st.number_input('Bulan Ke-4', min_value=-2, max_value=9, value=0)
    pay_5 = st.number_input('Bulan Ke-5', min_value=-2, max_value=9, value=0)
    pay_6 = st.number_input('Bulan Ke-6', min_value=-2, max_value=9, value=0)

with col3:
    st.subheader('Riwayat Tagihan Kartu Kredit')
    st.caption("Jumlah tagihan per bulan")
    bill_amt1 = st.number_input('Tagihan Ke-1', value=0, step=1000)
    bill_amt2 = st.number_input('Tagihan Ke-2', value=0, step=1000)
    bill_amt3 = st.number_input('Tagihan Ke-3', value=0, step=1000)
    bill_amt4 = st.number_input('Tagihan Ke-4', value=0, step=1000)
    bill_amt5 = st.number_input('Tagihan Ke-5', value=0, step=1000)
    bill_amt6 = st.number_input('Tagihan Ke-6', value=0, step=1000)

with col4:
    st.subheader('Riwayat Jumlah Pembayaran')
    st.caption('Jumlah pembayaran tiap bulan')
    pay_amt1 = st.number_input('Pembayaran Ke-1', value=0, step=1000)
    pay_amt2 = st.number_input('Pembayaran Ke-2', value=0, step=1000)
    pay_amt3 = st.number_input('Pembayaran Ke-3', value=0, step=1000)
    pay_amt4 = st.number_input('Pembayaran Ke-4', value=0, step=1000)
    pay_amt5 = st.number_input('Pembayaran Ke-5', value=0, step=1000)
    pay_amt6 = st.number_input('Pembayaran Ke-6', value=0, step=1000)

st.write("")
if st.button("Predict Default Risk", type='primary'):
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

    #konversi ke DataFrame
    input_df = pd.DataFrame([input_data])

    # membuat feature enginering sesuai dari notebook
    pay_cols = ['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']
    bill_cols = ['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']
    pay_amt_cols = ['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']

    input_df['PAY_MEAN'] = input_df[pay_cols].mean(axis=1)
    input_df['bill_total'] = input_df[bill_cols].sum(axis=1)
    input_df['pay_total'] = input_df[pay_amt_cols].sum(axis=1)
    input_df['PAY_RATIO'] = input_df['pay_total'] / (input_df['bill_total'] + 1)

    # memastikan urutan data sama persis dengan featur yang sudah dibuat
    input_df = input_df[feature_names]

    # mengambil hasil scaler yang sudah di training
    input_scaled = scaler.transform(input_df)

    # matikan perhitungan gradien biar kencang 
    with torch.no_grad():
        input_tensor = torch.FloatTensor(input_scaled) # mengubah data jadi tensor 
        prob = torch.sigmoid(model(input_tensor)).item()

    st.header('Hasil Prediksi')
    colpred1, colpred2 = st.columns(2)

    with colpred1:
        st.metric('Probalitas gagal barar', f'{prob:.3f}')
    
    # interpretasi hasil 
    with colpred2:
        if prob > 0.5:
            st.error('⚠️ Resiko besar ngasih utang ke bujank ini')
            st.write('Bunjank ini doyan ngutang jadi besar kemungkinan gak bakal di bayar, buntung kita')
        else:
            st.success('Bayar utang tepat waktu')
            st.write('Si kawan ini cuma kemungkinan kecil g bayar utang')
        
    # Info tambahan
    st.subheader("Resiko")
    if prob < 0.2:
        risk_level = "Sangat Kecil"
        color = "green"
    elif prob < 0.4:
        risk_level = "Kecil"
        color = "blue"
    elif prob < 0.6:
        risk_level = "Medium"
        color = "orange"
    elif prob < 0.8:
        risk_level = "Besar"
        color = "red"
    else:
        risk_level = "Sangat Besar"
        color = "darkred"

    st.markdown(f"<h3 style='color:{color};'>Risk Level: {risk_level}</h3>", unsafe_allow_html=True)
    
