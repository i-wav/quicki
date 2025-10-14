# app.py
import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier

# --- LOAD TRAINED MODEL ---
MODEL_PATH = "catboost_upi_fraud_model.cbm"  # your saved model
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# --- ESSENTIAL FEATURES ---
feature_columns = [
    'Sender_UPI_ID', 'Receiver_UPI_ID', 'Amount_INR', 'Transaction_Type',
    'Merchant_Category', 'Channel', 'Device_Type', 'Device_ID',
    'IP_Risk_Score', 'City', 'Sender_Age_Group', 'Sender_Bank', 'Receiver_Bank',
    'Account_Age_Days', 'Num_Txns_Last_24H', 'Avg_Amount_Last_7d',
    'Prev_Fraud_Count_Sender', 'Prev_Fraud_Count_Receiver',
    'Transaction_Note', 'Is_Night_Txn', 'Device_Change_Flag'
]

categorical_cols = [
    'Sender_UPI_ID', 'Receiver_UPI_ID', 'Transaction_Type', 'Merchant_Category',
    'Channel', 'Device_Type', 'Device_ID', 'City', 'Sender_Age_Group',
    'Sender_Bank', 'Receiver_Bank', 'Transaction_Note'
]

# --- STREAMLIT UI ---
st.title("💳 UPI Fraud Detection")
st.write("Enter transaction details to check for fraud:")

# Collect only essential inputs from user
sender_upi = st.text_input("Sender UPI ID", "user010055@okicici")
receiver_upi = st.text_input("Receiver UPI ID", "recv027483@okicici")
amount = st.number_input("Transaction Amount (INR)", 0.0, 10000000.0, 5000.0, step=100.0)
txn_type = st.selectbox("Transaction Type", ["P2P", "P2M", "M2P"])

# Predefined Merchant Categories
merchant_categories = [
    "Healthcare", "Recharge", "Food & Beverages", "Utilities", "Shopping",
    "Travel", "Education", "Entertainment", "Other"
]
merchant_cat = st.selectbox("Merchant Category", merchant_categories)

channel = st.selectbox("Channel", ["PhonePe", "Paytm", "GooglePay"])
device_type = st.selectbox("Device Type", ["Mobile", "Desktop"])
device_id = st.text_input("Device ID", "DEV815551")

# Button to trigger prediction
if st.button("Scan / Detect Fraud"):

    # Build new transaction with defaults
    new_txn = {
        'Sender_UPI_ID': sender_upi,
        'Receiver_UPI_ID': receiver_upi,
        'Amount_INR': amount,
        'Transaction_Type': txn_type,
        'Merchant_Category': merchant_cat,
        'Channel': channel,
        'Device_Type': device_type,
        'Device_ID': device_id,
        'IP_Risk_Score': 0.0,
        'City': 'Unknown',
        'Sender_Age_Group': '18-25',
        'Sender_Bank': 'Unknown',
        'Receiver_Bank': 'Unknown',
        'Account_Age_Days': 365,
        'Num_Txns_Last_24H': 0,
        'Avg_Amount_Last_7d': amount,
        'Prev_Fraud_Count_Sender': 0,
        'Prev_Fraud_Count_Receiver': 0,
        'Transaction_Note': 'payment',
        'Is_Night_Txn': 0,
        'Device_Change_Flag': 0
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([new_txn])

    # Ensure categorical columns are strings
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)

    # Reorder columns
    input_df = input_df[feature_columns]

    # Predict
    fraud_prob = model.predict_proba(input_df)[:,1][0]
    fraud_label = int(model.predict(input_df)[0])

    # Display results
    st.subheader("Prediction Results")
    st.write(f"**Fraud Probability:** {fraud_prob:.4f}")
    st.write(f"**Predicted Label:** {'Fraud' if fraud_label==1 else 'Legit'}")
