import streamlit as st
import pandas as pd
import datetime
from catboost import CatBoostClassifier

MODEL_PATH = "src/model/catboost_upi_fraud_model.cbm"  
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

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

st.set_page_config(page_title="UPI Fraud Detection", layout="centered")
st.title("UPI Fraud Detection System")
st.write("Enter transaction details to predict whether it's a potential fraud.")

sender_upi = st.text_input("Sender UPI ID", "user010055@okicici")
receiver_upi = st.text_input("Receiver UPI ID", "recv027483@okicici")
amount = st.number_input("Transaction Amount (INR)", 0.0, 10000000.0, 5000.0, step=100.0)
txn_type = st.selectbox("Transaction Type", ["P2P", "P2M", "M2P"])

merchant_categories = [
    "Healthcare", "Recharge", "Food & Beverages", "Utilities", "Shopping",
    "Travel", "Education", "Entertainment", "Other"
]
merchant_cat = st.selectbox("Merchant Category", merchant_categories)

channel = st.selectbox("Channel", ["PhonePe", "Paytm", "GooglePay"])
device_type = st.selectbox("Device Type", ["Mobile", "Desktop"])
device_id = st.text_input("Device ID", "DEV815551")

txn_date = st.date_input("Transaction Date", datetime.date.today())
txn_time = st.time_input("Transaction Time (24-hr)", datetime.datetime.now().time())
hour = txn_time.hour
is_night_txn = 1 if (hour >= 22 or hour < 6) else 0

sender_age = st.number_input("Sender Age (years)", 13, 100, 25, step=1)
if sender_age < 25:
    sender_age_group = "18-25"
elif sender_age < 35:
    sender_age_group = "25-35"
elif sender_age < 50:
    sender_age_group = "35-50"
else:
    sender_age_group = "50+"

sender_bank = st.selectbox("Sender Bank", ["HDFC", "ICICI", "SBI", "Axis", "Unknown"])
receiver_bank = st.selectbox("Receiver Bank", ["HDFC", "ICICI", "SBI", "Axis", "Unknown"])
city = st.text_input("City", "Unknown")

st.markdown("**Account age (choose calculation method):**")
use_creation_date = st.checkbox("Compute Account Age from Account Creation Date", value=True)

if use_creation_date:
    account_creation_date = st.date_input("Account Creation Date", datetime.date.today() - datetime.timedelta(days=365))
    account_age_days = (txn_date - account_creation_date).days
    if account_age_days < 0:
        st.warning("Account creation date is after transaction date. Using 0 days instead.")
        account_age_days = 0
    st.info(f"Computed Account_Age_Days = {account_age_days} days")
else:
    account_age_days = st.number_input("Account Age (days, manual)", 0, 10000, 365, step=1)

num_txns_last_24h = st.number_input("No. of Txns in last 24h", 0, 500, 2, step=1)
avg_amt_last_7d = st.number_input("Avg Amount (last 7 days)", 0.0, 1000000.0, amount, step=100.0)
prev_fraud_sender = st.number_input("Previous Fraud Count (Sender)", 0, 100, 0, step=1)
prev_fraud_receiver = st.number_input("Previous Fraud Count (Receiver)", 0, 100, 0, step=1)

device_change_flag_input = st.selectbox("Device changed recently?", ["No", "Yes"])
device_change_flag = 1 if device_change_flag_input == "Yes" else 0

transaction_note = st.text_input("Transaction Note", "payment")

if st.button(" Detect Fraud"):

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
        'City': city,
        'Sender_Age_Group': sender_age_group,
        'Sender_Bank': sender_bank,
        'Receiver_Bank': receiver_bank,
        'Account_Age_Days': int(account_age_days),
        'Num_Txns_Last_24H': int(num_txns_last_24h),
        'Avg_Amount_Last_7d': float(avg_amt_last_7d),
        'Prev_Fraud_Count_Sender': int(prev_fraud_sender),
        'Prev_Fraud_Count_Receiver': int(prev_fraud_receiver),
        'Transaction_Note': transaction_note,
        'Is_Night_Txn': int(is_night_txn),
        'Device_Change_Flag': int(device_change_flag)
    }

    input_df = pd.DataFrame([new_txn])
    for col in categorical_cols:
        input_df[col] = input_df[col].astype(str)
    input_df = input_df[feature_columns]

    fraud_prob = float(model.predict_proba(input_df)[:, 1][0])
    fraud_label = int(model.predict(input_df)[0])

    st.subheader("Prediction Results")
    st.write(f"**Fraud Probability:** {fraud_prob:.4f}")
    st.write(f"**Predicted Label:** {' Fraudulent' if fraud_label == 1 else ' Legitimate'}")

    if fraud_label == 1:
        st.error(" High likelihood of fraud detected! Please verify this transaction.")
    else:
        st.success(" Transaction appears legitimate.")

    st.caption(f"Transaction evaluated for {txn_date} at {txn_time.strftime('%H:%M')} hrs.")
