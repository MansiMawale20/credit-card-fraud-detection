import streamlit as st
import pandas as pd
from sklearn.ensemble import IsolationForest

# load data
data = pd.read_csv("creditcard.csv")

st.title("💳 Fraud Detection Dashboard")

# show dataset
if st.checkbox("Show Dataset"):
    st.write(data.head())

# class distribution
st.subheader("Class Distribution")
st.bar_chart(data["Class"].value_counts())

# train model
X = data.drop("Class", axis=1)

model = IsolationForest(contamination=0.05, random_state=42)
model.fit(X)

pred = model.predict(X)
pred = [1 if x == -1 else 0 for x in pred]

# alerts
st.subheader("🚨 Fraud Alerts")
count = 0
for i, p in enumerate(pred[:100]):
    if p == 1:
        st.write(f"⚠ Fraud detected at index {i}")
        count += 1

st.write(f"Total fraud alerts (first 100): {count}")