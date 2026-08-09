import streamlit as st
from cust_seg import classify
import joblib as jb

final_centroids=jb.load("centroids.pkl")
cluster_labels=jb.load("cluster_labels.pkl")
st.title("🛍️ Customer Segment Classifier")

income = st.number_input("Annual Income", min_value=0, max_value=200, value=50)
spending_score = st.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

if st.button("Classify Customer"):
    cluster_id, label = classify(income, spending_score, final_centroids, cluster_labels)
    st.success(f"This customer belongs to: **{label}**")