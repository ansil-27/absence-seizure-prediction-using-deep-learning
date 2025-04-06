import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load trained model
model = load_model("C:\\Users\\user\\OneDrive\\Desktop\\seizure\\my_cnn_lstm_model.h5")

# Streamlit UI
st.title("🧠 Epilepsy Detection Dashboard")

# File Upload
uploaded_file = st.file_uploader("📂 Upload a .txt file with exactly 172 numerical values", type=["txt"])

if uploaded_file is not None:
    try:
        # Read file
        data = np.loadtxt(uploaded_file)
        
        # Validate input length
        if len(data) != 172:
            st.error(f"⚠️ Error: Expected 172 values, but got {len(data)}. Please check your file.")
        else:
            # Plot EEG Signal
            st.subheader("📈 EEG Signal Visualization")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(range(1, 173), data, marker="o", linestyle="-", color="b", linewidth=1.5)
            ax.set_xlabel("Time Points")
            ax.set_ylabel("Amplitude")
            ax.set_title("EEG Signal Over Time")
            ax.grid(True)
            st.pyplot(fig)

            # Normalize and reshape data
            scaler = StandardScaler()
            data = scaler.fit_transform(data.reshape(-1, 1)).reshape(1, 172, 1)

            # Make prediction
            prediction = model.predict(data)[0][0]
            epilepsy_status = "Chance of Epilepsy (1)" if prediction >= 0.5 else "No Epilepsy (0)"
            color = "red" if prediction >= 0.5 else "green"

            # Styled Prediction Output
            st.subheader("🔍 Prediction Result")
            st.markdown(
                f"""
                <div style="text-align: center; padding: 20px; border-radius: 10px; background-color: #f8f9fa;">
                    <h2 style="color: {color}; font-size: 28px;"> {epilepsy_status} </h2>
                    <h3 style="color: #333;"> Confidence: {prediction:.2f} </h3>
                </div>
                """,
                unsafe_allow_html=True
            )

    except Exception as e:
        st.error(f"⚠️ Error processing file: {e}")