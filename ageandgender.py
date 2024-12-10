import streamlit as st
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load the trained model
model = load_model("age_gender_model.keras")

# App Title
st.set_page_config(page_title="Age and Gender Prediction", layout="wide")
st.title("🧠 Age and Gender Prediction App")
st.subheader("Upload your image(s) to predict the age and gender of the subject.")

# Sidebar for additional details
st.sidebar.title("About the App")
st.sidebar.info(
    
   ''' This application uses a Convolutional Neural Network (CNN) to predict:
    - **Gender**: Male or Female
    - **Age**: Estimated age of the person
    
    **How it works**:
    1. Upload an image in  JPG, JPEG, PNG format.
    2. The model processes the image and predicts the results.
    
    Developed by: **Haris Faheem and Hassan Nasir**
    Contact: [LinkedIn Profile](https://www.linkedin.com/in/haris-faheem-1376982a3/)'''
    
)

# Function to preprocess and predict
def preprocess_and_predict(image):
    # Preprocess the image
    img = image.resize((128, 128), Image.Resampling.LANCZOS).convert("L")
    st.image(img, caption="Preprocessed Image", width=150, channels="L")
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = img_array.reshape(1, 128, 128, 1)  # Reshape for model input

    # Predict using the loaded model
    gender_pred, age_pred = model.predict(img_array)
    gender = "Female" if gender_pred[0][0] > 0.5 else "Male"
    age = int(age_pred[0][0])
    
    return gender, age

# File uploader for single or multiple files
uploaded_files = st.file_uploader("Upload your image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    results = []  # To store results for batch processing
    for uploaded_file in uploaded_files:
        st.markdown("---")
        st.subheader(f"Prediction for: {uploaded_file.name}")
        
        # Display the uploaded image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Perform prediction
        with st.spinner("Predicting..."):
            gender, age = preprocess_and_predict(image)
        
        # Display results
        st.success("Prediction complete!")
        st.write(f"**Predicted Gender:** {gender}")
        st.write(f"**Predicted Age:** {age}")
        
        # Save results for batch download
        results.append({"Image": uploaded_file.name, "Gender": gender, "Age": age})
    
    # Batch download option
    if len(results) > 1:
        st.markdown("---")
        st.subheader("Batch Processing Results")
        results_df = pd.DataFrame(results)
        st.dataframe(results_df)
        
        # Download button
        csv = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )
