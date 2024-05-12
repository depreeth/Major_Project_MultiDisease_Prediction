import pickle
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_option_menu import option_menu
from tensorflow.keras.models import load_model
import cv2
import os
import joblib
from io import BytesIO
from keras.preprocessing.image import img_to_array


heart_disease_model = pickle.load(open('../savedModels/Heart_Disease_Prediction.sav','rb'))

# breast_cancer_model = pickle.load(open('breast_cancer.sav', 'rb'))
breast_cancer_model = load_model("../savedModels/Breast_Cancer_Prediction.h5")

alzheimer_model = pickle.load(open('../savedModels/alzheimer_model.sav', 'rb'))
pneumonia_model = load_model('../savedModels/pneumonia_model.h5')
brain_tumor_model = load_model("../savedModels/Brain_tumor_Detection.h5")
# brain_model = load_model("brain_tumor_model.h5")

def preprocess_image(img, target_size):
    # Convert the PIL image to numpy array
    img_array = np.array(img)
    
    # Convert the image to grayscale
    gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to the target size
    resized_img = cv2.resize(gray_img, target_size)
    
    # Flatten the image array
    flattened_img = resized_img.flatten()
    
    # Normalize the pixel values to be in the range [0, 1]
    normalized_img = flattened_img.astype('float32') / 255.0
    
    # Reshape the image to match the input shape expected by the model
    preprocessed_img = normalized_img.reshape(1, -1)
    
    return preprocessed_img


# sidebar for navigation
with st.sidebar:
    
    selected = option_menu('Multiple Disease Prediction System',
                          
                          [
                           'Heart Disease Prediction',
                           'Breast Cancer Prediction',
                           'Alzheimer Prediction',
                           'Brain Tumor Prediction',
                           'Pneumonia Prediction'
                           ],
                          icons=['activity','heart','person'],
                          default_index=0)
    
st.title("Multiple Disease Prediction System")
    

# Heart Disease Prediction Page
if (selected == 'Heart Disease Prediction'):
    
    # page title
    st.subheader('Heart Disease Prediction')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.text_input('Age')
        
    with col2:
        sex = st.text_input('Sex')
        
    with col3:
        cp = st.text_input('Chest Pain types')
        
    with col1:
        trestbps = st.text_input('Resting Blood Pressure')
        
    with col2:
        chol = st.text_input('Serum Cholestoral in mg/dl')
        
    with col3:
        fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
        
    with col1:
        restecg = st.text_input('Resting Electrocardiographic results')
        
    with col2:
        thalach = st.text_input('Maximum Heart Rate achieved')
        
    with col3:
        exang = st.text_input('Exercise Induced Angina')
        
    with col1:
        oldpeak = st.text_input('ST depression induced by exercise')
        
    with col2:
        slope = st.text_input('Slope of the peak exercise ST segment')
        
    with col3:
        ca = st.text_input('Major vessels colored by flourosopy')
        
    with col1:
        thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
        
        
     
     
    # code for Prediction
    heart_diagnosis = ''
    
    # creating a button for Prediction
    
    if st.button('Heart Disease Test Result'):
        heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
        
        if (heart_prediction[0] == 1):
          heart_diagnosis = 'The person is having heart disease'
        else:
          heart_diagnosis = 'The person does not have any heart disease'
        
    st.success(heart_diagnosis)
        
    
    

# Breast Cancer Prediction Page
if (selected == "Breast Cancer Prediction"):
    
    # page title
    st.subheader("Breast Cancer Prediction")
    
    col1, col2 = st.columns(2)  
    
    with col1:
        mr = st.text_input('mean radius')
        
    with col2:
        mp = st.text_input('mean perimeter')
        
    with col1:
        ma = st.text_input('mean area	')
        
    with col2:
        mc = st.text_input('mean compactness')
        
    with col1:
        mcon = st.text_input('mean concavity')

    with col2:
        mcp = st.text_input('mean concave points')
        
    
    
    # code for Prediction
    breast_cancer_diagnosis = ''
    
    # creating a button for Prediction    
    if st.button("Breast Cancer Test Result"):
        input_data = np.array([[float(mr), float(mp), float(ma), float(mc), float(mcon), float(mcp)]])
        breast_cancer_prediction = breast_cancer_model.predict(input_data)                          
        
        if (breast_cancer_prediction[0] >= 0.7):
          breast_cancer_diagnosis = "The person has Breast Cancer disease"
        else:
          breast_cancer_diagnosis = "The person does not have Breast Cancer disease"
        
    st.success(breast_cancer_diagnosis)
    

# Heart Disease Prediction Page
if (selected == 'Alzheimer Prediction'):
    new_size = (256, 256) 
    width, height = 256, 256
    st.subheader('Alzheimer Prediction')
    
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image")
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        try:
            img = Image.open(uploaded_file)
        except:
            st.error("Invalid image file or unable to open image.")
            st.stop()  # Stop execution if an error occurs
        
        # Resize the image if needed (modify 'new_size' according to your requirements)
        img = img.resize(new_size)
        
        # Convert the image to a numpy array
        array_temp = np.array(img)
        
        # Reshape the image
        shape_new = width * height
        img_wide = array_temp.reshape(1, shape_new)
        
        # Predict the image
        prediction = alzheimer_model.predict(img_wide)
        
        # Map prediction to corresponding label
        if prediction == 0:
            alzheimer_diagnosis = "The person does not have Alzheimer's disease."
        elif prediction == 1:
            alzheimer_diagnosis = "The person has very mild Alzheimer's disease."
        elif prediction == 2:
            alzheimer_diagnosis = "The person has mild Alzheimer's disease."
        else:
            alzheimer_diagnosis = "The person has moderate Alzheimer's disease."

        
        # Display the prediction result
        st.success(alzheimer_diagnosis)

if (selected == 'Brain Tumor Prediction'):
    new_size = (256, 256) 
    width, height = 256, 256
    st.subheader('Brain Tumor Prediction')
    
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image")
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        try:
            img = Image.open(uploaded_file)
        except:
            st.error("Invalid image file or unable to open image.")
            st.stop()  # Stop execution if an error occurs
        
        # Resize the image if needed (modify 'new_size' according to your requirements)
        target_size = (100, 100)
        preprocessed_image = preprocess_image(img, target_size)        
        
        # # Predict the image
        prediction = brain_tumor_model.predict(preprocessed_image)
        
        # Map prediction to corresponding label
        if prediction < 0.6:
            brain_tumor_diagnosis = "The person has Brain Tumor disease.",prediction
        else:
            brain_tumor_diagnosis = "The person does not have Brain Tumor disease.",prediction

        # Display the prediction result
        st.success(brain_tumor_diagnosis)

def preprocess_image1(image):
    # Resize the image to the target size
    image = image.resize((120, 120))
    
    # Convert the image to an array
    image_array = img_to_array(image)
    
    # Rescale the image to [0, 1]
    image_array /= 255.0
    
    return image_array

if selected == 'Pneumonia Prediction':
    st.subheader('Pneumonia Prediction')
    new_size = (256, 256) 
    width, height = 256, 256

    
    # Allow user to upload an image
    uploaded_file = st.file_uploader("Upload an image")
    
    if uploaded_file is not None:
        # Convert the uploaded file to an image
        try:
            img = Image.open(uploaded_file)
            img = img.convert('RGB')
        except:
            st.error("Invalid image file or unable to open image.")
            st.stop()  # Stop execution if an error occurs
        
        # Resize the image if needed (modify 'new_size' according to your requirements)
        target_size = (120, 120)
        preprocessed_image = preprocess_image1(img)
        
        # # Predict the image
        prediction = pneumonia_model.predict(np.expand_dims(preprocessed_image, axis=0))

        
        # Map prediction to corresponding label
        if prediction[0] > 0.5:
            pneumonia_model_diagnosis = "The person has pneumonia disease."
        else:
            pneumonia_model_diagnosis = "The person does not have pneumonia disease."
        # Display the prediction result
        st.success(pneumonia_model_diagnosis)



def set_bg_from_url(url, opacity=1):
    
    footer = """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.2.0/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-gH2yIJqKdNHPEq0n4Mqa/HGKIhSkIHeL5AyhkYV8i59U5AR6csBvApHHNl/vI1Bx" crossorigin="anonymous">

"""
    st.markdown(footer, unsafe_allow_html=True)
    
    
    # Set background image using HTML and CSS
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: {opacity};
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Set background image from URL
set_bg_from_url("https://images.everydayhealth.com/homepage/health-topics-2.jpg?w=768", opacity=0.875)
