import keras.utils
import streamlit as st
import keras
import tensorflow as tf
from PIL import Image
import numpy as np

try:
    model = keras.models.load_model('model_0.h5')
except Exception as e:
    st.error(f"Error loading model: {e}. Make sure 'model_0.h5' is in the correct directory.")
    st.stop() 

st.write("This is a toy app, that I created, where in you can upload image of some animal and get it's name")
st.write("The backend is using a very basic CNN, whose architecture is as given : ")


try:
    st.image('CNN_ARCHITECTURE.bmp', caption='ARCHITECTURE')
except FileNotFoundError:
    st.warning("CNN_ARCHITECTURE.bmp not found. Please ensure it's in the same directory.")
except Exception as e:
    st.error(f"Error displaying architecture image: {e}")


upload_image = st.file_uploader("Upload Your Image", type=['jpeg', 'png', 'jpg'])


if upload_image is not None:
  
    img = Image.open(upload_image)


    st.image(img, caption='Uploaded Image', use_column_width=True)

    img = img.resize((256,256)) 
    img_arr = keras.utils.img_to_array(img)
    img_arr /= 255.0 
    input_image_for_prediction = np.expand_dims(img_arr, axis=0) 

    predictions = model.predict(input_image_for_prediction)

    predicted_class_index = np.argmax(predictions[0])

    
    class_names = ['dog', 'horse', 'elephant', 'butterfly', 'chicken',
                   'cat', 'cow', 'sheep', 'spider', 'squirrel'] 

   

    st.write(f"### Predicted Animal: {class_names[predicted_class_index]}")
    st.write(f"Confidence: {predictions[0][predicted_class_index]*100:.2f}%")

    st.write("---")
    st.write("#### All Probabilities:")
    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

else:
    st.write("Please upload an image to get a prediction.")