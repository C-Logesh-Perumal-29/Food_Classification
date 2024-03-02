# Import Libraries (or) Packages :

import streamlit as st
from PIL import Image,ImageOps
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tensorflow.keras import preprocessing
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
import os
import h5py
import requests
import json
import streamlit as st
from streamlit_lottie import st_lottie
import streamlit.components.v1 as com

img = Image.open("Icon.png")
st.set_page_config(page_title="Food Classification",page_icon=img,layout="wide")

# Hide Menu_Bar & Footer :

hide_menu_style = """
    <style>
    #MainMenu {visibility : hidden;}
    footer {visibility : hidden;}
    </style>
"""
st.markdown(hide_menu_style , unsafe_allow_html=True)

# Load images :
    
img_icon = Image.open("Icon.png")
bg_image = "https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais"

# Set background image :
    
background_style = f"""
    <style>
        .stApp {{
            background-image: url("{bg_image}");
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
        }}
        header {{
            background-image: url("{bg_image}"); /* Add background image to header */
            background-size: cover;
            background-repeat: no-repeat;
            background-position: center;
            background-color: transparent !important; /* Set background to transparent */
        }}
    </style>
"""

# Center alignment :
    
center_style = """
    <style>
        .center {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0 20%;
        }
    </style>
"""

st.markdown(background_style, unsafe_allow_html=True)
st.markdown(center_style, unsafe_allow_html=True)
st.markdown(hide_menu_style, unsafe_allow_html=True)


col1, col2, col3 = st.columns([1, 4, 1])
with col2:
    st.image(img_icon, use_column_width=True)
    st.markdown("<h1 style='text-align: center; color: #03045e; font-family: Arial Rounded MT Bold;font-size: 80px;'>Food Classification</h1>", unsafe_allow_html=True)

st.markdown("<marquee style='background-color: #ffc6ff; border-radius: 15px; font-family:ROG Fonts; color: #f72585;'>********** Food Classification Using Convolutional Neural Networks **********</marquee>", unsafe_allow_html=True)

# Animations :

x, y, z = st.columns(3)

def get_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with x:
    url1 = get_url("https://assets8.lottiefiles.com/temp/lf20_nXwOJj.json")
    if url1:
        st_lottie(url1, speed=1, width=438, height=200)
        st.markdown('</div>', unsafe_allow_html=True)

with y:
    url2 = get_url("https://assets8.lottiefiles.com/packages/lf20_tll0j4bb.json")
    if url2:
        st_lottie(url2, speed=1.5, width=438, height=200)
        st.markdown('</div>', unsafe_allow_html=True)

with z:
    url3 = get_url("https://assets9.lottiefiles.com/private_files/lf30_y0m027fl.json")
    if url3:
        st_lottie(url3, speed=0.5, width=438, height=200)
        st.markdown('</div>', unsafe_allow_html=True)

# Text : 
    
st.write(""" <div style="color: dodgerblue; font-size: 24px; font-family: Bell MT; text-align: justify; text-justify: inter-word; background-color: rgba(255, 255, 255, 0.3); padding: 20px; border-radius: 15px; box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);">  
                 Identifying food items using an image that is the fascinating thing with various applications. An approach has been presented to classify images of food using convolutional neural networks with the help of machine learning and deep learning. The Convolution neural network is the most popular and extensively used in image classification techniques, Image classification is performed on food dataset using various transfer learning techniques. Therefore, pre-trained models are used in this project which to add more weights of image and also it has given better results. The food dataset of sixteen classes with many images in each class is used for training and also validating. Using these pre-trained models, the given food will be recognized and will be predicted based on the color in the image. Convolutional neural networks have the capability of estimating the score function directly from image pixels. An accuracy of my model is 88%.In these food classification you have to upload an image or using live camera to  my model will predict the accurate results of your uploaded images.
            </div>
        """, unsafe_allow_html=True)

# Audio :
    
btn = st.button("Audio File")

if btn:
    st.audio("Intro.mp3")
    
st.balloons()  

# Select Box to select the items : 
    
s = st.selectbox("Select Upload Type :", options = ["Select One","File Uploader Input","Live Camera Input","Model Predictions"])
    
if s == "File Uploader Input":
        
    def main():
        file_uploader = st.file_uploader("Choose the file",type = ['jpg','jpeg','png'])
        if file_uploader is not None:
            image = Image.open(file_uploader)
            figure = plt.figure()
            plt.imshow(image)
            plt.axis("off")
            result = predict_class(image)
            st.pyplot(figure)                     
            st.success(result) 
                                         
    def predict_class(image):        
        classifier_model = tf.keras.models.load_model("FC_Model.h5")
        shape = ((228,228,3))
        tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
        test_image = image.resize((228,228))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image,axis=0)
        class_names = ['Burger','Chapathi','Dosa','Fried_rice','Idli','Jalebi','Kulfi','Noodles','Paani_poori','Paneer','Parota','Pizza','Poori','Samosa','Soup','Tea']
        predictions = classifier_model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        result = "Prediction of the uploaded image is : {}".format(image_class)
        return result
        st.balloons()
        
    if __name__ == "__main__":
        main()
            
    st.balloons()


elif s == "Live Camera Input":
        
    up = st.camera_input("Capture image",help="This is just a basic example")

    filename = up.name
    with open(filename,'wb') as imagefile:
        imagefile.write(up.getbuffer())

    def main():
            
        if up is not None:
            image = Image.open(up)
            figure = plt.figure()
            plt.imshow(image)
            plt.axis("off")
            result = predict_class(image)
            st.success(result)
            #st.pyplot(figure)
                
    def predict_class(image):
        classifier_model = tf.keras.models.load_model("FC_Model.h5")
        shape = ((228,228,3))
        tf.keras.Sequential([hub.KerasLayer(classifier_model,input_shape=shape)])
        test_image = image.resize((228,228))
        test_image = preprocessing.image.img_to_array(test_image)
        test_image = test_image/255.0
        test_image = np.expand_dims(test_image,axis=0)
        class_names = ['Burger','Chapathi','Dosa','Fried_rice','Idli','Jalebi','Kulfi','Noodles','Paani_poori','Paneer','Parota','Pizza','Poori','Samosa','Soup','Tea']
        predictions = classifier_model.predict(test_image)
        scores = tf.nn.softmax(predictions[0])
        scores = scores.numpy()
        image_class = class_names[np.argmax(scores)]
        result = "Prediction of the captured image is : {}".format(image_class)
        return result

    if __name__ == "__main__":
            main()
        
    st.balloons()

elif s == "Model Predictions":
    
    a1,b1,c1,d1 = st.columns(4)
    with a1:
        st.image("./Predictions/0.png", use_column_width=True)   
    with b1:
        st.image("./Predictions/1.png", use_column_width=True) 
    with c1:
        st.image("./Predictions/2.png", use_column_width=True) 
    with d1:
        st.image("./Predictions/3.png", use_column_width=True)   
        
    a2,b2,c2,d2 = st.columns(4)
    with a2:
        st.image("./Predictions/4.png", use_column_width=True)   
    with b2:
        st.image("./Predictions/5.png", use_column_width=True) 
    with c2:
        st.image("./Predictions/6.png", use_column_width=True) 
    with d2:
        st.image("./Predictions/7.png", use_column_width=True)   
    
    a3,b3,c3,d3 = st.columns(4)
    with a3:
        st.image("./Predictions/8.png", use_column_width=True)   
    with b3:
        st.image("./Predictions/9.png", use_column_width=True) 
    with c3:
        st.image("./Predictions/10.png", use_column_width=True) 
    with d3:
        st.image("./Predictions/11.png", use_column_width=True)   
        
    a4,b4,c4,d4 = st.columns(4)
    with a4:
        st.image("./Predictions/12.png", use_column_width=True)   
    with b4:
        st.image("./Predictions/13.png", use_column_width=True) 
    with c4:
        st.image("./Predictions/14.png", use_column_width=True) 
    with d4:
        st.image("./Predictions/15.png", use_column_width=True)   
        
    a5,b5,c5,d5 = st.columns(4)
    with a5:
        st.image("./Predictions/16.png", use_column_width=True)   
    with b5:
        st.image("./Predictions/17.png", use_column_width=True) 
    with c5:
        st.image("./Predictions/18.png", use_column_width=True) 
    with d5:
        st.image("./Predictions/19.png", use_column_width=True)   
              
    a6,b6,c6,d6 = st.columns(4)
    with a6:
        st.image("./Predictions/20.png", use_column_width=True)   
    with b6:
        st.image("./Predictions/21.png", use_column_width=True) 
    with c6:
        st.image("./Predictions/22.png", use_column_width=True) 
    with d6:
        st.image("./Predictions/23.png", use_column_width=True)   
        
    a7,b7,c7,d7 = st.columns(4)
    with a7:
        st.image("./Predictions/24.png", use_column_width=True)   
    with b7:
        st.image("./Predictions/25.png", use_column_width=True) 
    with c7:
        st.image("./Predictions/26.png", use_column_width=True) 
    with d7:
        st.image("./Predictions/27.png", use_column_width=True)   
        
    a8,b8,c8,d8 = st.columns(4)
    with a8:
        st.image("./Predictions/28.png", use_column_width=True)   
    with b8:
        st.image("./Predictions/29.png", use_column_width=True) 
    with c8:
        st.image("./Predictions/30.png", use_column_width=True) 
    with d8:
        st.image("./Predictions/s1.png", use_column_width=True)   
        
    st.balloons()   

