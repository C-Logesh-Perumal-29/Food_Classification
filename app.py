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
import pickle

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

# Set the background image :

Background_image = """
<style>
[data-testid="stAppViewContainer"] > .main
{
background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");

background-size : 100%
background-position : top left;
background-position: center;
background-size: cover;
background-repeat : repeat;
background-repeat: round;
background-attachment : local;
background-image: url("https://img.freepik.com/free-photo/vivid-blurred-colorful-background_58702-2655.jpg?size=626&ext=jpg&ga=GA1.1.2087154549.1663432512&semt=ais");
background-position: right bottom;
background-repeat: no-repeat;
}  

[data-testid="stHeader"]
{
background-color : rgba(0,0,0,0);
}
</style>                                
"""
st.markdown(Background_image,unsafe_allow_html=True)

col11,col22 = st.columns([1,5])

with col11:
    img = Image.open("13.jpg")
    st.image(img)

with col22:

    title_html = f'<p style="color:#03045e; text-align:center;font-family:Eras Bold ITC;font-size:96px;">Food Classification</p>'
    st.markdown(title_html, unsafe_allow_html=True)

st.markdown('<marquee style="background-color:#ffc6ff;font-family:ROG Fonts;border-radius:15px; color:#f72585;"> ********** Food Classification Using Convolutional Neural Networks ********** <marquee>',unsafe_allow_html=True)

# Title :

with open("d.css") as source:
   st.markdown(f"<style>{source.read()}</style>",unsafe_allow_html=True)
   
# Animations :

x,y,z = st.columns(3)

def get_url(url:str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

with x:
    url1 = get_url("https://assets8.lottiefiles.com/temp/lf20_nXwOJj.json")
    st_lottie(url1)
    
with y:
    url2 = get_url("https://assets8.lottiefiles.com/packages/lf20_tll0j4bb.json")
    st_lottie(url2)
    
with z:
    url3 = get_url("https://assets9.lottiefiles.com/private_files/lf30_y0m027fl.json")
    st_lottie(url3)

x1,y1 = st.columns([5,2])

# Text & Images :

with x1:
    st.write("Identifying food items using an image that is the fascinating thing with various applications. An approach has been presented to classify images of food using convolutional neural networks with the help of machine learning and deep learning. The Convolution neural network is the most popular and extensively used in image classification techniques, Image classification is performed on food dataset using various transfer learning techniques. Therefore, pre-trained models are used in this project which to add more weights of image and also it has given better results. The food dataset of sixteen classes with many images in each class is used for training and also validating. Using these pre-trained models, the given food will be recognized and will be predicted based on the color in the image. Convolutional neural networks have the capability of estimating the score function directly from image pixels. An accuracy of my model is 88%.In these food classification you have to upload an image or using live camera to  my model will predict the accurate results of your uploaded images.")
    st.markdown(
        """  
        1) Burger 
        2) Chapathi 
        3) Dosa  
        4) Fried_rice 
        5) Idli  
        6) Jalebi 
        7) Kulfi 
        8) Noodles 
        9) Paani_poori 
        10) Panner 
        11) Parotta 
        12) Pizza
        13) Poori 
        14) Samosa 
        15) Soup 
        16) Tea 
        """ )     
            

with y1:
    st.image("f1.jpg")
    st.image("f2.jpeg")
    st.image("f3.jpeg")
    
st.balloons()

# Text & Image alteration :

text = """
<style>

.css-1fv8s86.e16nr0p34
{
     color:black;
     text-align: justify;
}
</style>
"""
st.markdown(text,unsafe_allow_html=True)

image = """
<style>
.css-1v0mbdj.etr89bj1
{
     box-shadow: 0 4px 8px 0 rgba(0, 0, 0, 0.2), 0 6px 20px 0 rgba(0, 0, 0, 0.19);
}
</style> 
"""

st.markdown(image,unsafe_allow_html=True)

# Audio :
    
col1,col2,col3 = st.columns([2,1,2])
with col2:
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
            #st.write(result)
            st.pyplot(figure)
            st.success(result) 
                                         
    def predict_class(image):        
        classifier_model = tf.keras.models.load_model("FC_Model.h5")
        #classifier_model = pickle.load(open("D:\\Project_Web\\Model\\C_LP-FC.pkl", 'rb'))
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
            st.pyplot(figure)
                
    def predict_class(image):
        classifier_model = tf.keras.models.load_model("FC_Model.h5")
        #classifier_model = pickle.load(open("D:\\Project_Web\\Model\\C_LP-FC.pkl", 'rb'))
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
        st.image("./Predictions/0.png")   
    with b1:
        st.image("./Predictions/1.png") 
    with c1:
        st.image("./Predictions/2.png") 
    with d1:
        st.image("./Predictions/3.png")   
        
    a2,b2,c2,d2 = st.columns(4)
    with a2:
        st.image("./Predictions/4.png")   
    with b2:
        st.image("./Predictions/5.png") 
    with c2:
        st.image("./Predictions/6.png") 
    with d2:
        st.image("./Predictions/7.png")   
    
    a3,b3,c3,d3 = st.columns(4)
    with a3:
        st.image("./Predictions/8.png")   
    with b3:
        st.image("./Predictions/9.png") 
    with c3:
        st.image("./Predictions/10.png") 
    with d3:
        st.image("./Predictions/11.png")   
        
    a4,b4,c4,d4 = st.columns(4)
    with a4:
        st.image("./Predictions/12.png")   
    with b4:
        st.image("./Predictions/13.png") 
    with c4:
        st.image("./Predictions/14.png") 
    with d4:
        st.image("./Predictions/15.png")   
        
    a5,b5,c5,d5 = st.columns(4)
    with a5:
        st.image("./Predictions/16.png")   
    with b5:
        st.image("./Predictions/17.png") 
    with c5:
        st.image("./Predictions/18.png") 
    with d5:
        st.image("./Predictions/19.png")   
              
    a6,b6,c6,d6 = st.columns(4)
    with a6:
        st.image("./Predictions/20.png")   
    with b6:
        st.image("./Predictions/21.png") 
    with c6:
        st.image("./Predictions/22.png") 
    with d6:
        st.image("./Predictions/23.png")   
        
    a7,b7,c7,d7 = st.columns(4)
    with a7:
        st.image("./Predictions/24.png")   
    with b7:
        st.image("./Predictions/25.png") 
    with c7:
        st.image("./Predictions/26.png") 
    with d7:
        st.image("./Predictions/27.png")   
        
    a8,b8,c8,d8 = st.columns(4)
    with a8:
        st.image("./Predictions/28.png")   
    with b8:
        st.image("./Predictions/29.png") 
    with c8:
        st.image("./Predictions/30.png") 
    with d8:
        st.image("./Predictions/s1.png")   
        
    st.balloons()   
