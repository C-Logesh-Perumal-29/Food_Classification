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

#st.set_page_config(page_title="Hello",page_icon=":smile:")

img = Image.open("13.jpg")
st.set_page_config(page_title="Food Classification Using CNN",page_icon=img,layout="wide")

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
background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20210228/pngtree-spring-color-creative-floral-background-image_570994.jpg");

background-size : 100%
background-position : top left;

background-position: center;
background-size: cover;
background-repeat : repeat;
background-repeat: round;


background-attachment : local;

background-image: url("https://png.pngtree.com/thumb_back/fh260/background/20210228/pngtree-spring-color-creative-floral-background-image_570994.jpg");
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

    #st.title("Food Classification")
    title_html = f'<p style="color:#FF5733; text-align:center;font-family:Eras Bold ITC;font-size:96px;">Food Classification</p>'
    st.markdown(title_html, unsafe_allow_html=True)

st.markdown('<marquee style="background-color:#FF00FF;font-family:ROG Fonts;border-radius:15px; color:white;"> ********** Food Classification Using Convolutional Neural Networks ********** <marquee>',unsafe_allow_html=True)

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
    
#st.image("Flowchart.png","Food Classification Classes")
    
# Select Box to select the items : 

q = st.selectbox("Select your choice",["Select One","Food Classification","Sample training Images","Code","Predictions"])


if q == "Food Classification":
    
    s = st.selectbox("Select Upload Type :", options = ["Select One","File Uploader","Camera Input"])
    
    if s == "File Uploader":
        
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
            classifier_model = tf.keras.models.load_model("Model/FC_Model.h5")
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


    elif s == "Camera Input":
        
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
            classifier_model = tf.keras.models.load_model("Model/FC_Model.h5")
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
        
elif q == "Sample training Images":
    
    q1,w,e,ff = st.columns(4)
    with q1: 
        st.image("./Dataset/1.jpg")
    with w: 
        st.image("./Dataset/2.jpg")
    with e: 
        st.image("./Dataset/3.jpg")
    with ff:
        st.image("./Dataset/4.jpg")
        
    r,t,y,u = st.columns(4)
    with r: 
        st.image("./Dataset/5.jpg")
    with t: 
        st.image("./Dataset/6.jpg")
    with y: 
        st.image("./Dataset/7.jpg")
    with u:
        st.image("./Dataset/8.jpg")
        
    u1,i,o,p = st.columns(4)
    with i: 
        st.image("./Dataset/9.jpg")
    with o: 
        st.image("./Dataset/10.jpg")
    with p: 
        st.image("./Dataset/11.jpg")
    with u1:
         st.image("./Dataset/20.jpg")
        
    m,n,v,c = st.columns(4)
    with m: 
        st.image("./Dataset/12.jpg")
    with n: 
        st.image("./Dataset/13.jpg")
    with v: 
         st.image("./Dataset/14.jpg")
    with c: 
         st.image("./Dataset/15.jpg")
              
    z,z1,z2,z3 = st.columns(4)
    with z: 
        st.image("./Dataset/16.jpg")
    with z1: 
        st.image("./Dataset/17.jpg")
    with z2: 
        st.image("./Dataset/18.jpg")
    with z3: 
        st.image("./Dataset/19.jpg")
    st.balloons()
             
elif q == "Predictions":
            
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


elif q == "Code":
    

    c01 = '''
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image'''
    st.code(c01, language='python')
    
    c02 = '''
data_dir="D:\\Pro\\Dataset"
data = tf.keras.preprocessing.image_dataset_from_directory(data_dir)'''
    st.code(c02, language='python')    

    st.success("Found 4228 files belonging to 16 classes.")
        
    c03 = '''
# Create an ImageDataGenerator and do Image Augmentation
datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split = 0.2)'''
    st.code(c03, language='python')
        
    c04 = '''height = 228
width = 228
channels = 3
batch_size = 32
img_shape = (height, width, channels)
img_size = (height, width)'''
    st.code(c04, language='python')  
    
    c05 = '''train_data = datagen.flow_from_directory(
    data_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training')

val_data = datagen.flow_from_directory(
    data_dir,
    target_size = img_size,
    batch_size = batch_size,
    class_mode='categorical',
    subset = 'validation')'''
    st.code(c05, language='python')  
    
    st.success("""Found 3388 images belonging to 16 classes. 
               Found 840 images belonging to 16 classes.""")
     
    c06 = '''num_classes = len(data.class_names)
print('.... Number of Classes : {0} ....'.format(num_classes))'''
    st.code(c06,language='python')  
    
    st.success(".... Number of Classes : 16 ....")
    
    c07 = '''#Defing a function to see images
def show_img(data):
    plt.figure(figsize=(15,15))
    for images, labels in data.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            ax.imshow(images[i].numpy().astype("uint8"))
            ax.axis("off")            
#Plotting the images in dataset
show_img(data)'''
    st.code(c07, language='python')


    c08 = '''# load pre-trained InceptionV3
pre_trained = InceptionV3(weights='imagenet', include_top=False, input_shape=img_shape, pooling='avg')

for layer in pre_trained.layers:
    layer.trainable = False'''
    st.code(c08, language='python')  
    
    c09 = '''x = pre_trained.output
x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
x = Dropout(0.2)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs = pre_trained.input, outputs = predictions)
model.compile(optimizer = Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary() # Shows the informations or details'''
    st.code(c09, language='python')      
    
    st.success("""
               
Model: "model"
__________________________________________________________________________________________________
 Layer (type)                   Output Shape         Param #     Connected to                     
==================================================================================================
 input_1 (InputLayer)           [(None, 228, 228, 3  0           []                               
                                )]                                                                
                                                                                                  
 conv2d (Conv2D)                (None, 113, 113, 32  864         ['input_1[0][0]']                
                                )                                                                 
                                                                                                  
 batch_normalization (BatchNorm  (None, 113, 113, 32  96         ['conv2d[0][0]']                 
 alization)                     )                                                                 
                                                                                                  
 activation (Activation)        (None, 113, 113, 32  0           ['batch_normalization[0][0]']    
                                )                                                                 
                                                                                                  
 conv2d_1 (Conv2D)              (None, 111, 111, 32  9216        ['activation[0][0]']             
                                )                                                                 
                                                                                                  
 batch_normalization_1 (BatchNo  (None, 111, 111, 32  96         ['conv2d_1[0][0]']               
 rmalization)                   )                                                                 
                                                                                                  
 activation_1 (Activation)      (None, 111, 111, 32  0           ['batch_normalization_1[0][0]']  
                                )                                                                 
                                                                                                  
 conv2d_2 (Conv2D)              (None, 111, 111, 64  18432       ['activation_1[0][0]']           
                                )                                                                 
                                                                                                  
 batch_normalization_2 (BatchNo  (None, 111, 111, 64  192        ['conv2d_2[0][0]']               
 rmalization)                   )                                                                 
                                                                                                  
 activation_2 (Activation)      (None, 111, 111, 64  0           ['batch_normalization_2[0][0]']  
                                )                                                                 
                                                                                                  
 max_pooling2d (MaxPooling2D)   (None, 55, 55, 64)   0           ['activation_2[0][0]']           
                                                                                                  
 conv2d_3 (Conv2D)              (None, 55, 55, 80)   5120        ['max_pooling2d[0][0]']          
                                                                                                  
 batch_normalization_3 (BatchNo  (None, 55, 55, 80)  240         ['conv2d_3[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_3 (Activation)      (None, 55, 55, 80)   0           ['batch_normalization_3[0][0]']  
                                                                                                  
 conv2d_4 (Conv2D)              (None, 53, 53, 192)  138240      ['activation_3[0][0]']           
                                                                                                  
 batch_normalization_4 (BatchNo  (None, 53, 53, 192)  576        ['conv2d_4[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_4 (Activation)      (None, 53, 53, 192)  0           ['batch_normalization_4[0][0]']  
                                                                                                  
 max_pooling2d_1 (MaxPooling2D)  (None, 26, 26, 192)  0          ['activation_4[0][0]']           
                                                                                                  
 conv2d_8 (Conv2D)              (None, 26, 26, 64)   12288       ['max_pooling2d_1[0][0]']        
                                                                                                  
 batch_normalization_8 (BatchNo  (None, 26, 26, 64)  192         ['conv2d_8[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_8 (Activation)      (None, 26, 26, 64)   0           ['batch_normalization_8[0][0]']  
                                                                                                  
 conv2d_6 (Conv2D)              (None, 26, 26, 48)   9216        ['max_pooling2d_1[0][0]']        
                                                                                                  
 conv2d_9 (Conv2D)              (None, 26, 26, 96)   55296       ['activation_8[0][0]']           
                                                                                                  
 batch_normalization_6 (BatchNo  (None, 26, 26, 48)  144         ['conv2d_6[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 batch_normalization_9 (BatchNo  (None, 26, 26, 96)  288         ['conv2d_9[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 activation_6 (Activation)      (None, 26, 26, 48)   0           ['batch_normalization_6[0][0]']  
                                                                                                  
 activation_9 (Activation)      (None, 26, 26, 96)   0           ['batch_normalization_9[0][0]']  
                                                                                                  
 average_pooling2d (AveragePool  (None, 26, 26, 192)  0          ['max_pooling2d_1[0][0]']        
 ing2D)                                                                                           
                                                                                                  
 conv2d_5 (Conv2D)              (None, 26, 26, 64)   12288       ['max_pooling2d_1[0][0]']        
                                                                                                  
 conv2d_7 (Conv2D)              (None, 26, 26, 64)   76800       ['activation_6[0][0]']           
                                                                                                  
 conv2d_10 (Conv2D)             (None, 26, 26, 96)   82944       ['activation_9[0][0]']           
                                                                                                  
 conv2d_11 (Conv2D)             (None, 26, 26, 32)   6144        ['average_pooling2d[0][0]']      
                                                                                                  
 batch_normalization_5 (BatchNo  (None, 26, 26, 64)  192         ['conv2d_5[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 batch_normalization_7 (BatchNo  (None, 26, 26, 64)  192         ['conv2d_7[0][0]']               
 rmalization)                                                                                     
                                                                                                  
 batch_normalization_10 (BatchN  (None, 26, 26, 96)  288         ['conv2d_10[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_11 (BatchN  (None, 26, 26, 32)  96          ['conv2d_11[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_5 (Activation)      (None, 26, 26, 64)   0           ['batch_normalization_5[0][0]']  
                                                                                                  
 activation_7 (Activation)      (None, 26, 26, 64)   0           ['batch_normalization_7[0][0]']  
                                                                                                  
 activation_10 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_10[0][0]'] 
                                                                                                  
 activation_11 (Activation)     (None, 26, 26, 32)   0           ['batch_normalization_11[0][0]'] 
                                                                                                  
 mixed0 (Concatenate)           (None, 26, 26, 256)  0           ['activation_5[0][0]',           
                                                                  'activation_7[0][0]',           
                                                                  'activation_10[0][0]',          
                                                                  'activation_11[0][0]']          
                                                                                                  
 conv2d_15 (Conv2D)             (None, 26, 26, 64)   16384       ['mixed0[0][0]']                 
                                                                                                  
 batch_normalization_15 (BatchN  (None, 26, 26, 64)  192         ['conv2d_15[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_15 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_15[0][0]'] 
                                                                                                  
 conv2d_13 (Conv2D)             (None, 26, 26, 48)   12288       ['mixed0[0][0]']                 
                                                                                                  
 conv2d_16 (Conv2D)             (None, 26, 26, 96)   55296       ['activation_15[0][0]']          
                                                                                                  
 batch_normalization_13 (BatchN  (None, 26, 26, 48)  144         ['conv2d_13[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_16 (BatchN  (None, 26, 26, 96)  288         ['conv2d_16[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_13 (Activation)     (None, 26, 26, 48)   0           ['batch_normalization_13[0][0]'] 
                                                                                                  
 activation_16 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_16[0][0]'] 
                                                                                                  
 average_pooling2d_1 (AveragePo  (None, 26, 26, 256)  0          ['mixed0[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_12 (Conv2D)             (None, 26, 26, 64)   16384       ['mixed0[0][0]']                 
                                                                                                  
 conv2d_14 (Conv2D)             (None, 26, 26, 64)   76800       ['activation_13[0][0]']          
                                                                                                  
 conv2d_17 (Conv2D)             (None, 26, 26, 96)   82944       ['activation_16[0][0]']          
                                                                                                  
 conv2d_18 (Conv2D)             (None, 26, 26, 64)   16384       ['average_pooling2d_1[0][0]']    
                                                                                                  
 batch_normalization_12 (BatchN  (None, 26, 26, 64)  192         ['conv2d_12[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_14 (BatchN  (None, 26, 26, 64)  192         ['conv2d_14[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_17 (BatchN  (None, 26, 26, 96)  288         ['conv2d_17[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_18 (BatchN  (None, 26, 26, 64)  192         ['conv2d_18[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_12 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_12[0][0]'] 
                                                                                                  
 activation_14 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_14[0][0]'] 
                                                                                                  
 activation_17 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_17[0][0]'] 
                                                                                                  
 activation_18 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_18[0][0]'] 
                                                                                                  
 mixed1 (Concatenate)           (None, 26, 26, 288)  0           ['activation_12[0][0]',          
                                                                  'activation_14[0][0]',          
                                                                  'activation_17[0][0]',          
                                                                  'activation_18[0][0]']          
                                                                                                  
 conv2d_22 (Conv2D)             (None, 26, 26, 64)   18432       ['mixed1[0][0]']                 
                                                                                                  
 batch_normalization_22 (BatchN  (None, 26, 26, 64)  192         ['conv2d_22[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_22 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_22[0][0]'] 
                                                                                                  
 conv2d_20 (Conv2D)             (None, 26, 26, 48)   13824       ['mixed1[0][0]']                 
                                                                                                  
 conv2d_23 (Conv2D)             (None, 26, 26, 96)   55296       ['activation_22[0][0]']          
                                                                                                  
 batch_normalization_20 (BatchN  (None, 26, 26, 48)  144         ['conv2d_20[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_23 (BatchN  (None, 26, 26, 96)  288         ['conv2d_23[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_20 (Activation)     (None, 26, 26, 48)   0           ['batch_normalization_20[0][0]'] 
                                                                                                  
 activation_23 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_23[0][0]'] 
                                                                                                  
 average_pooling2d_2 (AveragePo  (None, 26, 26, 288)  0          ['mixed1[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_19 (Conv2D)             (None, 26, 26, 64)   18432       ['mixed1[0][0]']                 
                                                                                                  
 conv2d_21 (Conv2D)             (None, 26, 26, 64)   76800       ['activation_20[0][0]']          
                                                                                                  
 conv2d_24 (Conv2D)             (None, 26, 26, 96)   82944       ['activation_23[0][0]']          
                                                                                                  
 conv2d_25 (Conv2D)             (None, 26, 26, 64)   18432       ['average_pooling2d_2[0][0]']    
                                                                                                  
 batch_normalization_19 (BatchN  (None, 26, 26, 64)  192         ['conv2d_19[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_21 (BatchN  (None, 26, 26, 64)  192         ['conv2d_21[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_24 (BatchN  (None, 26, 26, 96)  288         ['conv2d_24[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_25 (BatchN  (None, 26, 26, 64)  192         ['conv2d_25[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_19 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_19[0][0]'] 
                                                                                                  
 activation_21 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_21[0][0]'] 
                                                                                                  
 activation_24 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_24[0][0]'] 
                                                                                                  
 activation_25 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_25[0][0]'] 
                                                                                                  
 mixed2 (Concatenate)           (None, 26, 26, 288)  0           ['activation_19[0][0]',          
                                                                  'activation_21[0][0]',          
                                                                  'activation_24[0][0]',          
                                                                  'activation_25[0][0]']          
                                                                                                  
 conv2d_27 (Conv2D)             (None, 26, 26, 64)   18432       ['mixed2[0][0]']                 
                                                                                                  
 batch_normalization_27 (BatchN  (None, 26, 26, 64)  192         ['conv2d_27[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_27 (Activation)     (None, 26, 26, 64)   0           ['batch_normalization_27[0][0]'] 
                                                                                                  
 conv2d_28 (Conv2D)             (None, 26, 26, 96)   55296       ['activation_27[0][0]']          
                                                                                                  
 batch_normalization_28 (BatchN  (None, 26, 26, 96)  288         ['conv2d_28[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_28 (Activation)     (None, 26, 26, 96)   0           ['batch_normalization_28[0][0]'] 
                                                                                                  
 conv2d_26 (Conv2D)             (None, 12, 12, 384)  995328      ['mixed2[0][0]']                 
                                                                                                  
 conv2d_29 (Conv2D)             (None, 12, 12, 96)   82944       ['activation_28[0][0]']          
                                                                                                  
 batch_normalization_26 (BatchN  (None, 12, 12, 384)  1152       ['conv2d_26[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_29 (BatchN  (None, 12, 12, 96)  288         ['conv2d_29[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_26 (Activation)     (None, 12, 12, 384)  0           ['batch_normalization_26[0][0]'] 
                                                                                                  
 activation_29 (Activation)     (None, 12, 12, 96)   0           ['batch_normalization_29[0][0]'] 
                                                                                                  
 max_pooling2d_2 (MaxPooling2D)  (None, 12, 12, 288)  0          ['mixed2[0][0]']                 
                                                                                                  
 mixed3 (Concatenate)           (None, 12, 12, 768)  0           ['activation_26[0][0]',          
                                                                  'activation_29[0][0]',          
                                                                  'max_pooling2d_2[0][0]']        
                                                                                                  
 conv2d_34 (Conv2D)             (None, 12, 12, 128)  98304       ['mixed3[0][0]']                 
                                                                                                  
 batch_normalization_34 (BatchN  (None, 12, 12, 128)  384        ['conv2d_34[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_34 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_34[0][0]'] 
                                                                                                  
 conv2d_35 (Conv2D)             (None, 12, 12, 128)  114688      ['activation_34[0][0]']          
                                                                                                  
 batch_normalization_35 (BatchN  (None, 12, 12, 128)  384        ['conv2d_35[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_35 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_35[0][0]'] 
                                                                                                  
 conv2d_31 (Conv2D)             (None, 12, 12, 128)  98304       ['mixed3[0][0]']                 
                                                                                                  
 conv2d_36 (Conv2D)             (None, 12, 12, 128)  114688      ['activation_35[0][0]']          
                                                                                                  
 batch_normalization_31 (BatchN  (None, 12, 12, 128)  384        ['conv2d_31[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_36 (BatchN  (None, 12, 12, 128)  384        ['conv2d_36[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_31 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_31[0][0]'] 
                                                                                                  
 activation_36 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_36[0][0]'] 
                                                                                                  
 conv2d_32 (Conv2D)             (None, 12, 12, 128)  114688      ['activation_31[0][0]']          
                                                                                                  
 conv2d_37 (Conv2D)             (None, 12, 12, 128)  114688      ['activation_36[0][0]']          
                                                                                                  
 batch_normalization_32 (BatchN  (None, 12, 12, 128)  384        ['conv2d_32[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_37 (BatchN  (None, 12, 12, 128)  384        ['conv2d_37[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_32 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_32[0][0]'] 
                                                                                                  
 activation_37 (Activation)     (None, 12, 12, 128)  0           ['batch_normalization_37[0][0]'] 
                                                                                                  
 average_pooling2d_3 (AveragePo  (None, 12, 12, 768)  0          ['mixed3[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_30 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed3[0][0]']                 
                                                                                                  
 conv2d_33 (Conv2D)             (None, 12, 12, 192)  172032      ['activation_32[0][0]']          
                                                                                                  
 conv2d_38 (Conv2D)             (None, 12, 12, 192)  172032      ['activation_37[0][0]']          
                                                                                                  
 conv2d_39 (Conv2D)             (None, 12, 12, 192)  147456      ['average_pooling2d_3[0][0]']    
                                                                                                  
 batch_normalization_30 (BatchN  (None, 12, 12, 192)  576        ['conv2d_30[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_33 (BatchN  (None, 12, 12, 192)  576        ['conv2d_33[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_38 (BatchN  (None, 12, 12, 192)  576        ['conv2d_38[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_39 (BatchN  (None, 12, 12, 192)  576        ['conv2d_39[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_30 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_30[0][0]'] 
                                                                                                  
 activation_33 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_33[0][0]'] 
                                                                                                  
 activation_38 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_38[0][0]'] 
                                                                                                  
 activation_39 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_39[0][0]'] 
                                                                                                  
 mixed4 (Concatenate)           (None, 12, 12, 768)  0           ['activation_30[0][0]',          
                                                                  'activation_33[0][0]',          
                                                                  'activation_38[0][0]',          
                                                                  'activation_39[0][0]']          
                                                                                                  
 conv2d_44 (Conv2D)             (None, 12, 12, 160)  122880      ['mixed4[0][0]']                 
                                                                                                  
 batch_normalization_44 (BatchN  (None, 12, 12, 160)  480        ['conv2d_44[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_44 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_44[0][0]'] 
                                                                                                  
 conv2d_45 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_44[0][0]']          
                                                                                                  
 batch_normalization_45 (BatchN  (None, 12, 12, 160)  480        ['conv2d_45[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_45 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_45[0][0]'] 
                                                                                                  
 conv2d_41 (Conv2D)             (None, 12, 12, 160)  122880      ['mixed4[0][0]']                 
                                                                                                  
 conv2d_46 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_45[0][0]']          
                                                                                                  
 batch_normalization_41 (BatchN  (None, 12, 12, 160)  480        ['conv2d_41[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_46 (BatchN  (None, 12, 12, 160)  480        ['conv2d_46[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_41 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_41[0][0]'] 
                                                                                                  
 activation_46 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_46[0][0]'] 
                                                                                                  
 conv2d_42 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_41[0][0]']          
                                                                                                  
 conv2d_47 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_46[0][0]']          
                                                                                                  
 batch_normalization_42 (BatchN  (None, 12, 12, 160)  480        ['conv2d_42[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_47 (BatchN  (None, 12, 12, 160)  480        ['conv2d_47[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_42 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_42[0][0]'] 
                                                                                                  
 activation_47 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_47[0][0]'] 
                                                                                                  
 average_pooling2d_4 (AveragePo  (None, 12, 12, 768)  0          ['mixed4[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_40 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed4[0][0]']                 
                                                                                                  
 conv2d_43 (Conv2D)             (None, 12, 12, 192)  215040      ['activation_42[0][0]']          
                                                                                                  
 conv2d_48 (Conv2D)             (None, 12, 12, 192)  215040      ['activation_47[0][0]']          
                                                                                                  
 conv2d_49 (Conv2D)             (None, 12, 12, 192)  147456      ['average_pooling2d_4[0][0]']    
                                                                                                  
 batch_normalization_40 (BatchN  (None, 12, 12, 192)  576        ['conv2d_40[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_43 (BatchN  (None, 12, 12, 192)  576        ['conv2d_43[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_48 (BatchN  (None, 12, 12, 192)  576        ['conv2d_48[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_49 (BatchN  (None, 12, 12, 192)  576        ['conv2d_49[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_40 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_40[0][0]'] 
                                                                                                  
 activation_43 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_43[0][0]'] 
                                                                                                  
 activation_48 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_48[0][0]'] 
                                                                                                  
 activation_49 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_49[0][0]'] 
                                                                                                  
 mixed5 (Concatenate)           (None, 12, 12, 768)  0           ['activation_40[0][0]',          
                                                                  'activation_43[0][0]',          
                                                                  'activation_48[0][0]',          
                                                                  'activation_49[0][0]']          
                                                                                                  
 conv2d_54 (Conv2D)             (None, 12, 12, 160)  122880      ['mixed5[0][0]']                 
                                                                                                  
 batch_normalization_54 (BatchN  (None, 12, 12, 160)  480        ['conv2d_54[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_54 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_54[0][0]'] 
                                                                                                  
 conv2d_55 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_54[0][0]']          
                                                                                                  
 batch_normalization_55 (BatchN  (None, 12, 12, 160)  480        ['conv2d_55[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_55 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_55[0][0]'] 
                                                                                                  
 conv2d_51 (Conv2D)             (None, 12, 12, 160)  122880      ['mixed5[0][0]']                 
                                                                                                  
 conv2d_56 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_55[0][0]']          
                                                                                                  
 batch_normalization_51 (BatchN  (None, 12, 12, 160)  480        ['conv2d_51[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_56 (BatchN  (None, 12, 12, 160)  480        ['conv2d_56[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_51 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_51[0][0]'] 
                                                                                                  
 activation_56 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_56[0][0]'] 
                                                                                                  
 conv2d_52 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_51[0][0]']          
                                                                                                  
 conv2d_57 (Conv2D)             (None, 12, 12, 160)  179200      ['activation_56[0][0]']          
                                                                                                  
 batch_normalization_52 (BatchN  (None, 12, 12, 160)  480        ['conv2d_52[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_57 (BatchN  (None, 12, 12, 160)  480        ['conv2d_57[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_52 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_52[0][0]'] 
                                                                                                  
 activation_57 (Activation)     (None, 12, 12, 160)  0           ['batch_normalization_57[0][0]'] 
                                                                                                  
 average_pooling2d_5 (AveragePo  (None, 12, 12, 768)  0          ['mixed5[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_50 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed5[0][0]']                 
                                                                                                  
 conv2d_53 (Conv2D)             (None, 12, 12, 192)  215040      ['activation_52[0][0]']          
                                                                                                  
 conv2d_58 (Conv2D)             (None, 12, 12, 192)  215040      ['activation_57[0][0]']          
                                                                                                  
 conv2d_59 (Conv2D)             (None, 12, 12, 192)  147456      ['average_pooling2d_5[0][0]']    
                                                                                                  
 batch_normalization_50 (BatchN  (None, 12, 12, 192)  576        ['conv2d_50[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_53 (BatchN  (None, 12, 12, 192)  576        ['conv2d_53[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_58 (BatchN  (None, 12, 12, 192)  576        ['conv2d_58[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_59 (BatchN  (None, 12, 12, 192)  576        ['conv2d_59[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_50 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_50[0][0]'] 
                                                                                                  
 activation_53 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_53[0][0]'] 
                                                                                                  
 activation_58 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_58[0][0]'] 
                                                                                                  
 activation_59 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_59[0][0]'] 
                                                                                                  
 mixed6 (Concatenate)           (None, 12, 12, 768)  0           ['activation_50[0][0]',          
                                                                  'activation_53[0][0]',          
                                                                  'activation_58[0][0]',          
                                                                  'activation_59[0][0]']          
                                                                                                  
 conv2d_64 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed6[0][0]']                 
                                                                                                  
 batch_normalization_64 (BatchN  (None, 12, 12, 192)  576        ['conv2d_64[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_64 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_64[0][0]'] 
                                                                                                  
 conv2d_65 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_64[0][0]']          
                                                                                                  
 batch_normalization_65 (BatchN  (None, 12, 12, 192)  576        ['conv2d_65[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_65 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_65[0][0]'] 
                                                                                                  
 conv2d_61 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed6[0][0]']                 
                                                                                                  
 conv2d_66 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_65[0][0]']          
                                                                                                  
 batch_normalization_61 (BatchN  (None, 12, 12, 192)  576        ['conv2d_61[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_66 (BatchN  (None, 12, 12, 192)  576        ['conv2d_66[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_61 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_61[0][0]'] 
                                                                                                  
 activation_66 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_66[0][0]'] 
                                                                                                  
 conv2d_62 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_61[0][0]']          
                                                                                                  
 conv2d_67 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_66[0][0]']          
                                                                                                  
 batch_normalization_62 (BatchN  (None, 12, 12, 192)  576        ['conv2d_62[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_67 (BatchN  (None, 12, 12, 192)  576        ['conv2d_67[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_62 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_62[0][0]'] 
                                                                                                  
 activation_67 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_67[0][0]'] 
                                                                                                  
 average_pooling2d_6 (AveragePo  (None, 12, 12, 768)  0          ['mixed6[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_60 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed6[0][0]']                 
                                                                                                  
 conv2d_63 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_62[0][0]']          
                                                                                                  
 conv2d_68 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_67[0][0]']          
                                                                                                  
 conv2d_69 (Conv2D)             (None, 12, 12, 192)  147456      ['average_pooling2d_6[0][0]']    
                                                                                                  
 batch_normalization_60 (BatchN  (None, 12, 12, 192)  576        ['conv2d_60[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_63 (BatchN  (None, 12, 12, 192)  576        ['conv2d_63[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_68 (BatchN  (None, 12, 12, 192)  576        ['conv2d_68[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_69 (BatchN  (None, 12, 12, 192)  576        ['conv2d_69[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_60 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_60[0][0]'] 
                                                                                                  
 activation_63 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_63[0][0]'] 
                                                                                                  
 activation_68 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_68[0][0]'] 
                                                                                                  
 activation_69 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_69[0][0]'] 
                                                                                                  
 mixed7 (Concatenate)           (None, 12, 12, 768)  0           ['activation_60[0][0]',          
                                                                  'activation_63[0][0]',          
                                                                  'activation_68[0][0]',          
                                                                  'activation_69[0][0]']          
                                                                                                  
 conv2d_72 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed7[0][0]']                 
                                                                                                  
 batch_normalization_72 (BatchN  (None, 12, 12, 192)  576        ['conv2d_72[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_72 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_72[0][0]'] 
                                                                                                  
 conv2d_73 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_72[0][0]']          
                                                                                                  
 batch_normalization_73 (BatchN  (None, 12, 12, 192)  576        ['conv2d_73[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_73 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_73[0][0]'] 
                                                                                                  
 conv2d_70 (Conv2D)             (None, 12, 12, 192)  147456      ['mixed7[0][0]']                 
                                                                                                  
 conv2d_74 (Conv2D)             (None, 12, 12, 192)  258048      ['activation_73[0][0]']          
                                                                                                  
 batch_normalization_70 (BatchN  (None, 12, 12, 192)  576        ['conv2d_70[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_74 (BatchN  (None, 12, 12, 192)  576        ['conv2d_74[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_70 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_70[0][0]'] 
                                                                                                  
 activation_74 (Activation)     (None, 12, 12, 192)  0           ['batch_normalization_74[0][0]'] 
                                                                                                  
 conv2d_71 (Conv2D)             (None, 5, 5, 320)    552960      ['activation_70[0][0]']          
                                                                                                  
 conv2d_75 (Conv2D)             (None, 5, 5, 192)    331776      ['activation_74[0][0]']          
                                                                                                  
 batch_normalization_71 (BatchN  (None, 5, 5, 320)   960         ['conv2d_71[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_75 (BatchN  (None, 5, 5, 192)   576         ['conv2d_75[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_71 (Activation)     (None, 5, 5, 320)    0           ['batch_normalization_71[0][0]'] 
                                                                                                  
 activation_75 (Activation)     (None, 5, 5, 192)    0           ['batch_normalization_75[0][0]'] 
                                                                                                  
 max_pooling2d_3 (MaxPooling2D)  (None, 5, 5, 768)   0           ['mixed7[0][0]']                 
                                                                                                  
 mixed8 (Concatenate)           (None, 5, 5, 1280)   0           ['activation_71[0][0]',          
                                                                  'activation_75[0][0]',          
                                                                  'max_pooling2d_3[0][0]']        
                                                                                                  
 conv2d_80 (Conv2D)             (None, 5, 5, 448)    573440      ['mixed8[0][0]']                 
                                                                                                  
 batch_normalization_80 (BatchN  (None, 5, 5, 448)   1344        ['conv2d_80[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_80 (Activation)     (None, 5, 5, 448)    0           ['batch_normalization_80[0][0]'] 
                                                                                                  
 conv2d_77 (Conv2D)             (None, 5, 5, 384)    491520      ['mixed8[0][0]']                 
                                                                                                  
 conv2d_81 (Conv2D)             (None, 5, 5, 384)    1548288     ['activation_80[0][0]']          
                                                                                                  
 batch_normalization_77 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_77[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_81 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_81[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_77 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_77[0][0]'] 
                                                                                                  
 activation_81 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_81[0][0]'] 
                                                                                                  
 conv2d_78 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_77[0][0]']          
                                                                                                  
 conv2d_79 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_77[0][0]']          
                                                                                                  
 conv2d_82 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_81[0][0]']          
                                                                                                  
 conv2d_83 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_81[0][0]']          
                                                                                                  
 average_pooling2d_7 (AveragePo  (None, 5, 5, 1280)  0           ['mixed8[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_76 (Conv2D)             (None, 5, 5, 320)    409600      ['mixed8[0][0]']                 
                                                                                                  
 batch_normalization_78 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_78[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_79 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_79[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_82 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_82[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_83 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_83[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 conv2d_84 (Conv2D)             (None, 5, 5, 192)    245760      ['average_pooling2d_7[0][0]']    
                                                                                                  
 batch_normalization_76 (BatchN  (None, 5, 5, 320)   960         ['conv2d_76[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_78 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_78[0][0]'] 
                                                                                                  
 activation_79 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_79[0][0]'] 
                                                                                                  
 activation_82 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_82[0][0]'] 
                                                                                                  
 activation_83 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_83[0][0]'] 
                                                                                                  
 batch_normalization_84 (BatchN  (None, 5, 5, 192)   576         ['conv2d_84[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_76 (Activation)     (None, 5, 5, 320)    0           ['batch_normalization_76[0][0]'] 
                                                                                                  
 mixed9_0 (Concatenate)         (None, 5, 5, 768)    0           ['activation_78[0][0]',          
                                                                  'activation_79[0][0]']          
                                                                                                  
 concatenate (Concatenate)      (None, 5, 5, 768)    0           ['activation_82[0][0]',          
                                                                  'activation_83[0][0]']          
                                                                                                  
 activation_84 (Activation)     (None, 5, 5, 192)    0           ['batch_normalization_84[0][0]'] 
                                                                                                  
 mixed9 (Concatenate)           (None, 5, 5, 2048)   0           ['activation_76[0][0]',          
                                                                  'mixed9_0[0][0]',               
                                                                  'concatenate[0][0]',            
                                                                  'activation_84[0][0]']          
                                                                                                  
 conv2d_89 (Conv2D)             (None, 5, 5, 448)    917504      ['mixed9[0][0]']                 
                                                                                                  
 batch_normalization_89 (BatchN  (None, 5, 5, 448)   1344        ['conv2d_89[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_89 (Activation)     (None, 5, 5, 448)    0           ['batch_normalization_89[0][0]'] 
                                                                                                  
 conv2d_86 (Conv2D)             (None, 5, 5, 384)    786432      ['mixed9[0][0]']                 
                                                                                                  
 conv2d_90 (Conv2D)             (None, 5, 5, 384)    1548288     ['activation_89[0][0]']          
                                                                                                  
 batch_normalization_86 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_86[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_90 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_90[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_86 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_86[0][0]'] 
                                                                                                  
 activation_90 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_90[0][0]'] 
                                                                                                  
 conv2d_87 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_86[0][0]']          
                                                                                                  
 conv2d_88 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_86[0][0]']          
                                                                                                  
 conv2d_91 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_90[0][0]']          
                                                                                                  
 conv2d_92 (Conv2D)             (None, 5, 5, 384)    442368      ['activation_90[0][0]']          
                                                                                                  
 average_pooling2d_8 (AveragePo  (None, 5, 5, 2048)  0           ['mixed9[0][0]']                 
 oling2D)                                                                                         
                                                                                                  
 conv2d_85 (Conv2D)             (None, 5, 5, 320)    655360      ['mixed9[0][0]']                 
                                                                                                  
 batch_normalization_87 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_87[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_88 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_88[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_91 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_91[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 batch_normalization_92 (BatchN  (None, 5, 5, 384)   1152        ['conv2d_92[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 conv2d_93 (Conv2D)             (None, 5, 5, 192)    393216      ['average_pooling2d_8[0][0]']    
                                                                                                  
 batch_normalization_85 (BatchN  (None, 5, 5, 320)   960         ['conv2d_85[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_87 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_87[0][0]'] 
                                                                                                  
 activation_88 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_88[0][0]'] 
                                                                                                  
 activation_91 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_91[0][0]'] 
                                                                                                  
 activation_92 (Activation)     (None, 5, 5, 384)    0           ['batch_normalization_92[0][0]'] 
                                                                                                  
 batch_normalization_93 (BatchN  (None, 5, 5, 192)   576         ['conv2d_93[0][0]']              
 ormalization)                                                                                    
                                                                                                  
 activation_85 (Activation)     (None, 5, 5, 320)    0           ['batch_normalization_85[0][0]'] 
                                                                                                  
 mixed9_1 (Concatenate)         (None, 5, 5, 768)    0           ['activation_87[0][0]',          
                                                                  'activation_88[0][0]']          
                                                                                                  
 concatenate_1 (Concatenate)    (None, 5, 5, 768)    0           ['activation_91[0][0]',          
                                                                  'activation_92[0][0]']          
                                                                                                  
 activation_93 (Activation)     (None, 5, 5, 192)    0           ['batch_normalization_93[0][0]'] 
                                                                                                  
 mixed10 (Concatenate)          (None, 5, 5, 2048)   0           ['activation_85[0][0]',          
                                                                  'mixed9_1[0][0]',               
                                                                  'concatenate_1[0][0]',          
                                                                  'activation_93[0][0]']          
                                                                                                  
 global_average_pooling2d (Glob  (None, 2048)        0           ['mixed10[0][0]']                
 alAveragePooling2D)                                                                              
                                                                                                  
 batch_normalization_94 (BatchN  (None, 2048)        8192        ['global_average_pooling2d[0][0]'
 ormalization)                                                   ]                                
                                                                                                  
 dropout (Dropout)              (None, 2048)         0           ['batch_normalization_94[0][0]'] 
                                                                                                  
 dense (Dense)                  (None, 1024)         2098176     ['dropout[0][0]']                
                                                                                                  
 dropout_1 (Dropout)            (None, 1024)         0           ['dense[0][0]']                  
                                                                                                  
 dense_1 (Dense)                (None, 16)           16400       ['dropout_1[0][0]']              
                                                                                                  
==================================================================================================
Total params: 23,925,552
Trainable params: 2,118,672
Non-trainable params: 21,806,880
__________________________________________________________________""")
    
    c10 = '''STEP_SIZE_TRAIN = train_data.n // train_data.batch_size
STEP_SIZE_VALID = val_data.n // val_data.batch_size

history = model.fit_generator(train_data,
                    steps_per_epoch = STEP_SIZE_TRAIN,
                    validation_data = val_data,
                    validation_steps = STEP_SIZE_VALID,
                    epochs = 5,
                    verbose = 1)'''
    st.code(c10,language='python')
    
    st.success("""Epoch 1/5
105/105 [==============================] - 201s 2s/step - loss: 1.5508 - accuracy: 0.6561 - val_loss: 0.5987 - val_accuracy: 0.8209
Epoch 2/5
105/105 [==============================] - 262s 3s/step - loss: 0.9730 - accuracy: 0.7667 - val_loss: 0.5563 - val_accuracy: 0.8413
Epoch 3/5
105/105 [==============================] - 189s 2s/step - loss: 0.8308 - accuracy: 0.7875 - val_loss: 0.5344 - val_accuracy: 0.8534
Epoch 4/5
105/105 [==============================] - 193s 2s/step - loss: 0.7806 - accuracy: 0.7995 - val_loss: 0.5243 - val_accuracy: 0.8510
Epoch 5/5
105/105 [==============================] - 192s 2s/step - loss: 0.6273 - accuracy: 0.8302 - val_loss: 0.5381 - val_accuracy: 0.8582""")
  
    c11 = ''' # Graphical Representation
plt.xlabel('Epoch Number')
plt.ylabel('Loss')
plt.plot(history.history['loss'], label='training set')
plt.plot(history.history['val_loss'], label='test set')
plt.legend()'''
    st.code(c11,language='python')
    
    a0000,a1111,a2222 = st.columns([3,5,3])
    
    with a1111:
        st.image("./Predictions/Loss.png",width=500)
    
    
    c12 = '''plt.xlabel('Epoch Number')
plt.ylabel('Accuracy')
plt.plot(history.history['accuracy'], label='training set')
plt.plot(history.history['val_accuracy'], label='test set')
plt.legend()'''
    st.code(c12,language='python')
    
    a00000,a11111,a22222 = st.columns([3,5,3])
    
    with a11111:
        st.image("./Predictions/Accuracy.png",width=500)
    
    c13 ='''class_map = train_data.class_indices
classes = []
for key in class_map.keys():
    classes.append(key)
print(classes)'''
    st.code(c13,language='python')
    
    st.success("""['Burger', 'Chapathi', 'Dosa', 'Fried_rice', 'Idli', 'Jalebi', 'Kulfi', 'Noodles', 'Paani_puri', 'Paneer', 'Parotta', 'Pizza', 'Poori', 'Samosa', 'Soup', 'Tea']""")
    
    c14 = '''def predict_image(filename, model):
    img_ = image.load_img(filename, target_size=(228, 228))
    img_array = image.img_to_array(img_)
    img_processed = np.expand_dims(img_array, axis=0)
    img_processed /= 255.
    
    prediction = model.predict(img_processed)
    
    index = np.argmax(prediction)
    
    plt.title("Prediction - {}".format(str(classes[index]).title()), size=18, color='red')
    plt.imshow(img_array)'''
    st.code(c14,language='python')
    
    c15 = """ # Give the image path for predictions
    predict_image("D:\\test_dataset\\Fried_rice\\036.jpg",model)
"""
    st.code(c15,language='python')

    c16 = """y_pred=model.evaluate(train_data)"""
    st.code(c16,language='python')
    
    st.success("106/106 [==============================] - 148s 1s/step - loss: 0.3702 - accuracy: 0.8878")
    
    c17 = """# Save our model
    model.save("F_C.h5")"""
    st.code(c17,language='python')
    
    c18 = '''# Then we have to load the model
from tensorflow.keras.models import load_model
loaded_model = load_model("F_C.h5")'''
    st.code(c18,language='python')
    
    c19 = """# And Make prediction to the saved model
    predict_image("D:\\test_dataset\\Parotta\\001.jpg",loaded_model)"""
    
    st.code(c19,language='python')

    st.balloons()
        
    st.write("👍"*56)   