import streamlit as st
import numpy as np
from PIL import Image
import os
from model_utils import build_autoencoder

st.title("MNIST Denoizing")
st.markdown("Predict a number and denoize image")

mnist_class_names = ["zero","one","two","three","four","five","six","seven","eight","nine"]

@st.cache(allow_output_mutation=True) #cache the model at first loading
def load_model_cache():

    model = build_autoencoder() #build empty model with the right architecure

    path_folder = os.path.dirname(__file__)#Get Current directory Path File
    model.load_weights(os.path.join(path_folder,"autoencoder_weights.h5")) #load weights only from h5 file

    return model

model = load_model_cache()

## Image Loader
uploaded_file = st.file_uploader("Gimme image", type=["png", "jpg", "jpeg"])
res = None
image=None

if uploaded_file:
    image = Image.open(uploaded_file)
    img = image.convert("P") #convert PNG to 1channel image
    img = img.resize((28,28)) #resize (28,28)
    st.image(img)
    imgArray = np.array(img).reshape(28,28,1) /255. #convert/resize and reshape PNG file to get one channel + rescale
    if not imgArray.sum() >0:
        image = None
        st.write("Invalid Image")

    if st.button('Predict'):
        # Send to API, endpoint must accept POST
        if image is not None:

            classif, res_img = model.predict(np.expand_dims(imgArray,axis=0))
            #classif--> softmax from classification prediction
            #img --> image array from generator part of the model

            #-----------Classification Problem-------
            class_idx = np.argmax(classif) #get softmax prediction
            st.write(f"Predicted number:   {mnist_class_names[class_idx].upper()}")
            #----------------------------------------

            #Image Generation Problem ---------------
            img = res_img[0] #batch of 1 prediction ---> extract element 0
            res = Image.fromarray(img.reshape(28,28)*255).convert("L") # Convert image Array to PIL Image
            st.write("Here is the Denoized Image:")
            st.image(res)

            #----------------------------------------
