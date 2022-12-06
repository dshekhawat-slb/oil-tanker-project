import streamlit as st
import numpy as np
from PIL import Image
import os


# with open( "website/style.css" ) as css:
#     st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.set_page_config(layout="wide")

st.title("🕵️‍♂️ Oil Tanker Detection")
# col1, col2, col3, col4, col5, = st.columns([5,1,1,1,1])
# # col3, col4, col5, col6 =st.columns(4)
# with col1:
#     st.title("🕵️‍♂️ Oil Tanker Detection")
# with col2:
#     st.write("Credits")
#     img1 = Image.open('website/credits/115072063.jfif')
#     st.image(img1, width=20)
# with col3:
#     img2 = Image.open('website/credits/116726399.jfif')
#     st.image(img2,width=20)

st.markdown("""
<style>
.big-font {
    font-size:25px !important;
}
.block-container {
    padding: 3rem 4rem 10rem !important;
}
div[class="css-1kyxreq etr89bj2"] {
    margin-top: 37px;
}
div[class="css-1a32fsj e19lei0e0"]{
    margin-top: 25px;
}
div[data-testid="stImage"] {
    margin-left: 65px;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font"> ❗❗❗  Predict oil tank in a 512 ❌ 512 satellite image ❗❗❗ </p>',
            unsafe_allow_html= True)

### custom loss ###
import tensorflow as tf

def custom_loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)

        return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    o = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred) + dice_loss(y_true, y_pred)
    return tf.reduce_mean(o)

from tensorflow.keras.models import load_model

@st.cache(allow_output_mutation=True)
def load_model_cache():

    # model = build_autoencoder() #build empty model with the right architecure

    path_folder = os.path.dirname(__file__)#Get Current directory Path File

    model_path = 'models/cp_model_11_3.h5'

    # model.load_weights(os.path.join(path_folder,model_path)) #load weights only from h5 file
    model = load_model(model_path, custom_objects={'custom_loss': custom_loss})

    return model
model = load_model_cache()

## Image Loader
uploaded_file = st.file_uploader("Upload 512*512 image", type=["png", "jpg", "jpeg"])
res = None
image=None

from tensorflow.image import resize
def load_test(img_path):
    im_test = Image.open(img_path)
    im_test_arr = np.array(im_test)/255
    im_test_arr_resized = resize(im_test_arr, (256,256))
    im_test_arr_resized_expand_dim = np.expand_dims(im_test_arr_resized, axis = 0)
    #     plt.imshow(im_test)
    return im_test_arr_resized_expand_dim

import cv2
def get_final_pred_image(img_test_path):
    img_raw = np.array(Image.open(img_test_path))
    #     np.stack(X)/255.
    X_test_single_img = load_test(img_test_path)
    im_pred = model.predict(X_test_single_img)
    im_pred = resize(im_pred, (512,512))
    im_pred = im_pred[0]
    image_8bit = np.uint8(im_pred * 255)
    ret, thresh = cv2.threshold(image_8bit, 127,255,0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        contours, hierarchy = contours, hierarchy[0]
    else:
        contours, hierarchy = contours, None
    if len(contours)>0:
        for contour, hierarchy in zip(contours, hierarchy):
            if hierarchy[-1] == -1 and len(contour)>10:
                x,y,w,h = cv2.boundingRect(contour)
                cv2.rectangle(img_raw,(x,y),(x+w,y+h),(255,0,0),2)
    return img_raw


if uploaded_file:
    img = Image.open(uploaded_file)
    # img = image.convert("P") #convert PNG to 1channel image
    # img = img.resize((28,28)) #resize (28,28)
    col1, col2 = st.columns(2)

    # st.write("Selected Image")
    # button_pred = st.button('Predict')
    with col1:
        st.write("Selected Image")
        button_pred = st.button('Predict')
        st.image(img, width=400)

    if button_pred :
        img_pred_final = get_final_pred_image(uploaded_file)
        with col2:
            st.write("Voila!!!!!!!!!!!!!!!!")
            st.write("Here, is our prediction of oil tankers")
            st.image(img_pred_final, width=405)

    # imgArray = np.array(img).reshape(28,28,1) /255. #convert/resize and reshape PNG file to get one channel + rescale
    # if not imgArray.sum() >0:
    #     image = None
    #     st.write("Invalid Image")
