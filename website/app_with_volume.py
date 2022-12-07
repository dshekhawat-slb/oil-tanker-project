import streamlit as st
import numpy as np
from PIL import Image
import os


# with open( "website/style.css" ) as css:
#     st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

st.set_page_config(layout="wide")

st.title("🕵️‍♂️ Oil Tank Detection")
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
if "pred_button" not in st.session_state:
    st.session_state["pred_button"]=False
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


#  ### new function below ####

import cv2

def get_tanks(img_test_path):
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



from skimage import morphology
from skimage import color
# im_test = Image.open(img_path)
import cv2
def get_final_pred_image(img_test_path,model, draw_bbox=True):
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

    l_contours=[]
    if len(contours)>0:
        for contour, hierarchy in zip(contours, hierarchy):
            if hierarchy[-1] == -1 and len(contour)>30:
                x,y,w,h = cv2.boundingRect(contour)
                #if draw_bbox:
                #  cv2.rectangle(img_raw,(x,y),(x+w,y+h),(255,0,0),2)
                l_contours.append((x,y,w,h))

    return img_raw, l_contours

# im_pred_final, l_contours = get_final_pred_image(img_path)
def find_fill_percent(img_cropped):
    lab = cv2.cvtColor(img_cropped, cv2.COLOR_RGB2LAB).astype(np.float32)
    l1 = lab[:,:,0]*100/255
    a = lab[:,:,1]-128
    b = lab[:,:,2]-128

    # lab[:,:,0] = lab[:,:,0]*100/255
    # lab[:,:,1]= lab[:,:,1]-128
    # lab[:,:,2] = lab[:,:,2]-128
    # #calculation
    # mean_a = np.mean(a)
    # mean_b = np.mean(b)
    # #mask creation
    # mask = np.zeros(l1.shape)
    # mask[l1<np.mean(l1)-np.std(l1)/3] = 1

    # #mask = cv2.inRange(lab, (0, -128, -128), (np.mean(l1)-np.std(l1)/3, 127, np.mean(b)-np.std(b)/3))




    # #morphological operation
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    # erosion= cv2.erode(mask,kernel)
    # #morphological operation
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(6,6))
    # dilated = cv2.dilate(erosion, kernel)
    # mask_8bit = np.uint8(dilated*255)
    # #find contours of shadow

    lab = np.stack([l1,a,b], axis = 2).astype('int16')
    treshold_mean = 9
    if (np.mean(a) + np.mean(b)) < treshold_mean:
      frame_threshold = np.zeros(l1.shape)
      frame_threshold[l1<np.mean(l1)-np.std(l1)/3] = 1
    else:
      frame_threshold = cv2.inRange(lab, (0, -128, -128), (np.mean(l1)-np.std(l1)/3,
                                                     127, np.mean(b)-np.std(b)/3))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    cv2.morphologyEx(frame_threshold,cv2.MORPH_CLOSE,kernel,frame_threshold)
    cv2.morphologyEx(frame_threshold,cv2.MORPH_OPEN,kernel,frame_threshold)
    #erosion= cv2.erode(frame_threshold,kernel)
    #morphological operation
    #kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(4,4))
    #dilated = cv2.dilate(frame_threshold, kernel)
    mask_8bit = np.uint8(frame_threshold*255)



    contours_shadow, hierarchy_shadow = cv2.findContours(mask_8bit, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #area per contour for every cropped tank image
    contour_area = []
    for contour_shadow in contours_shadow:
      area = cv2.contourArea(contour_shadow)
      contour_area.append((area,contour_shadow))
    contour_area.sort(key = lambda x: x[0], reverse=True)

    if len(contour_area) > 1:
      outer_area,outer_contour = contour_area[0]
      inner_area,inner_contour = contour_area[1]
      tank_fill_perc = round((1-inner_area/outer_area)*100, 0)
    else:
      tank_fill_perc=-1
      inner_contour=[]
      outer_contour=[]

    return tank_fill_perc,outer_contour,inner_contour


def final_image(img_path,model, bbox=True,shadow=True, volume_est=True):
  im_pred_final, l_contours = get_final_pred_image(img_path, model)
  im_mask=im_pred_final*0
  #multi_tank_contour_area = []
  for index, item in enumerate(l_contours):
    x,y,w,h = l_contours[index]
    x_min,x_max,y_min,y_max = max(int(y-0.3*h),0),min(int(y+1.3*h),512), max(int(x-0.3*w),0),min(int(x+1.3*w),512)
    img_cropped = np.array(im_pred_final)[x_min:x_max,y_min:y_max,:]
    img_cropped_mask =img_cropped*0
    #RGB to LAB conversion
    tank_fill_perc,outer_contour,inner_contour = find_fill_percent(img_cropped)

    # img_cropped_mask = cv2.fillPoly(img_cropped_mask,[outer_contour],(255,0,0))
    # img_cropped_mask = cv2.fillPoly(img_cropped_mask,[inner_contour],(0,255,0))

    if len(outer_contour)>0:
      img_cropped_mask = cv2.fillPoly(img_cropped_mask,[outer_contour],(255,0,0))
    if len(inner_contour)>0:
      img_cropped_mask = cv2.fillPoly(img_cropped_mask,[inner_contour],(0,255,0))


    im_mask[x_min:x_max,y_min:y_max,:]=img_cropped_mask

    if bbox:
      cv2.rectangle(im_pred_final,(x,y),(x+w,y+h),(255,0,0),2)

    if tank_fill_perc!=-1 and volume_est:
      final_image_label = cv2.putText(img = im_pred_final, text = str(int(tank_fill_perc))+'%', org = (int(x+w/4), int(y+h/2)),
    fontFace = cv2.FONT_HERSHEY_DUPLEX, fontScale = 0.9, color = (225, 0, 0), thickness = 2)

  if shadow:
      alpha=0.8
      image_new = cv2.addWeighted(im_pred_final, alpha, im_mask, 1 - alpha, 0)
  else:
      image_new = im_pred_final
  return image_new

if uploaded_file is None:
    st.session_state["pred_button"]=False

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
        st.session_state["pred_button"]=True

    if st.session_state["pred_button"]:
        # img_pred_final = final_image(uploaded_file, model, bbox = False)
        #img_pred_final = get_tanks(uploaded_file)
        with col2:
            st.write("Voila!! Here, is our prediction of oil tanks.")
            checkbox_pred_bbox = st.checkbox('Predict Bbox',value=True)
            checkbox_pred_shadow = st.checkbox('Predict Shadows',value=False)
            checkbox_pred_volume = st.checkbox('Predict Volume in Tanks',value=False)
            #st.image(img_pred_final, width=405)

            img_pred_final_vol = final_image(uploaded_file,
                                             model,
                                             bbox=checkbox_pred_bbox,
                                             shadow=checkbox_pred_shadow,
                                             volume_est=checkbox_pred_volume)

            st.image(img_pred_final_vol, width=405)


    # imgArray = np.array(img).reshape(28,28,1) /255. #convert/resize and reshape PNG file to get one channel + rescale
    # if not imgArray.sum() >0:
    #     image = None
    #     st.write("Invalid Image")
