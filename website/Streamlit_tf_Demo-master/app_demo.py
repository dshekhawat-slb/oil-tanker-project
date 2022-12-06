import streamlit as st
from PIL import Image
import numpy as np
from app_imageLoader import Imageloader
from app_pred import pred



st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", ["image","pred"])

if selection == "image":
    page = Imageloader()  #PAGES[selection]
elif selection == "pred":
    page = pred()
page.app()
