# print("Hello")

import streamlit as st

st.title('Oil Tank Detection', anchor=None)
# st.write("Hello World. I am here. How are you ??")

import os
from PIL import Image

filename = st.text_input('Enter a file path:')
try:
    # with open(filename) as input:
    #     st.text(input.read())
    image = Image.open(filename)
    st.image(image)
except FileNotFoundError:
    st.error('File not found.')
