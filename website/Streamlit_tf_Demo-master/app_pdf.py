import streamlit as st
import numpy as np

st.title("PDF save")
st.markdown("Load pdf file and save it again")

## Image Loader
uploaded_file = st.file_uploader("Gimme pdf", type=["pdf"])
res = None
image=None

if uploaded_file:
    st.write(uploaded_file)
