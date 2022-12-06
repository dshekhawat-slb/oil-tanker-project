import streamlit as st
from PIL import Image
import numpy as np

class Imageloader():

    def app(self):
        st.title("Streamlit first Demo")
        st.text("Test text function")

        uploader = st.file_uploader(
            'File uploader Button',
            type=["png", "jpg", "tiff", "jpeg"],
        )

        if uploader is not None:
            im = Image.open(uploader)

            arr_im = np.array(im)
            st.image(arr_im)

            arr_transformed = arr_im
            st.image(arr_im)

if __name__=='__main__':
    app = Imageloader()
    app.app()
