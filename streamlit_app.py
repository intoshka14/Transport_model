import streamlit as st
from fastai.vision.all import *

import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
# title
st.title('Transportni klassifikatsiya qiluvchi model')

# rasmni joylash
files=st.file_uploader("Rasm yuklash", type=["avif","png","jpeg","gif","svg"])
if files:
    st.image(files) # rasmni chiqarish
    # PIL convert
    img=PILImage.create(files)
    model = load_learner('transport_model.pkl')
    
    # Bashorat qiymatni topamiz
    pred,pred_id, probs =model.predict(img)
    st.success(f"Bashorat: {pred}")
    st.info(f"Ehtimollik: {probs[pred_id]*100:.1f}%")

    # plotting
    fig=px.bar(x=probs*100, y=model.dls.vocab)
    st.plotly_chart(fig)
