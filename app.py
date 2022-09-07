import streamlit as st
from fastai.vision.all import *
import pathlib
import plotly.express as px
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath


st.title("Barcha maxluqotlar haqida (yirtqich hayvonlar, qisqichbaqasimonlar, sutemizuvchi ...)ni aniqlovchi Web DS app.")

st.write("""
# Misol uchun:
baliq, sher yoki ayiq rasmlaridan birini yuklang.
""")


file = st.file_uploader('Rasm yuklash', type=['png','jpg','gif','jpeg','svg'])
if file:
    st.image(file)

    img = PILImage.create(file)

    model = load_learner('type_of_animals.pkl')

    pred, pred_id, probs = model.predict(img)
    st.success(f"Bashorat:{pred}")
    st.info(f"Ehtimollik:{probs[pred_id]*100:.1f}%")

    fig = px.bar(x=probs*100, y=model.dls.vocab)
    st.subheader('Classlarga tushish diogrammasi:')
    st.plotly_chart(fig)