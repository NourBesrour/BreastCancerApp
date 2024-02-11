import streamlit as st
import numpy as np
import pandas as pd
import pickle
from main import LR

# Charger le modèle
with open("xgbpipe.joblib", "rb") as f:
    model = pickle.load(f)

st.title("Benign or Malignant ? ")
st.image("BC.png")

st.markdown(" Whether the patient is pro or postmenopausal at the time diagnose,0 MEANS THAT THE PATIENT HAS REACHED MENOPAUSE WHILE 1 MEANS THAT THE PATIENT HAS NOT REACHED MENOPAUSE YET.")
menopause = st.select_slider("Menopause", (0, 1))

st.markdown(" The number of axillary lymph nodes that contain metastatic, CODED AS A BINARY DISTRI UTION OF EITHER PRESENT OR ASENT. 1 MEANS PRESENT, 0 MEANS ABSENT")
IN = st.select_slider("involved nodes", (0, 1))

st.markdown(" If it occurs on the left or right side,CODED AS A BINARY DISTRIBUTION 1 MEANS THE CANCER HAS SPREAD, 0 MEANS IT HASN'T SPREAD YET.")
breast = st.select_slider("breast", (0, 1))

st.markdown("If the cancer has spread to other part of the body or organ.")
metastatic = st.select_slider("metastatic", (0, 1))

st.markdown(" The gland is divided into 4 sections with nipple as a central point")
BQ = st.select_slider("breast quadrant", (0, 1))

st.markdown("If the patient has any history or family history on cancer,1 means there is a history of cancer , 0 means no history")
History = st.select_slider("History", (0, 1))

st.markdown("age : age of the patient at the time of diagnose ")
age = st.number_input("age")

TS = st.number_input("tumor size")

columns = ['Age','Menopause','TS','Inv-Nodes','Breast','Metastasis','BQ','History']

def predict():
    row = np.array([menopause, IN, breast, metastatic, BQ, History, age, TS])
    X = pd.DataFrame([row], columns=columns)
    prediction = LR.predict(X)[0]
    if prediction == 1:
        st.success("Malignant")
    else:
        st.error("Benign")

st.button('Predict',on_click=predict)