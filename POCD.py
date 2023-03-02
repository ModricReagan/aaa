# -*- coding: utf-8 -*-

from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import shap
import pickle
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import catboost
from catboost import CatBoostClassifier
import plotly.figure_factory as ff
import matplotlib.pyplot as plt

plt.style.use('default')

st.set_page_config(
    page_title = 'Machine learning: POCD risk prediction at 3 months after surgery',
    page_icon = '🕵️‍♀️',
    layout = 'wide'
)

# dashboard title
#st.title("Real-Time Fraud Detection Dashboard")
st.markdown("<h1 style='text-align: center; color: black;'>机器学习： 术后3个月POCD风险预测</h1>", unsafe_allow_html=True)
st.markdown("<h1 style='text-align: center; color: black;'>Machine learning: POCD risk prediction at 3 months after surgery</h1>", unsafe_allow_html=True)

# side-bar 
def user_input_features():
    st.sidebar.header('Make a prediction')
    st.sidebar.write('User input parameters below ⬇️')
    a1 = st.sidebar.slider('Age', 60.0, 100.0, 0.0)
    a2 = st.sidebar.slider('Hb', 70.0, 200.0, 0.0)
    a3 = st.sidebar.slider('VAS_Score', 0.0, 6.0, .0)
    a4 = st.sidebar.slider('Blood_loss', 0.0, 1500.0, 0.0)
    a5 = st.sidebar.slider('Surgery_duration', 0.0, 600.0, 0.0)
    a6 = st.sidebar.selectbox("Hypotension? 0=NO,1=YES", ('0', '1'))
    
    output = [a1,a2,a3,a4,a5,a6]
    return output

outputdf = user_input_features()


# understand the dataset
df = pd.read_csv('60岁3个月lasso.csv')







shapdatadf =pd.read_excel(r'shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf.xlsx')






postmodel = pickle.load(open("postmodel.pkl","rb"))

st.title('Make predictions')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)







p1 = postmodel.predict(outputdf)[0]
p2 = postmodel.predict_proba(outputdf)
p3 = p2[:,1]*100

st.write(p2)  
st.write(f'Predicted class: {p3*100}')
st.write(p3)
