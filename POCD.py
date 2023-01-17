# -*- coding: utf-8 -*-

from PIL import Image
#from streamlit_shap import st_shap
import streamlit as st
import numpy as np 
import pandas as pd 
import time
import plotly.express as px 
import seaborn as sns
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import mean_squared_error,confusion_matrix,accuracy_score,recall_score,precision_score,classification_report,roc_auc_score
import shap
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
    a1 = st.sidebar.slider('Action1', 60.0, 100.0, 0.0)
    a2 = st.sidebar.slider('Action2', 70.0, 200.0, 0.0)
    a3 = st.sidebar.slider('Action3', 0.0, 6.0, 0.0)
    a4 = st.sidebar.slider('Action4', 0.0, 1500.0, 100.0)
    a5 = st.sidebar.slider('Action5', 0.0, 600.0, 0.0)
    a6 = st.sidebar.selectbox("Hypotension? 0=NO,1=YES", ('0', '1'))
    
    output = [a1,a2,a3,a4,a5,a6]
    return output

outputdf = user_input_features()





st.title('SHAP Value')

image4 = Image.open('SHAP.png')
shapdatadf =pd.read_excel(r'shapdatadf.xlsx')
shapvaluedf =pd.read_excel(r'shapvaluedf.xlsx')





placeholder5 = st.empty()
with placeholder5.container():
    f1,f2 = st.columns(2)

    with f1:
        st.subheader('Summary plot')
        st.write('👈 class 0: Real')
        st.write('👉 class 1: Fraud')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.image(image4)     
    with f2:
        st.subheader('Dependence plot for features')
        cf = st.selectbox("Choose a feature", (shapdatadf.columns))
        

        fig = px.scatter(x = shapdatadf[cf], 
                         y = shapvaluedf[cf], 
                         color=shapdatadf[cf],
                         color_continuous_scale= ['blue','red'],
                         labels={'x':'Original value', 'y':'shap value'})
        st.write(fig)  

catmodel = CatBoostClassifier()
catmodel.load_model("POCD")

st.title('Make predictions')
outputdf = pd.DataFrame([outputdf], columns= shapdatadf.columns)

#st.write('User input parameters below ⬇️')
#st.write(outputdf)




p1 = catmodel.predict(outputdf)[0]
p2 = catmodel.predict_proba(outputdf)


placeholder6 = st.empty()
with placeholder6.container():
    f1,f2 = st.columns(2)
    with f1:
        st.write('User input parameters below ⬇️')
        st.write(outputdf)
        st.write(f'Predicted class: {p1}')
        st.write('Predicted class Probability')
        st.write('0️⃣ means no POCD , 1️⃣ means POCD')
        st.write(p2)
    with f2:
        
        explainer = shap.Explainer(catmodel)
        shap_values = explainer(outputdf)

        #st_shap(shap.plots.waterfall(shap_values[0]),  height=500, width=1700)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        shap.plots.waterfall(shap_values[0])
        st.pyplot(bbox_inches='tight')