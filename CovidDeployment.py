#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import streamlit as st 
from sklearn.linear_model import WHO

st.title('Model Deployment: Covid Recovery Rate')

st.sidebar.header('Details')

def user_input_features():
    COUNTRY = st.sidebar.number_input("Insert the Country")
    POPULATION = st.sidebar.number_input("Insert Population_(in_thousands)_total")
    POPULATIONAGR = st.sidebar.number_input("Insert Population_annual_growth_rate_(%)")
    CONFIRMED = st.sidebar.number_input("Insert Confirmed")
    DEATHS = st.sidebar.number_input("Insert Deaths")
    RECOVERED = st.sidebar.number_input("Insert Recovered")
    ACTIVE = st.sidebar.number_input("Insert Active")
    NEW_CASES = st.sidebar.number_input("Insert New_cases")
    NEW_DEATHS = st.sidebar.number_input("Insert New_deaths")
    NEW_RECOVERED = st.sidebar.number_input("Insert New_recovered")
    TOTAL_RECOVERED = st.sidebar.number_input("Insert Total_Recovered")
    RECOVERED_PERCENT = st.sidebar.number_input("Insert Recovered_Percent")
    data = {'COUNTRY':COUNTRY,
            'POPULATION':POPULATION,
            'POPULATIONAGR':POPULATIONAGR,
            'CONFIRMED':CONFIRMED,
            'DEATHS':DEATHS,
            'RECOVERED':RECOVERED,
            'ACTIVE':ACTIVE,
            'NEW_CASES':NEW_CASES,
            'NEW_DEATHS':NEW_DEATHS,
            'NEW_RECOVERED':NEW_RECOVERED,
            'TOTAL_RECOVERED':TOTAL_RECOVERED,
            'RECOVERED_PERCENT':RECOVERED_PERCENT}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

Recovered_cases = pd.read_csv("Recovered_cases.csv")

X = Recovered_cases.iloc[:,[0,1,2,3,4,5,6,7,8,8,10]]
Y = Recovered_cases.iloc[:,11]
clf = MultiLinearRegression()
clf.fit(X,Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Predicted Result')

#if prediction==0:
    #st.write("Person will hire an attorney")
    
#else:
    #st.write("Person will not hire an attorney")


#st.subheader('Prediction Probability')
#st.write(prediction_proba)

