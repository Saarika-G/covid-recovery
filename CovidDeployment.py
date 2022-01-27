#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas as pd
import streamlit as st 
from sklearn.linear_model import LinearRegression

st.title('Model Deployment: Covid Recovery Rate')

st.sidebar.header('Details')

def user_input_features():
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
    data = {'POPULATION':POPULATION,
            'POPULATIONAGR':POPULATIONAGR,
            'CONFIRMED':CONFIRMED,
            'DEATHS':DEATHS,
            'RECOVERED':RECOVERED,
            'ACTIVE':ACTIVE,
            'NEW_CASES':NEW_CASES,
            'NEW_DEATHS':NEW_DEATHS,
            'NEW_RECOVERED':NEW_RECOVERED,
            'TOTAL_RECOVERED':TOTAL_RECOVERED}
    f = pd.DataFrame(data,index = [0])
    return f
    
df = user_input_features()
st.subheader('User Input parameters')
st.write(df)

Recovered_cases = pd.read_csv("Recovered_Cases.csv")

X = Recovered_cases.iloc[:,[1,2,3,4,5,6,7,8,8,10]]
Y = Recovered_cases.iloc[:,11]
clf = LinearRegression()
clf.fit(X,Y)

Recovered_Percent = clf.predict(df)

st.subheader('Prediction Recovered precentage')
st.write(Recovered_Percent)


# In[ ]:




