import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import r2_score, accuracy_score

st.title('Data Analysis Web App')

st.sidebar.header('Upload Data')
uploaded_file = st.sidebar.file_uploader('Choose a CSV file', type='csv')
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info('Using sample data')
    data = st.sidebar.selectbox('Select sample dataset', ('Titanic', 'Tips', 'Diamonds'))
    if data == 'Titanic':
        df = sns.load_dataset('titanic')
    elif data == 'Tips':
        df = sns.load_dataset('tips')
    else:
        df = sns.load_dataset('diamonds')

if st.sidebar.checkbox('Show data'):
    st.subheader('Raw data')
    st.write(df)

if st.sidebar.checkbox('EDA'):
    st.subheader('Exploratory Data Analysis')
    
    st.markdown('#### Correlation matrix')
    st.write(df.corr())
    
    st.markdown('#### Summary statistics')
    st.write(df.describe().T)
    
    st.markdown('#### Info')
    st.write(df.info())
    
    st.markdown('#### Sample rows')
    st.write(df.head())

if st.sidebar.checkbox('Impute missing values'):
    impute = st.sidebar.selectbox('Imputation method', ('Mean', 'Median', 'Mode'))
    if impute == 'Mean':
        df = df.fillna(df.mean())
    elif impute == 'Median':
        df = df.fillna(df.median())
    else:
        df = df.fillna(df.mode().iloc[0])
        
if st.sidebar.checkbox('Plot data'):
    x = st.sidebar.selectbox('X axis', df.columns)
    y = st.sidebar.selectbox('Y axis', df.columns)
    plot_type = st.sidebar.selectbox('Plot type', ['Scatter', 'Line', 'Bar', 'Box', 'Histogram'])
    
    if plot_type == 'Scatter':
        plt.scatter(df[x], df[y])
    elif plot_type == 'Line':
        plt.plot(df[x], df[y])
    elif plot_type == 'Bar':
        plt.bar(df[x], df[y])
    elif plot_type == 'Box':
        plt.boxplot(df[y])
    else:
        plt.hist(df[y])
        
    st.pyplot()
    
    if st.sidebar.checkbox('Download plot'):
        # Handle plot download
        st.subheader('Statistical Analysis')
        st.write(df.describe())

st.subheader('Machine Learning')

X = st.selectbox('Select feature columns', df.columns)
y = st.selectbox('Select target column', df.columns)

encode = st.selectbox('Encoding', ['None', 'Label', 'One-Hot'])
if encode == 'Label':
   # Handle label encoding  
   print("Hello")
elif encode == 'One-Hot':
    # Handle one-hot encoding
    print("Hellp")

split = st.slider('Train-test split ratio (% for train)', 10, 90, 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split/100)

task = st.selectbox('Task', ['Regression', 'Classification'])

if task == 'Regression':
    models = ['Linear Regression', 'Random Forest Regressor']
    model = st.selectbox('Algorithm', models)
    
    if model == models[0]:
        lm = LinearRegression()
        lm.fit(X_train, y_train)
        y_pred = lm.predict(X_test)
        score = r2_score(y_test, y_pred)
    else:
        rf = RandomForestClassifier()
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        
else:
    models = ['Logistic Regression', 'Random Forest Classifier']
    model = st.selectbox('Algorithm', models)
    
    if model == models[0]:
        # Fit logistic regression
        score = accuracy_score(y_test, y_pred) 
    else:
        # Fit random forest classifier
        score = accuracy_score(y_test, y_pred)
        
st.write('{} score: {}'.format(model, score))

if st.button('Download trained model'):
    # Handle model download
    print("HEllo")