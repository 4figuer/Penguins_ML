
import pickle
import streamlit as st 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

import streamlit as st

penguins_file = st.file_uploader('Caso queira, carregue os dados:')
if penguins_file is None:
    with open('model.pkl', 'rb') as file:
        modelo = pickle.load(file)
    file.close()

    with open('output.pkl', 'rb') as file:
        penguins_map = pickle.load(file)
    file.close()
else:
    df = pd.read_csv(penguins_file)
    df = df.dropna()
    output = df['species']
    features = df[['island', 'bill_length_mm', 'bill_depth_mm',
                           'flipper_length_mm', 'body_mass_g', 'sex']]
    
    X = pd.get_dummies(features)
    y, penguins_map = pd.factorize(output)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestClassifier(random_state=15)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    score = round(accuracy*100, 2)
    st.subheader('Dados Carregados!')
    time.sleep(2)
    st.subheader('Treinando o modelo...')
    time.sleep(5)
    
    st.write(
        f"""We trained a Random Forest model on these
        data, it has a score of {score} % ! Use the
        inputs below to try out the model"""
    )
    
    fig, ax = plt.subplots()
    ax = sns.barplot(x=modelo.feature_importances_, y=X.columns)
    plt.title('As variáveis mais importantes \npara a predição da espécie são:')
    plt.xlabel('Importância')
    plt.ylabel('Variáveis')
    plt.tight_layout()
    fig.savefig('feature_importance.png')

    
    

with st.form('user_inputs'):
    island = st.selectbox('Qual Ilha?', options=["Biscoe", "Dream", "Torgerson"])
    sex = st.selectbox("Sex", options=["Female", "Male"])
    bill_length = st.number_input("Bill Length (mm)", min_value=0)
    bill_depth = st.number_input("Bill Depth (mm)", min_value=0)
    flipper_length = st.number_input("Flipper Length (mm)", min_value=0)
    body_mass = st.number_input("Body Mass (g)", min_value=0)
    st.form_submit_button()

island_biscoe, island_dream, island_torgeron = 0,0,0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgeron = 1
    
sex_male, sex_female = 0,0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1
    
  
user_input = [island, sex, bill_length, bill_depth, flipper_length, body_mass]

st.write(f'the user inputs are {user_input}')

new_prediction = modelo.predict([[bill_length,
                                 bill_depth,
                                 flipper_length,
                                 body_mass,
                                 island_biscoe,
                                 island_dream,
                                 island_torgeron,
                                 sex_female,
                                 sex_male]])

prediction_species = penguins_map[new_prediction][0]
st.write(f'Nossa predição nos diz que o Pinguim é da espécie: {prediction_species}')
st.image('feature_importance.png')
