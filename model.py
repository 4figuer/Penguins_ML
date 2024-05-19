import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle
import time

df = pd.read_csv('penguins.csv')
print(df.head())
print('Valores nulos')
print(df.isnull().sum())
df.dropna(how='any', inplace=True)

X = df[['island','bill_length_mm',
        'bill_depth_mm','flipper_length_mm',
        'body_mass_g','sex']]
X = pd.get_dummies(X)
y = df['species']
y, uniques = pd.factorize(y)

print('\nLimpeza / Separação / Dummy ...\n')
time.sleep(4)
print(y[:5])
print(X[:5])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)


model = RandomForestClassifier(random_state=15)
print('\niniciando treinamento...\n')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
time.sleep(5)
print('\nTreinamento terminado.\n')
accuracy = accuracy_score(y_test, y_pred)
print(f'\nAcurácia do modelo: {round(accuracy*100, 2)} %\n')


with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)
f.close()
print('Modelo salvo no diretório.')
with open('output.pkl', 'wb') as j:
        pickle.dump(uniques, j)
j.close()
print('Output salvo no diretório.')


