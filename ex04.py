#Escolha cinco variáveis e
# as utilize comovariáveis independentes
# para a construção de um modelode Regressão Linear Múltipla.
# Apredição deve ter como alvo avariável RainTomorrow
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('weatherAUS.csv')
#dataset inteiro
##print(dataset)
#duas primeiras linhas, por exemplo
##print(dataset[0:5])


independent = dataset[['Location','MinTemp','MaxTemp','Rainfall','WindGustSpeed','RainTomorrow']]
dependent = dataset[['RainTomorrow']]


##print(dataset.iloc[0:1,:])

##print(dataset.iloc[0:6,0])

transformer = ColumnTransformer(transformers=[('encoder',
    OneHotEncoder(), [len])], remainder='passthrough')
independent = np.array((transformer.fit_transform(independent)))

print(independent)


ind_train, ind_teste ,dep_train, dep_test = train_test_split(ind(independent, dependent, 
        teste_size=0.6, randon_states =0)


linearRegression.fit(ind_train, dep_train)


dep_pred = linearRegression.predict(ind_test)


np.set_printoptions(precision=3)

print (np.concatenate((dep_pred.reshape(len(dep_pred), 1),
    dep_test.reshape(len(dep_pred),1)),axis=1))