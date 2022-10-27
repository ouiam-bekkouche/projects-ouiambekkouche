import pandas as pd #Organise les donnees dans des tables structurees
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score #evaluer notre modele (performant ou non)

heart_disease = pd.read_csv('/content/data.csv')
heart_disease.head() #afficher les 5premieres lignes
heart_disease.info() #infos sur la DataSet
heart_disease.isnull().sum() #chekcher le nombre de valers manquantes pour chaque colonne
heart_disease['target'].value_counts() #checker la distribution de la variable target
X = heart_disease.drop(columns='target',axis=1)
Y=heart_disease['target']
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
model = LogisticRegression()

model.fit(X_train,Y_train) #training the logistic model with training data
X_train_prediction = model.predict(X_train)
accuracy_training_model = accuracy_score(X_train_prediction,Y_train)
X_test_prediction = model.predict(X_test)
accuracy_test_model = accuracy_score(X_test_prediction,Y_test)
input_data = (58,0,3,150,283,1,0,162,0,1,2,0,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)
# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('The Person does not have a Heart Disease')
else:
  print('The Person has Heart Disease')

