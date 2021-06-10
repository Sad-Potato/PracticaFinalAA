# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split










# Fijamos la semilla de las 
# componentes aleatorias
np.random.seed(42)

#---------------------------------------------------------------------------------#
#------------------------- Lectura de datos y preprocesado -----------------------#
#---------------------------------------------------------------------------------#

# Lectura de la base de datos

# Lectura de la base de datos

Data=np.loadtxt("./datos/letter-recognition.data",dtype=str,delimiter=",")

X=np.array(Data[:,1:]).astype(np.int32)
Yn=Data[:,0] # Yn letras que corresponden a la clase
Y=np.empty(np.size(Data[:,0])) # Clases como numeros 
for i in np.arange(np.size(Yn)):
    Y[i]=ord(Yn[i])-65

# Dividimos los datos en training y test
'''
    Dividimos en training y test, con un tamaño de test del 20% de la muestra, 
    un valor de pseudo-aleatoriedad, barajamos la muestra porque no sabemos como viene el 
    fichero con los datos y usamos stratify para que no haya clases infrarepresentadas.

'''
X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)
print("Tamaños de la bd:","\n\tTrain: ",np.size(X_train[:,0]),"\n\tTest: ",np.size(X_test[:,0]),"\n\tTotal: ",np.size(Y[:]))



cabecera = ['letra']
cabecera = cabecera + ['C' + str(i) for i in range(X_train.shape[1]-1)]

prueba = pd.DataFrame(y_train.reshape(-1, 1), columns = ['letra'])

# Histograma de frecuencia de cada clase
plt.figure()
ax = prueba['letra'].plot.hist(bins = np.arange(27) -0.5, ec = 'black', xticks=np.arange(26))
ax.set_xlabel("Clases")
ax.set_ylabel("Frecuencia")
plt.title("Histograma de frecuencia")
plt.show()
