# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Fijamos la semilla de las componentes aleatorias
np.random.seed(42)

#---------------------------------------------------------------------------------#
#------------------------- Lectura de datos y preprocesado -----------------------#
#---------------------------------------------------------------------------------#

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
input("\n--- Pulsar tecla para continuar ---\n")

# Pasar datos de TRAINING a pandas para exploración

cabecera = ['C' + str(i) for i in range(X_train.shape[1])]    # Formateo de columnas
cabecera.append('letra')
dataTrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train.reshape(-1, 1))], axis = 1)
dataTrain.set_axis(cabecera, axis=1, inplace=True)

# Exploración y análisis de datos
"""
    Exploración y análisis de datos
    Histograma de frecuencia de clase: sirve para visualizar y detectar clases desbalanceadas
    Boxplot de cada característica: información visual estadística de cada característica
                                    útil para ver visualmente los distintos rangos en los que se mueven las características
    Boxplot tras estandarización: quizás debería estar en el apartado de procesado?
    Matriz de correlaciones: información visual de la correlación entre variables
    Estadísticos de cada característico: información estadística
""" 

# Histograma de frecuencia de cada clase
plt.figure(figsize = (9, 6))
ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
ax.set_xlabel("Clases")
ax.set_ylabel("Frecuencia")
plt.title("Histograma de frecuencia")
plt.show()

print("Histograma de frecuencias agrupado por clases")
input("\n--- Pulsar tecla para continuar ---\n")

# Boxplot de las distintas características
plt.figure(figsize = (20, 10))
plt.xlabel("Características del problema")
boxplot = dataTrain[dataTrain.columns[0:-1]].boxplot()
plt.title("Boxplot de cada característica")
plt.show()

print("Boxplot para cada característica del problema")
input("\n--- Pulsar tecla para continuar ---\n")

# Muy poca información visual debido a valores anormales
# Hay que estandarizar los datos
scaler = sk.preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
dataTrainScaled = pd.DataFrame(np.concatenate((X_train_scaled, y_train.reshape(-1,1)), axis = 1),
                               columns = dataTrain.columns)
                            

# Boxplot de las distintas variables continuas tras estandarización
plt.figure(figsize = (20, 10))
plt.xlabel("Variables del problema")
boxplot = dataTrainScaled[dataTrainScaled.columns[0:-1]].boxplot()
plt.title("Boxplot de cada variable tras estandarización")
plt.show()

print("Boxplot para cada característica del problema tras estandarización")
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar gráfica de la matriz de correlaciones
f = plt.figure(figsize=(22, 15))
plt.matshow(dataTrain[dataTrain.columns[0:-1]].corr(), fignum=f.number)
plt.xticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=14, rotation=45)
plt.yticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)
plt.show()

print("Gráfica de la matriz de correlaciones")
input("\n--- Pulsar tecla para continuar ---\n")