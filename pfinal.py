# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Modelos
from sklearn.linear_model import LogisticRegression

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


# Eliminamos datos sin variabilidad si los hay (data snooping?)

selector = VarianceThreshold()
X=selector.fit_transform(X,Y)

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


# Eliminamos outliers
"""
    Eliminamos valores extremos, es decir instancias en cuales no un
    atributo esta "alejado" de los demas si no que en su conjunto esta 
    alejada del resto de instancias
"""
# OUTLIERS CON LOCAL OUTLIER FACTOR

# clf = LocalOutlierFactor(contamination=0.05) # Eliminamos como maximo 5% de outliers o probar con contamination='auto'?
# outliers=np.where(clf.fit_predict(X_train)==-1)
# X_train=np.delete(X_train,outliers,0)
# y_train=np.delete(y_train,outliers,0)



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
# plt.figure(figsize = (9, 6))
# ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
# ax.set_xlabel("Clases")
# ax.set_ylabel("Frecuencia")
# plt.title("Histograma de frecuencia")
# plt.show()

# print("Histograma de frecuencias agrupado por clases")
# input("\n--- Pulsar tecla para continuar ---\n")

# # Boxplot de las distintas características
# plt.figure(figsize = (20, 10))
# plt.xlabel("Características del problema")
# boxplot = dataTrain[dataTrain.columns[0:-1]].boxplot()
# plt.title("Boxplot de cada característica")
# plt.show()

# print("Boxplot para cada característica del problema")
# input("\n--- Pulsar tecla para continuar ---\n")

# Muy poca información visual debido a valores anormales
# Hay que estandarizar los datos
scaler = sk.preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
dataTrainScaled = pd.DataFrame(np.concatenate((X_train_scaled, y_train.reshape(-1,1)), axis = 1),
                               columns = dataTrain.columns)

# normalizamos tambien el test?¿
X_test=scaler.transform(X_test)
                            

# # Boxplot de las distintas variables continuas tras estandarización
# plt.figure(figsize = (20, 10))
# plt.xlabel("Variables del problema")
# boxplot = dataTrainScaled[dataTrainScaled.columns[0:-1]].boxplot()
# plt.title("Boxplot de cada variable tras estandarización")
# plt.show()

# print("Boxplot para cada característica del problema tras estandarización")
# input("\n--- Pulsar tecla para continuar ---\n")

# # Mostrar gráfica de la matriz de correlaciones
# f = plt.figure(figsize=(22, 15))
# plt.matshow(dataTrain[dataTrain.columns[0:-1]].corr(), fignum=f.number)
# plt.xticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=14, rotation=45)
# plt.yticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=14)
# cb = plt.colorbar()
# cb.ax.tick_params(labelsize=14)
# plt.title('Correlation Matrix', fontsize=16)
# plt.show()

# print("Gráfica de la matriz de correlaciones")
# input("\n--- Pulsar tecla para continuar ---\n")


# Tecnica de reduccion de dimensionalidad PCA / ?¿ Solo si me mejoran los resultados (guion)
"""
    Usamos la tecnica de PCA, la idea detras de esto es reducir el 
    número de variables manteniendo la maxima cantidad de informacion.
    En nuestro caso aplicamos PCA manteniendo un 95% de la varianza de los datos

"""

print("\n--------- PCA ---------")
pca = PCA(0.95,svd_solver='full')
dim_antes=np.shape(X_train)[1]
X_train = pca.fit_transform(X_train)
dim_despues=np.shape(X_train)[1]
X_test=pca.transform(X_test)
print("Aportación de los componentes: [",pca.explained_variance_ratio_[0],",",pca.explained_variance_ratio_[1],"...]")
print("Reducción de dimensionalidad: ",dim_antes," -> ",dim_despues)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train)
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


# Esquema de uso de cross-validation con Kfold

n=10 # Numero de folds
kf=KFold(n)
mean_accuracy=0
for train_index, test_index in kf.split(X_train):
    clf=LogisticRegression(multi_class='multinomial'
        ,penalty='l2',solver='lbfgs',max_iter=1500).fit(X_train[train_index],y_train[train_index])
    mean_accuracy+=accuracy_score(y_train[test_index],clf.predict(X_train[test_index]))
mean_accuracy=mean_accuracy/n
print(mean_accuracy)


# Grid search ?

from sklearn.model_selection import GridSearchCV
parameters = {'multi_class':('ovr','multinomial'),'solver':['lbfgs'],'max_iter':[1500]}
lr = LogisticRegression()
clf = GridSearchCV(lr,parameters,cv=10)
clf.fit(X_train, y_train)
print(clf.cv_results_) # pandas?

