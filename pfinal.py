# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import VarianceThreshold
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

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


# Exploración y análisis de datos de entrenamiento
"""
    Exploración y análisis de datos
    Histograma de frecuencia de clase: sirve para visualizar y detectar clases desbalanceadas
    Boxplot de cada característica: información visual estadística de cada característica
                                    útil para ver visualmente los distintos rangos en los que se mueven las características
    Matriz de correlaciones: información visual de la correlación entre variables
""" 

# # Pasar datos a dataframe de pandas
# cabecera = ['C' + str(i) for i in range(X_train.shape[1])]    # Formateo de columnas
# cabecera.append('letra')
# dataTrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train.reshape(-1, 1))], axis = 1)
# dataTrain.set_axis(cabecera, axis=1, inplace=True)

# # Histograma de frecuencia de cada clase
# plt.figure(figsize = (9, 6))
# ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
# ax.set_xlabel("Clases")
# ax.set_ylabel("Frecuencia")
# plt.title("Histograma de frecuencia")
# plt.show()

# print("Histograma de frecuencias agrupado por clases")
# input("\n--- Pulsar tecla para continuar ---\n")

# # Boxplot de las distintas características
# plt.figure(figsize = (14, 7))
# plt.xlabel("Características del problema")
# boxplot = dataTrain[dataTrain.columns[0:-1]].boxplot()
# plt.title("Boxplot de cada característica")
# plt.show()

# print("Boxplot para cada característica del problema")
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

"""
    Preprocesado de los datos
"""

# Escalado de características
"""
    Para escalar las características se usa estandarización, ya que los datos
    presentan pequeñas diferencias de escalas
"""
# Estandarización con StandardScaler
scaler = sk.preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# dataTrainScaled = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1),
#                                columns = dataTrain.columns)
                            
# # Boxplot de las distintas variables continuas tras estandarización
# plt.figure(figsize = (14, 7))
# plt.xlabel("Características del problema")
# boxplot = dataTrainScaled[dataTrainScaled.columns[0:-1]].boxplot()
# plt.title("Boxplot de cada variable tras estandarización")
# plt.show()

# print("Estandarización de los datos")
# print("Boxplot para cada característica del problema tras estandarización")
# input("\n--- Pulsar tecla para continuar ---\n")


# Eliminamos outliers
"""
    Eliminamos valores extremos, es decir instancias en cuales no un
    atributo esta "alejado" de los demas si no que en su conjunto esta 
    alejada del resto de instancias
"""
# OUTLIERS CON LOCAL OUTLIER FACTOR

lof = LocalOutlierFactor(contamination=0.05)
outliers=np.where(lof.fit_predict(X_train)==-1)
X_train=np.delete(X_train,outliers,0)
y_train=np.delete(y_train,outliers,0)

# # Obtener nuevo dataframe tras eliminación de outliers
# dataTrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train.reshape(-1, 1))], axis = 1)
# dataTrain.set_axis(cabecera, axis=1, inplace=True)

# # Histograma de frecuencia de cada clase después de eliminar outliers
# plt.figure(figsize = (9, 6))
# ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
# ax.set_xlabel("Clases")
# ax.set_ylabel("Frecuencia")
# plt.title("Histograma de frecuencia")
# plt.show()

# from sklearn.ensemble import IsolationForest
# clf = IsolationForest(random_state=0).fit(X_train)
# outliers=np.where(clf.predict(X_train)==-1)
# X_train=np.delete(X_train,outliers,0)
# y_train=np.delete(y_train,outliers,0)

print("Porcentaje de outliers: ",(np.size(outliers)/np.size(X_train[:,0]))*100,"%")
print("Histograma de frecuencias agrupado por clases después de eliminar outliers")
input("\n--- Pulsar tecla para continuar ---\n")

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
print("Aportación de los componentes: [",pca.explained_variance_ratio_[0],",",pca.explained_variance_ratio_[1],"...]")
print("Reducción de dimensionalidad: ",dim_antes," -> ",dim_despues)

plt.scatter(X_train[:,0],X_train[:,1],c=y_train) # Los puntos no dan mucha información no? Una alternativa sería t-SNE pero Nicolás igual se raya
plt.show()

ax = plt.axes(projection='3d')
ax.scatter3D(X_train[:,0], X_train[:,1], X_train[:,2], c=y_train);
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


#####################

#---------------------------------------------------------------------------------#
#------------------------ Modelos con cross-validation ---------------------------#
#---------------------------------------------------------------------------------#

# Usamos grid search

##################### Regresión logística ##########################


# parameters = {'multi_class':('ovr','multinomial'),'solver':['lbfgs'],'max_iter':[1500]}
# lr = LogisticRegression()
# clf = GridSearchCV(lr,parameters,cv=10)
# clf.fit(X_train, y_train)

# # Código para visualizar los resultados de grid search
# print("Grid scores on development set:")
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']

# #THIS IS WHAT YOU WANT ٩(◕‿◕｡)۶
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

##################### Perceptron multicapa #########################

# ~13 min con Ryzen5 2600 3.7GHz 12 cores

# parameters = {'hidden_layer_sizes':[(50,50), (75, 75), (100, 100)],'activation':('tanh', 'logistic'),
#               'alpha':[1e-3, 1e-4, 1e-5]} # Faltan
# mlp = MLPClassifier(batch_size = 64, random_state = 42)
# clf = GridSearchCV(mlp,parameters,cv = 10, n_jobs = -1)
# clf.fit(X_train, y_train)

# # mlp=MLPClassifier((100, 100), solver = 'adam', batch_size = 64, alpha = 0.01)
# # mlp.fit(X_train,y_train)
# # print(mlp.score(X_train,y_train))
# # X_test = scaler.transform(X_test)
# # X_test = pca.transform(X_test)
# # print(mlp.score(X_test, y_test))

# # Código para visualizar los resultados de grid search
# print("Grid scores on development set:")
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']

# #THIS IS WHAT YOU WANT
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     print("%0.4f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))



############################## SVM ##############################

# svc = SVC()
# svc.fit(X_train, y_train)
# print(svc.score(X_train,y_train))
# X_test = scaler.transform(X_test)
# X_test = pca.transform(X_test)
# print(svc.score(X_test, y_test))


# parametersRBF = {'kernel':['rbf'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85]}
# parametersPOLY = {'kernel':['poly'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85], 'degree':[5, 8], 'coef0':[0.15, 0.85]}
# svc = SVC(random_state = 42)
# clfPOLY = GridSearchCV(svc,parametersPOLY,cv=10,n_jobs=-1)
# clfRBF = GridSearchCV(svc,parametersRBF,cv=10,n_jobs=-1)
# clfPOLY.fit(X_train, y_train)
# clfRBF.fit(X_train, y_train)

# # Código para visualizar los resultados de grid search
# print("Grid scores on development set:")
# means = clfPOLY.cv_results_['mean_test_score']
# stds = clfPOLY.cv_results_['std_test_score']

# #THIS IS WHAT YOU WANT
# for mean, std, params in zip(means, stds, clfPOLY.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))
    

# # Código para visualizar los resultados de grid search
# print("Grid scores on development set:")
# means = clfRBF.cv_results_['mean_test_score']
# stds = clfRBF.cv_results_['std_test_score']

# #THIS IS WHAT YOU WANT
# for mean, std, params in zip(means, stds, clfRBF.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r"
#           % (mean, std * 2, params))

"""
    Mejor modelo
    SVM con kernel RBF C = 10, gamma = 0.15
"""
svc = MLPClassifier((100, 100), activation = 'tanh', alpha = 0.001)
svc.fit(X_train, y_train)
print(svc.score(X_train,y_train))
X_test = scaler.transform(X_test)
X_test = pca.transform(X_test)
print(svc.score(X_test, y_test))

#---------------------------------------------------------------------------------#
#---------------------- Selección de la mejor hipotesis --------------------------#
#---------------------------------------------------------------------------------#

# Seleccionamos la mejor hipotesis de entre las que hemos 
# considerado en cross-validation