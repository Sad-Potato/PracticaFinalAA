# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Eliminamos los warnings por haber llegado al
# límite de iteraciones
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import sklearn as sk
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_confusion_matrix

# Modelos
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

# Fijamos la semilla de las componentes aleatorias
np.random.seed(42)

"""
    IMPORTANTE: para la obtención de resultados
    hemos usado la variable que disponemos inmediatamente 
    abajo con el valor de -1 para usar todos los threads disponibles
    y asi reducir drasticamente el tiempo de computo de los algoritmos 
    (De potencialmente superior a la hora a cuestion de 10~ minutos en 
    algunos modelos). La variable indica el numero de threads no cores.
    Lo ponemos en 1 por defecto
"""
num_threads=1

#---------------------------------------------------------------------------------#
#------------------------- Lectura de datos y preprocesado -----------------------#
#---------------------------------------------------------------------------------#

"""
    El dataset debe de estar contenido en un carpteta llamada "datos" y dicha
    carpeta debe estar en el mismo directorio que el script.
    El dataset se encuentra disponible en la siguiente URL:
        http://archive.ics.uci.edu/ml/datasets/Letter+Recognition
"""

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
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2,random_state=42,shuffle=True,stratify=Y)
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

# Pasar datos a dataframe de pandas
cabecera = ['C' + str(i) for i in range(X_train.shape[1])]    # Formateo de columnas
cabecera.append('letra')
dataTrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train.reshape(-1, 1))], axis = 1)
dataTrain.set_axis(cabecera, axis=1, inplace=True)

# Histograma de frecuencia de cada clase
plt.figure(figsize = (8, 5))
ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
ax.set_xlabel("Clases")
ax.set_ylabel("Frecuencia")
plt.title("Histograma de frecuencia")
plt.show()

print("Histograma de frecuencias agrupado por clases")
input("\n--- Pulsar tecla para continuar ---\n")

# Boxplot de las distintas características
plt.figure(figsize = (8, 5))
plt.xlabel("Características del problema")
boxplot = dataTrain[dataTrain.columns[0:-1]].boxplot()
plt.title("Boxplot de cada característica")
plt.show()

print("Boxplot para cada característica del problema")
input("\n--- Pulsar tecla para continuar ---\n")

# Mostrar gráfica de la matriz de correlaciones
f = plt.figure(figsize=(8, 5))
plt.matshow(dataTrain[dataTrain.columns[0:-1]].corr(), fignum=f.number)
plt.xticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=9, rotation=45)
plt.yticks(range(dataTrain.select_dtypes(['number']).shape[1] - 1), dataTrain.select_dtypes(['number']).columns[0:-1], fontsize=9)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=9)
plt.title('Correlation Matrix', fontsize=11)
plt.show()

print("Gráfica de la matriz de correlaciones")
input("\n--- Pulsar tecla para continuar ---\n")

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
dataTrainScaled = pd.DataFrame(np.concatenate((X_train, y_train.reshape(-1,1)), axis = 1),
                               columns = dataTrain.columns)
                            
# Boxplot de las distintas variables continuas tras estandarización
plt.figure(figsize = (8, 5))
plt.xlabel("Características del problema")
boxplot = dataTrainScaled[dataTrainScaled.columns[0:-1]].boxplot()
plt.title("Boxplot de cada característica tras estandarización")
plt.show()

print("Estandarización de los datos")
print("Boxplot para cada característica del problema tras estandarización")
input("\n--- Pulsar tecla para continuar ---\n")


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

# Obtener nuevo dataframe tras eliminación de outliers
dataTrain = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train.reshape(-1, 1))], axis = 1)
dataTrain.set_axis(cabecera, axis=1, inplace=True)

# Histograma de frecuencia de cada clase después de eliminar outliers
plt.figure(figsize = (9, 6))
ax = dataTrain['letra'].plot.hist(bins = np.arange(27) - 0.5, ec = 'black', xticks=np.arange(26), rot = 45)
ax.set_xlabel("Clases")
ax.set_ylabel("Frecuencia")
plt.title("Histograma de frecuencia")
plt.show()

print("Porcentaje de outliers: ",(np.size(outliers)/np.size(X_train[:,0]))*100,"%")
print("Histograma de frecuencias agrupado por clases después de eliminar outliers")
input("\n--- Pulsar tecla para continuar ---\n")

# Tecnica de reduccion de dimensionalidad PCA 
"""
    Usamos la tecnica de PCA, la idea detras de esto es reducir el 
    número de variables manteniendo la maxima cantidad de informacion.
    En nuestro caso aplicamos PCA manteniendo un 95% de la varianza de los datos

"""

print("\n--------- PCA ---------")
pca = PCA(0.95,svd_solver='full')
dim_antes=np.shape(X_train)[1]
X_train_PCA = pca.fit_transform(X_train)
dim_despues=np.shape(X_train_PCA)[1]
print("Aportación de los componentes: [",pca.explained_variance_ratio_[0],",",pca.explained_variance_ratio_[1],"...]")
print("Reducción de dimensionalidad: ",dim_antes," -> ",dim_despues)

# Gráfica 2D
plt.figure(figsize = (9,6))
plt.scatter(X_train_PCA[:,0],X_train_PCA[:,1],c=y_train)
plt.show()

# Gráfica 3D
ax = plt.axes(projection='3d')
ax.scatter3D(X_train_PCA[:,0], X_train_PCA[:,1], X_train_PCA[:,2], c=y_train);
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")


"""
    Selección de modelos usando Cross-Validation
        Tal y como se indicó en la sesión de dudas, si se decide aplicar PCA
        hay que "resolver" dos problemas: uno transformando los datos con PCA
        y otro con los datos sin utilizar PCA (es decir, se toman como conjuntos
        de datos diferentes).
        Es por ello que se usa 10-fold CV para cada caso.
"""
"""
    IMPORTANTE:
        Todo el código relacionado con Cross-validation está comentado porque
        su ejecución (con un procesador Ryzen5 2600 3.7GHz utilizando los 12
        cores disponibles) puede llevar hasta una hora. Los resultados que
        se obtienen de la ejecución de dicho código están al final del script
        (comentados). Si se desea corroborar los resultados, descomentar las líneas
        225 y 385 y ejecutar el código.
"""
"""
#---------------------------------------------------------------------------------#
#-------------------------- DATOS OBTENIDOS CON PCA ------------------------------#
#---------------------------------------------------------------------------------#

print("CASO PCA\n")

##################### Regresión logística ##########################

# ~5 min con Ryzen5 2600 3.7GHz 12 cores
print("\nRegresion Logistica con PCA ...\n")
parameters = {'multi_class':('ovr','multinomial'), 'penalty':('l1', 'l2'), 'C':[1, 10, 100, 1000]}
lr = LogisticRegression(solver = 'saga', max_iter = 1000, random_state = 42)
clf = GridSearchCV(lr,parameters,cv=10, n_jobs=num_threads)
clf.fit(X_train_PCA, y_train)

# Código para visualizar los resultados de grid search
# Mostramos la media y la desviación típica de los errores del conjunto
# de validación para cada combinación de atributos
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

##################### Perceptron multicapa #########################

# ~13 min con Ryzen5 2600 3.7GHz 12 cores
print("\nPerceptron multicapa con PCA ...\n")
parameters = {'hidden_layer_sizes':[(50,50), (75, 75), (100, 100)],'activation':('tanh', 'logistic'),
              'alpha':[1e-3, 1e-4, 1e-5]} # Faltan
mlp = MLPClassifier(batch_size = 64, random_state = 42)
clf = GridSearchCV(mlp,parameters,cv = 10, n_jobs=num_threads)
clf.fit(X_train_PCA, y_train)

# Código para visualizar los resultados de grid search
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


############################## SVM ##############################

# ~7 min con Ryzen5 2600 3.7GHz 12 cores
print("\nSVM con PCA ...\n")
parametersRBF = {'kernel':['rbf'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85]}
parametersPOLY = {'kernel':['poly'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85], 'degree':[5, 8], 'coef0':[0.15, 0.85]}
svc = SVC(random_state = 42)
clfPOLY = GridSearchCV(svc,parametersPOLY,cv=10,n_jobs=num_threads)
clfRBF = GridSearchCV(svc,parametersRBF,cv=10,n_jobs=num_threads)
clfPOLY.fit(X_train_PCA, y_train)
clfRBF.fit(X_train_PCA, y_train)

# Código para visualizar los resultados de grid search
print("Grid scores on development set:")
means = clfPOLY.cv_results_['mean_test_score']
stds = clfPOLY.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clfPOLY.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
    

# Código para visualizar los resultados de grid search
means = clfRBF.cv_results_['mean_test_score']
stds = clfRBF.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clfRBF.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

################################################################

input("\n--- Pulsar tecla para continuar ---\n")

#---------------------------------------------------------------------------------#
#---------------------------- DATOS SIN APLICAR PCA ------------------------------#
#---------------------------------------------------------------------------------#

print("CASO NO PCA\n\n")

##################### Regresión logística ##########################

# ~5 min con Ryzen5 2600 3.7GHz 12 cores

print("\nRegresión logística sin PCA ...\n")
parameters = {'multi_class':('ovr','multinomial'), 'penalty':('l1', 'l2'), 'C':[1, 10, 100, 1000]}
lr = LogisticRegression(solver = 'saga', max_iter = 1000, random_state = 42)
clf = GridSearchCV(lr,parameters,cv=10, n_jobs=num_threads)
clf.fit(X_train, y_train)

# Código para visualizar los resultados de grid search
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

##################### Perceptron multicapa #########################

# ~13 min con Ryzen5 2600 3.7GHz 12 cores

print("\nPerceptron multicapa sin PCA ...\n")
parameters = {'hidden_layer_sizes':[(50,50), (75, 75), (100, 100)],'activation':('tanh', 'logistic'),
              'alpha':[1e-3, 1e-4, 1e-5]} # Faltan
mlp = MLPClassifier(batch_size = 64, random_state = 42)
clf = GridSearchCV(mlp,parameters,cv = 10, n_jobs=num_threads)
clf.fit(X_train, y_train)

# Código para visualizar los resultados de grid search
print("Grid scores on development set:")
means = clf.cv_results_['mean_test_score']
stds = clf.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    print("%0.4f (+/-%0.03f) for %r"
          % (mean, std * 2, params))


############################## SVM ##############################

# ~7 min con Ryzen5 2600 3.7GHz 12 cores

print("\nSVM sin PCA ...\n")
parametersRBF = {'kernel':['rbf'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85]}
parametersPOLY = {'kernel':['poly'], 'C':[1, 10, 100], 'gamma':[0.15, 0.5, 0.85], 'degree':[5, 8], 'coef0':[0.15, 0.85]}
svc = SVC(random_state = 42)
clfPOLY = GridSearchCV(svc,parametersPOLY,cv=10,n_jobs=num_threads)
clfRBF = GridSearchCV(svc,parametersRBF,cv=10,n_jobs=num_threads)
clfPOLY.fit(X_train, y_train)
clfRBF.fit(X_train, y_train)

# Código para visualizar los resultados de grid search
print("Grid scores on development set:")
means = clfPOLY.cv_results_['mean_test_score']
stds = clfPOLY.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clfPOLY.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

# Código para visualizar los resultados de grid search
means = clfRBF.cv_results_['mean_test_score']
stds = clfRBF.cv_results_['std_test_score']

for mean, std, params in zip(means, stds, clfRBF.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))

################################################################

input("\n--- Pulsar tecla para continuar ---\n")
"""

"""
    Mejor modelo
    PARA DATOS CON PCA: SVM con kernel RBF C = 10, gamma = 0.15
    PARA DATOS SIN PCA: SVM con kernel RBF C = 10, gamma = 0.15
"""

#---------------------------------------------------------------------------------#
#---------------------- Selección de la mejor hipotesis --------------------------#
#---------------------------------------------------------------------------------#

# CASO PCA
print("CASO PCA\n")

print("Mejor modelo: SVM (kernel = rbf, C = 10, gamma = 0.15)\n")
svc = SVC(C = 10, gamma = 0.15)
svc.fit(X_train_PCA, y_train)
print("Ein train: ",svc.score(X_train_PCA,y_train))
X_test_PCA = scaler.transform(X_test)
X_test_PCA = pca.transform(X_test_PCA)
print("Eout test: ",svc.score(X_test_PCA, y_test))

print("Matriz de confusión con los datos de test")

# Dibujar matriz de confusión
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(svc, X_test_PCA, y_test, cmap = 'Blues', ax = ax, xticks_rotation = 45)
plt.title("Confusion matrix (PCA)")
plt.show()

input("\n--- Pulsar tecla para continuar ---\n")

################################################################

# CASO NO PCA
print("CASO NO PCA\n")

print("Mejor modelo: SVM (kernel = rbf, C = 10, gamma = 0.15)\n")
svc = SVC(C = 10, gamma = 0.15)
svc.fit(X_train, y_train)
print("Ein train: ",svc.score(X_train,y_train))
X_test = scaler.transform(X_test)
print("Eout test: ",svc.score(X_test, y_test))

print("Matriz de confusión con los datos de test")

# Dibujar matriz de confusión
fig, ax = plt.subplots(figsize=(10, 8))
plot_confusion_matrix(svc, X_test, y_test, cmap = 'Blues', ax = ax, xticks_rotation = 45)
plt.title("Confusion matrix (NO PCA)")
plt.show()

################################################################

"""
    RESULTADOS 10-FOLD CROSS-VALIDATION:
    
CASO PCA


Regresion Logistica con PCA ...

Grid scores on development set:
0.668 (+/-0.024) for {'C': 1, 'multi_class': 'ovr', 'penalty': 'l1'}
0.669 (+/-0.023) for {'C': 1, 'multi_class': 'ovr', 'penalty': 'l2'}
0.724 (+/-0.026) for {'C': 1, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.723 (+/-0.024) for {'C': 1, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.669 (+/-0.023) for {'C': 10, 'multi_class': 'ovr', 'penalty': 'l1'}
0.669 (+/-0.023) for {'C': 10, 'multi_class': 'ovr', 'penalty': 'l2'}
0.724 (+/-0.027) for {'C': 10, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.725 (+/-0.027) for {'C': 10, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.669 (+/-0.023) for {'C': 100, 'multi_class': 'ovr', 'penalty': 'l1'}
0.669 (+/-0.023) for {'C': 100, 'multi_class': 'ovr', 'penalty': 'l2'}
0.725 (+/-0.027) for {'C': 100, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.725 (+/-0.027) for {'C': 100, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.669 (+/-0.023) for {'C': 1000, 'multi_class': 'ovr', 'penalty': 'l1'}
0.669 (+/-0.023) for {'C': 1000, 'multi_class': 'ovr', 'penalty': 'l2'}
0.725 (+/-0.027) for {'C': 1000, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.725 (+/-0.027) for {'C': 1000, 'multi_class': 'multinomial', 'penalty': 'l2'}

Perceptron multicapa con PCA ...

Grid scores on development set:
0.934 (+/-0.016) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50)}
0.949 (+/-0.014) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (75, 75)}
0.955 (+/-0.010) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100)}
0.932 (+/-0.016) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50)}
0.946 (+/-0.015) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (75, 75)}
0.951 (+/-0.010) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100)}
0.932 (+/-0.016) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50)}
0.945 (+/-0.014) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (75, 75)}
0.951 (+/-0.007) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (100, 100)}
0.923 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50)}
0.941 (+/-0.008) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (75, 75)}
0.950 (+/-0.010) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100)}
0.921 (+/-0.013) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50)}
0.940 (+/-0.009) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (75, 75)}
0.948 (+/-0.009) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100)}
0.921 (+/-0.013) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50)}
0.940 (+/-0.010) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (75, 75)}
0.948 (+/-0.008) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (100, 100)}

SVM con PCA ...

Grid scores on development set:
0.916 (+/-0.015) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.898 (+/-0.016) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.894 (+/-0.017) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.856 (+/-0.018) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.832 (+/-0.022) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.823 (+/-0.022) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.942 (+/-0.011) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.922 (+/-0.017) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.913 (+/-0.018) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.919 (+/-0.013) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.881 (+/-0.019) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.865 (+/-0.020) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.919 (+/-0.016) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.898 (+/-0.019) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.894 (+/-0.018) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.866 (+/-0.020) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.833 (+/-0.022) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.824 (+/-0.022) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.938 (+/-0.016) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.922 (+/-0.017) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.913 (+/-0.018) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.919 (+/-0.012) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.881 (+/-0.019) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.865 (+/-0.020) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.913 (+/-0.017) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.898 (+/-0.019) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.894 (+/-0.018) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.868 (+/-0.022) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.834 (+/-0.023) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.824 (+/-0.022) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.938 (+/-0.015) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.922 (+/-0.017) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.913 (+/-0.018) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.919 (+/-0.012) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.881 (+/-0.019) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.865 (+/-0.020) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.946 (+/-0.011) for {'C': 1, 'gamma': 0.15, 'kernel': 'rbf'}
0.955 (+/-0.014) for {'C': 1, 'gamma': 0.5, 'kernel': 'rbf'}
0.950 (+/-0.013) for {'C': 1, 'gamma': 0.85, 'kernel': 'rbf'}
0.962 (+/-0.008) for {'C': 10, 'gamma': 0.15, 'kernel': 'rbf'}
0.961 (+/-0.014) for {'C': 10, 'gamma': 0.5, 'kernel': 'rbf'}
0.953 (+/-0.012) for {'C': 10, 'gamma': 0.85, 'kernel': 'rbf'}
0.960 (+/-0.010) for {'C': 100, 'gamma': 0.15, 'kernel': 'rbf'}
0.960 (+/-0.014) for {'C': 100, 'gamma': 0.5, 'kernel': 'rbf'}
0.953 (+/-0.012) for {'C': 100, 'gamma': 0.85, 'kernel': 'rbf'}

--- Pulsar tecla para continuar ---

CASO NO PCA


Regresión logística sin PCA ...

Grid scores on development set:
0.728 (+/-0.027) for {'C': 1, 'multi_class': 'ovr', 'penalty': 'l1'}
0.729 (+/-0.027) for {'C': 1, 'multi_class': 'ovr', 'penalty': 'l2'}
0.778 (+/-0.019) for {'C': 1, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.778 (+/-0.020) for {'C': 1, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.728 (+/-0.028) for {'C': 10, 'multi_class': 'ovr', 'penalty': 'l1'}
0.728 (+/-0.028) for {'C': 10, 'multi_class': 'ovr', 'penalty': 'l2'}
0.779 (+/-0.020) for {'C': 10, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.779 (+/-0.020) for {'C': 10, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.728 (+/-0.028) for {'C': 100, 'multi_class': 'ovr', 'penalty': 'l1'}
0.728 (+/-0.028) for {'C': 100, 'multi_class': 'ovr', 'penalty': 'l2'}
0.779 (+/-0.020) for {'C': 100, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.779 (+/-0.020) for {'C': 100, 'multi_class': 'multinomial', 'penalty': 'l2'}
0.728 (+/-0.028) for {'C': 1000, 'multi_class': 'ovr', 'penalty': 'l1'}
0.728 (+/-0.028) for {'C': 1000, 'multi_class': 'ovr', 'penalty': 'l2'}
0.779 (+/-0.019) for {'C': 1000, 'multi_class': 'multinomial', 'penalty': 'l1'}
0.779 (+/-0.019) for {'C': 1000, 'multi_class': 'multinomial', 'penalty': 'l2'}

Perceptron multicapa sin PCA ...

Grid scores on development set:
0.9503 (+/-0.012) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50)}
0.9631 (+/-0.009) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (75, 75)}
0.9656 (+/-0.007) for {'activation': 'tanh', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100)}
0.9484 (+/-0.011) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50)}
0.9592 (+/-0.010) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (75, 75)}
0.9627 (+/-0.006) for {'activation': 'tanh', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100)}
0.9484 (+/-0.011) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50)}
0.9591 (+/-0.006) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (75, 75)}
0.9637 (+/-0.007) for {'activation': 'tanh', 'alpha': 1e-05, 'hidden_layer_sizes': (100, 100)}
0.9439 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (50, 50)}
0.9569 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (75, 75)}
0.9600 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.001, 'hidden_layer_sizes': (100, 100)}
0.9437 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (50, 50)}
0.9543 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (75, 75)}
0.9592 (+/-0.012) for {'activation': 'logistic', 'alpha': 0.0001, 'hidden_layer_sizes': (100, 100)}
0.9436 (+/-0.012) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (50, 50)}
0.9540 (+/-0.011) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (75, 75)}
0.9586 (+/-0.013) for {'activation': 'logistic', 'alpha': 1e-05, 'hidden_layer_sizes': (100, 100)}

SVM sin PCA ...

Grid scores on development set:
0.934 (+/-0.012) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.916 (+/-0.011) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.912 (+/-0.015) for {'C': 1, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.871 (+/-0.017) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.845 (+/-0.014) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.834 (+/-0.014) for {'C': 1, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.955 (+/-0.007) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.938 (+/-0.010) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.931 (+/-0.009) for {'C': 1, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.935 (+/-0.013) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.893 (+/-0.012) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.876 (+/-0.012) for {'C': 1, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.934 (+/-0.010) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.915 (+/-0.012) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.912 (+/-0.015) for {'C': 10, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.879 (+/-0.010) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.845 (+/-0.014) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.834 (+/-0.014) for {'C': 10, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.953 (+/-0.008) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.938 (+/-0.010) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.931 (+/-0.009) for {'C': 10, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.935 (+/-0.013) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.893 (+/-0.012) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.876 (+/-0.012) for {'C': 10, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.931 (+/-0.008) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.915 (+/-0.012) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.912 (+/-0.015) for {'C': 100, 'coef0': 0.15, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.879 (+/-0.013) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.845 (+/-0.014) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.834 (+/-0.014) for {'C': 100, 'coef0': 0.15, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.953 (+/-0.008) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.15, 'kernel': 'poly'}
0.938 (+/-0.010) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.5, 'kernel': 'poly'}
0.931 (+/-0.009) for {'C': 100, 'coef0': 0.85, 'degree': 5, 'gamma': 0.85, 'kernel': 'poly'}
0.935 (+/-0.013) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.15, 'kernel': 'poly'}
0.893 (+/-0.012) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.5, 'kernel': 'poly'}
0.876 (+/-0.012) for {'C': 100, 'coef0': 0.85, 'degree': 8, 'gamma': 0.85, 'kernel': 'poly'}
0.961 (+/-0.012) for {'C': 1, 'gamma': 0.15, 'kernel': 'rbf'}
0.966 (+/-0.011) for {'C': 1, 'gamma': 0.5, 'kernel': 'rbf'}
0.955 (+/-0.011) for {'C': 1, 'gamma': 0.85, 'kernel': 'rbf'}
0.974 (+/-0.008) for {'C': 10, 'gamma': 0.15, 'kernel': 'rbf'}
0.969 (+/-0.010) for {'C': 10, 'gamma': 0.5, 'kernel': 'rbf'}
0.957 (+/-0.010) for {'C': 10, 'gamma': 0.85, 'kernel': 'rbf'}
0.974 (+/-0.008) for {'C': 100, 'gamma': 0.15, 'kernel': 'rbf'}
0.969 (+/-0.010) for {'C': 100, 'gamma': 0.5, 'kernel': 'rbf'}
0.957 (+/-0.010) for {'C': 100, 'gamma': 0.85, 'kernel': 'rbf'}
"""