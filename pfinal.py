# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt











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
Y=Data[:,0]
Yn=np.empty(np.size(Data[:,0])) # Clases como numeros 
for i in np.arange(np.size(Yn)):
    Yn[i]=ord(Y[i])-65

cabecera = ['letra']
cabecera = cabecera + ['C' + str(i) for i in range(X.shape[1]-1)]

prueba = pd.DataFrame(Yn.reshape(-1, 1), columns = ['letra'])

# Histograma de frecuencia de cada clase
plt.figure()
ax = prueba['letra'].plot.hist(bins = np.arange(27) -0.5, ec = 'black', xticks=np.arange(26))
ax.set_xlabel("Clases")
ax.set_ylabel("Frecuencia")
plt.title("Histograma de frecuencia")
plt.show()
