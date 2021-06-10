# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk












# Fijamos la semilla de las 
# componentes aleatorias
np.random.seed(42)

#---------------------------------------------------------------------------------#
#------------------------- Lectura de datos y preprocesado -----------------------#
#---------------------------------------------------------------------------------#

# Lectura de la base de datos

Data=np.loadtxt("./datos/letter-recognition.data",dtype=str,delimiter=",")

X=np.array(Data[:,1:]).astype(np.int32)
Y=Data[:,0]
# Yn=np.empty(np.size(Data[:,0])) # Clases como numeros 
# for i in np.arange(np.size(Yn)):
#     Yn[i]=ord(Y[i])-65

print(Data,X,Y)
print(type(X))