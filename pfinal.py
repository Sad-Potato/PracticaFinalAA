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

print(Data,X,Y)
print(type(X))