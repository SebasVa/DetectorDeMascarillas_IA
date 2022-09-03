import cv2
import os
import numpy as np

#Dataset con las fotos de las mascarillas
dataPath = "C:\\Users\\USER\\Documents\\IA\\Proyecto\\Dataset_faces"
dir_list = os.listdir(dataPath)
print("Lista archivos:", dir_list)

#Para que las etiquetas estén asociadas a cada imagen
labels = []

#Almacen de los rostros
facesData = []

label = 0

#En esta parte lo que se va a hacer es construir el path respectivo a la imagen que se va a leer
for name_dir in dir_list:
     dir_path = dataPath + "/" + name_dir
     
     for file_name in os.listdir(dir_path):
          image_path = dir_path + "/" + file_name
          print(image_path)
          image = cv2.imread(image_path, 0)
          facesData.append(image)
          labels.append(label)
     label += 1

#Todas las imagenes sin mascarilla estarán con la etiqueta 1 y las que si tengan con etiqueta 0
print("Etiqueta 0: ", np.count_nonzero(np.array(labels) == 0))
print("Etiqueta 1: ", np.count_nonzero(np.array(labels) == 1))

# LBPH FaceRecognizer
face_mask = cv2.face.LBPHFaceRecognizer_create() 

# Entrenamiento
print("Entrenando...")
face_mask.train(facesData, np.array(labels))

# Almacenar modelo
face_mask.write("face_mask_model.xml")
print("Modelo almacenado")
