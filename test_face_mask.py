import cv2
import os
import mediapipe as mp

mp_face_detection = mp.solutions.face_detection
LABELS = ["Con_mascarilla", "Sin_mascarilla"]

# Leer el modelo
face_mask = cv2.face.LBPHFaceRecognizer_create()
face_mask.read("face_mask_model.xml")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

with mp_face_detection.FaceDetection(
     min_detection_confidence=0.5) as face_detection:
     while True:
          ret, frame = cap.read()
          if ret == False: break
          frame = cv2.flip(frame, 1)
          height, width, _ = frame.shape
          frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          results = face_detection.process(frame_rgb)
          if results.detections is not None:
               for detection in results.detections:
                    xmin = int(detection.location_data.relative_bounding_box.xmin * width)
                    ymin = int(detection.location_data.relative_bounding_box.ymin * height)
                    w = int(detection.location_data.relative_bounding_box.width * width)
                    h = int(detection.location_data.relative_bounding_box.height * height)
                    if xmin < 0 and ymin < 0:
                         continue
                    #Se aplica la escala de grises y se redimensiona la imagen para que sea como las usadas para el entrenamiento
                    face_image = frame[ymin : ymin + h, xmin : xmin + w]
                    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    face_image = cv2.resize(face_image, (72, 72), interpolation=cv2.INTER_CUBIC)
                    
                    #Con el rostro detectado se aplica el modelo entrenado
                    result = face_mask.predict(face_image)
                    #implementación del cuadro para detectar la mascarilla dependiendo de como este la etiqueta
                    if result[1] < 150:
                         color = (0, 255, 0) if LABELS[result[0]] == "Con_mascarilla" else (0, 0, 255)
                         cv2.putText(frame, "{}".format(LABELS[result[0]]), (xmin, ymin - 15), 2, 1, color, 1, cv2.LINE_AA)
                         cv2.rectangle(frame, (xmin, ymin), (xmin + w, ymin + h), color, 2)
          cv2.imshow("Frame", frame)
          k = cv2.waitKey(1)
          if k == 27:
               break

cap.release()
cv2.destroyAllWindows()