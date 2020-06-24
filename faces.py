import numpy as np
import cv2
import pickle

face_cascade=cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
#eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')
#smile_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_smile.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

# Capture video from file
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()

    if ret == True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        for(x, y, w, h) in faces:
        	#print(x, y, w, h)
        	roi_gray = gray[y:y+h, x:x+w] # (ycord_start, ycord_end) region of interest
        	roi_color = frame[y:y+h, x:x+w]

        	#for recognizing: deep learning model predict keras tensorflow pytorch  scikit
        	id_, conf = recognizer.predict(roi_gray) #confidence
        	if conf>=45 and conf<= 85:
        		#print(id_)
        		#print(labels[id_])
        		font = cv2.FONT_HERSHEY_SIMPLEX
        		name = labels[id_]
        		color = (255, 255 , 255)
        		stroke = 2
        		cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        	img_item= "10.png"
        	cv2.imwrite(img_item, roi_color)

        	color = (255, 0 ,0) #BGR 0-255
        	stroke = 2
        	end_cord_x= x + w 
        	end_cord_y = y + h 
        	cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        	#eyes = eye_cascade.detectMultiScale(roi_gray)
			#for (ex,ey,ew,eh) in eyes:
			#cv2.square(roi_color,(ex,ey),(ex+ew,ey+eh),(0,0,255),1)
        	#smiley = smile_cascade.detectMultiScale(roi_gray)
    		#for (ex,ey,ew,eh) in smiley:
    		#	cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),1)

        	#Display resulting frame
        cv2.imshow('Image Processing 101',frame)


        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    else:
        break

#release it when everything done
cap.release()
cv2.destroyAllWindows()
