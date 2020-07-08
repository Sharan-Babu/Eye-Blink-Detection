import cv2
import numpy as np
import dlib
from math import hypot

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\Sharan Babu\Anaconda3\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat")

cap = cv2.VideoCapture(0)

def midpoint(p1, p2):
	return int((p1.x + p2.x)/2), int((p1.y+p2.y)/2)



while True:
	_, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	faces = detector(gray)
	for face in faces:
		landmarks = predictor(gray, face)
		
		left_point = (landmarks.part(36).x, landmarks.part(36).y)
		right_point = (landmarks.part(39).x, landmarks.part(39).y)

		center_top = midpoint(landmarks.part(37), landmarks.part(38))
		center_bottom = midpoint(landmarks.part(41), landmarks.part(40))

		hor_line = cv2.line(frame, left_point, right_point, (0,255,0), 2)
		ver_line = cv2.line(frame, center_top, center_bottom, (0,255,0),2)

		h_length = hypot((left_point[0]-right_point[0]),(left_point[1]-right_point[1]))
		v_length = hypot((center_top[0]-center_bottom[0]),(center_top[1]-center_bottom[1]))

		if h_length/v_length > 5:
			cv2.putText(frame, "BLINKING",(50,150),cv2.FONT_HERSHEY_SIMPLEX,7,(255,0,0))



		l_point = (landmarks.part(42).x, landmarks.part(45).y)
		r_point = (landmarks.part(45).x, landmarks.part(45).y)

		c_top = midpoint(landmarks.part(43), landmarks.part(44))
		c_bottom = midpoint(landmarks.part(47), landmarks.part(46))

		h_line = cv2.line(frame, l_point, r_point,(0,255,0),2)
		v_line = cv2.line(frame, c_top, c_bottom, (0,255,0),2)

		hor_length = hypot((l_point[0]-r_point[0]),(l_point[1]-r_point[1]))
		ver_length = hypot((c_top[0]-c_bottom[0]),(c_top[1]-c_bottom[1]))

		if hor_length/ver_length > 5:
			cv2.putText(frame, "BLINKING",(50,150),cv2.FONT_HERSHEY_SIMPLEX,7,(255,0,0))

	cv2.imshow("Frame",frame)

	key = cv2.waitKey(1)
	if key == 27:
		break

cap.release()
cv2.destroyAllWindows()