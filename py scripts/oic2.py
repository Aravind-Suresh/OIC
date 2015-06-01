import cv2
import cv2.cv as cv
import numpy as np

cap = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
roi = []

while(1):
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	i = 0
	for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
		cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
		roi = img[b:b+d, a:a+c]
		#lapl = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)
		
		roi = cv2.medianBlur(img,5)
		circles = cv2.HoughCircles(roi,cv.CV_HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
		
		circles = np.uint16(np.around(circles))
		for i in circles[0,:]:
			# draw the outer circle
			cv2.circle(roi,(i[0],i[1]),i[2],(0,255,0),2)
			# draw the center of the circle
			cv2.circle(roi,(i[0],i[1]),2,(0,0,255),3)
			# draw the outer circle
			cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
			# draw the center of the circle
			cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
	#cv2.imshow("roi" + str(i), roi)
	#cv2.imshow("lapl" + str(i), lapl)
	cv2.imshow("img1", img)
		

	if cv2.waitKey(1) == 27:
		break

cap.release()


