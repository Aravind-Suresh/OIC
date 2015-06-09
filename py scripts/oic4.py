import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

while(1):
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	i = 0
	for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
		cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
		roi = img[b:b+d, a:a+c]
		roi_corner=roi
		lapl = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)
		edge = cv2.Canny(roi, 25, 25)
		corners = cv2.goodFeaturesToTrack(roi_corner,25,0.01,10)
		corners = np.int0(corners)
		for i in corners:
			x,y = i.ravel()
			cv2.circle(img,(x,y),3,255,-1)
		#for p in corners:
			#roi_corner[p]=255
		#cv2.imshow("roi" + str(i), roi)
		if i<3:
			cv2.imshow("lapl" + str(i), lapl)
			cv2.imshow("corner" + str(i), roi_corner)
			cv2.imshow("edge" + str(i), edge)
		i = i + 1

	cv2.imshow("img1", img)

	if cv2.waitKey(1) == 27:
		break

cap.release()