import cv2

cap = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

lapl2 = None

while(1):
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	i = 0
	for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
		cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
		roi = img[b:b+d, a:a+c]
		lapl1 = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)

		if lapl2 is not None:
			lapl = lapl1 - lapl2
		else :
			lapl = lapl1
		#cv2.imshow("roi" + str(i), roi)
		cv2.imshow("lapl" + str(i), lapl)
		i = i + 1

		if lapl2 is None:
			lapl2 = lapl1

	cv2.imshow("img1", img)

	if cv2.waitKey(1) == 27:
		break

cap.release()