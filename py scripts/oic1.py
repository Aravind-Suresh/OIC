import cv2

cap = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')
roi = []

while(1):
	ret, img = cap.read()
	i = 0
	for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
		cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
		roi = img[b:b+d, a:a+c]
		cv2.imshow("roi" + str(i), roi)
		i = i + 1

	cv2.imshow("img1", img)

	if cv2.waitKey(1) == 27:
		break
