import cv2

cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

img = cv2.imread('img1.png')
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
i = 0
for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
	cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
	roi = img[b:b+d, a:a+c]
	lapl = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)
	edge = cv2.Canny(roi, 100, 200)
	#cv2.imshow("roi" + str(i), roi)
	cv2.imshow("edge" + str(i), edge)
	cv2.imshow("lapl" + str(i), lapl)
	i = i + 1

cv2.imshow("img1", img)
cv2.waitKey(0)