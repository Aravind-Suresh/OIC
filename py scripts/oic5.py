import cv2
import numpy as np

cap = cv2.VideoCapture(0)
cas = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eye.xml')

while(1):
	ret, img = cap.read()
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	i = 0
	for (a,b,c,d) in cas.detectMultiScale(img, 1.3, 5):
		#cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
		roi = img[b:b+d, a:a+c]
		roi_corner=roi
		lapl = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)
		edge = cv2.Canny(roi, 25, 25)
		gray = np.float32(roi)
		dst = cv2.cornerHarris(gray,2,3,0.04)
		dst = cv2.dilate(dst,None)

		#distlapl = cv2.distanceTransform(lapl, cv2.cv.CV_DIST_L2,5)
		#dist = cv2.distanceTransform(lapl, cv2.cv.CV_DIST_L2, cv2.cv.CV_DIST_MASK_PRECISE)
		#dist3 = np.zeros((dist.shape[0], dist.shape[1], 3), dtype = np.uint8)
		#dist3[:, :, 0] = dist
		#dist3[:, :, 1] = dist
		#dist3[:, :, 2] = dist

		dist_transform = cv2.distanceTransform(lapl,cv2.cv.CV_DIST_L2,5)
		ret, sure_fg = cv2.threshold(dist_transform,0.5*dist_transform.max(),255,0)

		roi_corner[dst>0.01*dst.max()]=255
		#cv2.imshow("roi" + str(i), roi)
		if i<3:
			#cv2.imshow("lapl" + str(i), lapl)
			#cv2.imshow("corner" + str(i), roi_corner)
			#cv2.imshow("edge" + str(i), edge)
			cv2.imshow("dist" + str(i), dist_transform)
		i = i + 1

	cv2.imshow("img1", img)

	if cv2.waitKey(1) == 27:
		break

cap.release()