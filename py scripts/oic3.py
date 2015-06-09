import cv2
    ...: import numpy as np
    ...: import cv2.cv as cv
    ...: 
    ...: cap = cv2.VideoCapture(0)
    ...: cas_face = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
    ...: cas_eyes = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_eyes.xml')
    ...: roi = []
    ...: #img = cv2.imread("/home/rupesh/cvg/OIC/images/eyes2.jpeg",0)
    ...: 
    ...: while(1):
    ...:     ret, img = cap.read()
    ...:     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ...:     i=0
    ...:     for (a,b,c,d) in cas_face.detectMultiScale(img, 1.3, 5):
    ...:         cv2.rectangle(img, (a,b), (a+c,b+d), (255,255,255))
    ...:         roi = img[b:b+d, a:a+c]
    ...:         laplacian = cv2.Laplacian(img,cv2.CV_8UC1, ksize=7)
    ...:        #lapl = cv2.Laplacian(roi, cv2.CV_8UC1, ksize=7)
    ...:         roi = cv2.medianBlur(roi,5)
    ...:         edges = cv2.Canny(roi,10,100)
    ...:         circles = cv2.HoughCircles(img,cv.CV_HOUGH_GRADIENT,1,5,param1=50,param2=30,minRadius=0,maxRadius=0)
    ...:         #circles = np.uint16(np.around(circles))
    ...:         if circles is not None:
    ...:             for i in circles[0,:]:
    ...:                 if i[0]>b and i[1]<b+d and i[1]>a and i[1]<a+c: 
    ...:                     #draw the outer circle
    ...:                     cv2.circle(roi,(i[0],i[1]),i[2],(0,255,0),2)
    ...:                     # draw the center of the circle
    ...:                     cv2.circle(roi,(i[0],i[1]),2,(0,0,255),3)
    ...:                     # draw the outer circle
    ...:                     cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    ...:                        # draw the center of the circle
    ...:                     cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    ...:         #cv2.imshow("roi_edges" + str(i), edges)
    ...:     #cv2.imshow("lapl" + str(i), lapl)
    ...:     cv2.imshow("img1_lap", laplacian)
    ...:     cv2.imshow("img1_edges", edges)
    ...:     cv2.imshow("img1", img)
    ...:     if cv2.waitKey(1) == 27:
    ...:         break
    ...: 
    ...: cap.release()
