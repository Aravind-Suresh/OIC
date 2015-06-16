#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <queue>
#include <stdio.h>

#include "constants.h"
#include "helpers.h"

#include "findEyeCorner.h"

using namespace std;
using namespace cv;


 /** Global variables */
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye.xml" ;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";


Mat *leftCornerKernel;
Mat *rightCornerKernel;

// not constant because stupid opencv type signatures
float kEyeCornerKernel[4][6] = {
  {-1,-1,-1, 1, 1, 1},
  {-1,-1,-1,-1, 1, 1},
  {-1,-1,-1,-1, 0, 3},
  { 1, 1, 1, 1, 1, 1},
};

void createCornerKernels() {
  rightCornerKernel = new Mat(4,6,CV_32F,kEyeCornerKernel);
  leftCornerKernel = new Mat(4,6,CV_32F);
  // flip horizontally
  flip(*rightCornerKernel, *leftCornerKernel, 1);
}

void releaseCornerKernels() {
  delete leftCornerKernel;
  delete rightCornerKernel;
}

// TODO implement these
Mat eyeCornerMap(const Mat &region, bool left, bool left2) {
  Mat cornerMap;

  Size sizeRegion = region.size();
  Range colRange(sizeRegion.width / 4, sizeRegion.width * 3 / 4);
  Range rowRange(sizeRegion.height / 4, sizeRegion.height * 3 / 4);

  Mat miRegion(region, rowRange, colRange);

  filter2D(miRegion, cornerMap, CV_32F,
   (left && !left2) || (!left && !left2) ? *leftCornerKernel : *rightCornerKernel);

  return cornerMap;
}

Point2f findEyeCorner(Mat region, bool left, bool left2) {
  Mat cornerMap = eyeCornerMap(region, left, left2);

  Point maxP;
  minMaxLoc(cornerMap,NULL,NULL,NULL,&maxP);

  Point2f maxP2;
  maxP2 = findSubpixelEyeCorner(cornerMap, maxP);
  // GFTT
//  vector<Point2f> corners;
//  goodFeaturesToTrack(region, corners, 500, 0.005, 20);
//  for (int i = 0; i < corners.size(); ++i) {
//    circle(region, corners[i], 2, 200);
//  }
//  imshow("Corners",region);

  return maxP2;
}

Point2f findSubpixelEyeCorner(Mat region, Point maxP) {

  Size sizeRegion = region.size();

  // Matrix dichotomy
  // Not useful, matrix becomes too small

  /*int offsetX = 0;
  if(maxP.x - sizeRegion.width / 4 <= 0) {
    offsetX = 0;
  } else if(maxP.x + sizeRegion.width / 4 >= sizeRegion.width) {
    offsetX = sizeRegion.width / 2 - 1;
  } else {
    offsetX = maxP.x - sizeRegion.width / 4;
  }
  int offsetY = 0;
  if(maxP.y - sizeRegion.height / 4 <= 0) {
    offsetY = 0;
  } else if(maxP.y + sizeRegion.height / 4 >= sizeRegion.height) {
    offsetY = sizeRegion.height / 2 - 1;
  } else {
    offsetY = maxP.y - sizeRegion.height / 4;
  }
  Range colRange(offsetX, offsetX + sizeRegion.width / 2);
  Range rowRange(offsetY, offsetY + sizeRegion.height / 2);

  Mat miRegion(region, rowRange, colRange);


if(left){
    imshow("aa",miRegion);
  } else {
    imshow("aaa",miRegion);
  }*/

    Mat cornerMap(sizeRegion.height * 10, sizeRegion.width * 10, CV_32F);

    resize(region, cornerMap, cornerMap.size(), 0, 0, INTER_CUBIC);

    Point maxP2;
    minMaxLoc(cornerMap, NULL,NULL,NULL,&maxP2);

    return Point2f(sizeRegion.width / 2 + maxP2.x / 10,
     sizeRegion.height / 2 + maxP2.y / 10);
  }

/** @function main */
  int main( int argc, const char** argv )
  {

    VideoCapture cap(0);

    Mat frame; 

   //-- 1. Load the cascades
    if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
    if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream

    vector<Mat> imgs(25);
    createCornerKernels();

    while( true )
    {

      cap>>frame;

      for(int i=0;i<imgs.size();i++) {
        frame.copyTo(imgs[i]);
      }

   //-- 3. Apply the classifier to the frame
      if( !frame.empty() )
      { 
        vector<Rect> faces;
        Mat frame_gray;

        cvtColor( frame, frame_gray, CV_BGR2GRAY );
        equalizeHist( frame_gray, frame_gray );

          //-- Detect faces
        face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        //cout<<"here";

        for( size_t i = 0; i < faces.size(); i++ )
        {               
          rectangle( imgs[0], faces[i], Scalar(255,255,255), 2, 8, 0);

          Mat faceROI = frame_gray( faces[i] );
          vector<Rect> eyes;

          //cout<<"faces detected";

    //-- In each face, detect eyes
          eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

          for( size_t j = 0; j < eyes.size(); j++ )
          {  
                /*Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );*/

                /*Mat eyeROI(faceROI,eyes[j]);*/

                rectangle( imgs[0], Point(faces[i].x + eyes[j].x,faces[i].y + eyes[j].y ),Point(faces[i].x + eyes[j].x + eyes[j].width,faces[i].y + eyes[j].y + eyes[j].height) , Scalar(255,255,255), 2, 8, 0 );
                /*
                Point eyeLoc;
                eyeLoc=findEyeCenter(faceROI,eyes[j],"hello");

                Point center= Point(faces[i].x + eyes[j].x+eyeLoc.x,faces[i].y + eyes[j].y+eyeLoc.y);

                circle( imgs[0], center, 1, Scalar( 255, 0, 0 ), -1, 8, 0 );*/

                //cout<<"eyes detected";

                Point2f corner_roi = findEyeCorner(frame_gray(eyes[j]), true, false);
                Point2f corner = Point2f(faces[i].x + eyes[j].x + corner_roi.x, faces[i].y + eyes[j].y + corner_roi.y);
                circle(imgs[0], corner, 4, Scalar(0,0,255), -1, 8, 0);

              }

            }
            flip(imgs[0], imgs[0], 1);

            char str[] = {(char)(j), '\0'};

            imshow("eyes_detect",imgs[0]);
            imshow(str, frame_gray(eyes[j]));

          }

          else{ printf(" --(!) No captured frame -- Break!"); break; }

          int c = waitKey(1);
          if( (char)c == 'c' ) { 
            releaseCornerKernels();
            break; }

        }

        return 0;
      }