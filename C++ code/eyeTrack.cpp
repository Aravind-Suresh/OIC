#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <queue>
#include "constants.h"
#include "helpers.h"

using namespace std;
using namespace cv;

 /** Function Headers */

Point unscalePoint(Point p, Rect origSize) {
  float ratio = (((float)kFastEyeWidth)/origSize.width);
  int x = round(p.x / ratio);
  int y = round(p.y / ratio);
  return Point(x,y);
}



void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
  resize(src, dst, Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows));
}




Mat computeMatXGradient(const Mat &mat) {
  Mat out(mat.rows,mat.cols,CV_64F);
  
  for (int y = 0; y < mat.rows; ++y) {
    const uchar *Mr = mat.ptr<uchar>(y);
    double *Or = out.ptr<double>(y);
    
    Or[0] = Mr[1] - Mr[0];
    for (int x = 1; x < mat.cols - 1; ++x) {
      Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
    }
    Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
  }
  
  return out;
}




double computeDynamicThreshold(const Mat &mat, double stdDevFactor) {
  Scalar stdMagnGrad, meanMagnGrad;
  meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
}




Mat matrixMagnitude(const Mat &matX, const Mat &matY) {
  Mat mags(matX.rows,matX.cols,CV_64F);
  for (int y = 0; y < matX.rows; ++y) {
    const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
    double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < matX.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = sqrt((gX * gX) + (gY * gY));
      Mr[x] = magnitude;
    }
  }
  return mags;
}




bool inMat(cv::Point p,int rows,int cols) {
  return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}




bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
  return inMat(np, mat.rows, mat.cols);
}




Mat floodKillEdges(Mat &mat) {
  rectangle(mat,Rect(0,0,mat.cols,mat.rows),255);
  
  Mat mask(mat.rows, mat.cols, CV_8U, 255);
  queue<Point> toDo;
  toDo.push(Point(0,0));
  while (!toDo.empty()) {
    Point p = toDo.front();
    toDo.pop();
    if (mat.at<float>(p) == 0.0f) {
      continue;
    }
    // add in every direction
    Point np(p.x + 1, p.y); // right
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x - 1; np.y = p.y; // left
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y + 1; // down
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    np.x = p.x; np.y = p.y - 1; // up
    if (floodShouldPushPoint(np, mat)) toDo.push(np);
    // kill it
    mat.at<float>(p) = 0.0f;
    mask.at<uchar>(p) = 0;
  }
  return mask;
}




void testPossibleCentersFormula(int x, int y, const Mat &weight,double gx, double gy, Mat &out) {
  // for all possible centers
  for (int cy = 0; cy < out.rows; ++cy) {
    double *Or = out.ptr<double>(cy);
    const unsigned char *Wr = weight.ptr<unsigned char>(cy);
    for (int cx = 0; cx < out.cols; ++cx) {
      if (x == cx && y == cy) {
        continue;
      }
      // create a vector from the possible center to the gradient origin
      double dx = x - cx;
      double dy = y - cy;
      // normalize d
      double magnitude = sqrt((dx * dx) + (dy * dy));
      dx = dx / magnitude;
      dy = dy / magnitude;
      double dotProduct = dx*gx + dy*gy;
      dotProduct = max(0.0,dotProduct);
      // square and multiply by the weight
      if (kEnableWeight) {
        Or[cx] += dotProduct * dotProduct * (Wr[cx]/kWeightDivisor);
      } else {
        Or[cx] += dotProduct * dotProduct;
      }
    }
  }
}

Point findEyeCenter(Mat face, Rect eye, string debugWindow) {
  Mat eyeROIUnscaled = face(eye);
  Mat eyeROI;
  scaleToFastSize(eyeROIUnscaled, eyeROI);
  // draw eye region
  rectangle(face,eye,1234);
  //-- Find the gradient
  Mat gradientX = computeMatXGradient(eyeROI);
  Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
  Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
  double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
  //double gradientThresh = kGradientThreshold;
  //double gradientThresh = 0;
  //normalize
  for (int y = 0; y < eyeROI.rows; ++y) {
    double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    const double *Mr = mags.ptr<double>(y);
    for (int x = 0; x < eyeROI.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      double magnitude = Mr[x];
      if (magnitude > gradientThresh) {
        Xr[x] = gX/magnitude;
        Yr[x] = gY/magnitude;
      } else {
        Xr[x] = 0.0;
        Yr[x] = 0.0;
      }
    }
  }
  
  //imshow(debugWindow,gradientX);
  
  //-- Create a blurred and inverted image for weighting
  Mat weight;
  GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
  for (int y = 0; y < weight.rows; ++y) {
    unsigned char *row = weight.ptr<unsigned char>(y);
    for (int x = 0; x < weight.cols; ++x) {
      row[x] = (255 - row[x]);
    }
  }

  //-- Run the algorithm!
  Mat outSum = Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  
  for (int y = 0; y < weight.rows; ++y) {
    const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
    for (int x = 0; x < weight.cols; ++x) {
      double gX = Xr[x], gY = Yr[x];
      if (gX == 0.0 && gY == 0.0) {
        continue;
      }
      testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
    }
  }
  // scale all the values down, basically averaging them
  double numGradients = (weight.rows*weight.cols);
  Mat out;
  outSum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
  Point maxP;
  double maxVal;
  minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
  if(kEnablePostProcess) {
    Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
    double floodThresh = maxVal * kPostProcessThreshold;
    threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
    if(kPlotVectorField) {
      //plotVecField(gradientX, gradientY, floodClone);
      imwrite("eyeFrame.png",eyeROIUnscaled);
    }
    Mat mask = floodKillEdges(floodClone);
    
    // redo max
    minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
  }
  return unscalePoint(maxP, eye);
}


void showImages(int e ,int l, int h, vector<Mat> imgs) {
  for(int i=l;i<=h;i++) {
    char str[2];
    str[0] = (char)(i+49+e*(e+2));
    str[1] = '\0';
    //cout<<endl<<str;
    imshow(str, imgs[i]);
  }
}

 /** Global variables */
String face_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "/usr/share/opencv/haarcascades/haarcascade_eye.xml" ;
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";



 /** @function main */
int main( int argc, const char** argv )
{

  VideoCapture cap(0);
  
  Mat frame; 
  
   //-- 1. Load the cascades
  if( !face_cascade.load( face_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };
  if( !eyes_cascade.load( eyes_cascade_name ) ){ printf("--(!)Error loading\n"); return -1; };

   //-- 2. Read the video stream

  while( true )
  {

    cap>>frame;
    vector<Mat> imgs(25);

    for(int i=0;i<imgs.size();i++) {
      frame.copyTo(imgs[i]);
    }

    vector<Mat> rois(25);

   //-- 3. Apply the classifier to the frame
    if( !frame.empty() )
    { 
      vector<Rect> faces;
      Mat frame_gray;

      cvtColor( frame, frame_gray, CV_BGR2GRAY );
      equalizeHist( frame_gray, frame_gray );

          //-- Detect faces
      face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );

      for( size_t i = 0; i < faces.size(); i++ )
      {               
        rectangle( imgs[0], faces[i], Scalar(255,255,255), 2, 8, 0);

        Mat faceROI = frame_gray( faces[i] );
        vector<Rect> eyes;

    //-- In each face, detect eyes
        eyes_cascade.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CV_HAAR_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {  
                /*Point center( faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5 );
                int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
                circle( frame, center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );*/

                Mat eyeROI(faceROI,eyes[j]);

                rectangle( imgs[0], Point(faces[i].x + eyes[j].x,faces[i].y + eyes[j].y ),Point(faces[i].x + eyes[j].x + eyes[j].width,faces[i].y + eyes[j].y + eyes[j].height) , Scalar(255,255,255), 2, 8, 0 );

                Point eyeLoc;
                eyeLoc=findEyeCenter(faceROI,eyes[j],"hello");

                Point center= Point(faces[i].x + eyes[j].x+eyeLoc.x,faces[i].y + eyes[j].y+eyeLoc.y);

                circle( imgs[0], center, 3, Scalar( 255, 0, 0 ), -1, 8, 0 );

                center= Point(eyeLoc.x,eyeLoc.y);

                circle( eyeROI, center, 3, Scalar( 255, 0, 0 ), -1, 8, 0 );
                char str[] = {(char)(j), '\0'};                
                imshow(str, eyeROI);

              }

            }
            flip(imgs[0], imgs[0], 1);

            imshow("eyes_detect",imgs[0]);   

          }

          else{ printf(" --(!) No captured frame -- Break!"); break; }

          int c = waitKey(1);
          if( (char)c == 'c' ) { break; }

        }

        return 0;
      }