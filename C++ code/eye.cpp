 #include <stdio.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>

 using namespace std;
 using namespace cv;

 /** Function Headers */
 
void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
  //blur( src_gray, edge_gray, Size(3,3) );
  GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
  Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
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


 int edgeThresh = 1;     
 int lowThreshold = 30;
 int const max_lowThreshold = 100;
 int ratio = 1;
 int kernel_size = 3;
 int thresh = 200;


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

  /*
   *  0 - face detected
   *  1 - + canny
   *  2 - + laplacian
   *  3 - 
   *  4 - 
   *  5 - 
   *  6 - 
   *  7 - 
   *  8 - 
   *  9 - 
   */

   for(int i=0;i<imgs.size();i++) {
    frame.copyTo(imgs[i]);
   }

      vector<Mat> rois(25);
  /*
   *  0 - +laplacian
   *  1 - +canny
   *  2 - + harris
   *  3 - 
   *  4 - 
   *  5 - 
   *  6 - 
   *  7 - 
   *  8 -
   *  9 - 
   */
  
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

                Mat eye_roi(faceROI,eyes[j]);

                rectangle( imgs[0], Point(faces[i].x + eyes[j].x,faces[i].y + eyes[j].y ),Point(faces[i].x + eyes[j].x + eyes[j].width,faces[i].y + eyes[j].y + eyes[j].height) , Scalar(255,255,255), 2, 8, 0 );

//roi laplacian
                Laplacian(eye_roi,rois[0], CV_8UC1, 3);
    
//roi canny edge
                CannyThreshold(eye_roi, rois[1], lowThreshold, lowThreshold*ratio, kernel_size);

//roi Harris corner
                Mat dst, dst_norm, dst_norm_scaled;
                dst = Mat::zeros( eye_roi.size(), CV_32FC1 );

      /// Detector parameters
                int blockSize = 2;
                int apertureSize = 3;
                double k = 0.04;

      /// Detecting corners
                cornerHarris( eye_roi, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

    /// Normalizing
                normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
                convertScaleAbs( dst_norm, dst_norm_scaled );

    
                dst_norm_scaled.copyTo(rois[2]);

    /// Drawing a circle around corners
                for( int p = 0; p < dst_norm.rows ; p++ )
                { 
                  for( int q = 0; q < dst_norm.cols; q++ )
                    {
        
                      if( (int) dst_norm.at<float>(p,q) > thresh )
                        {
                          circle( rois[2], Point( q, p ), 1,  Scalar(0), 2, 8, 0 );
                          circle( imgs[1], Point(q+eyes[j].x+faces[i].x,p+eyes[j].y+faces[i].y) ,1,Scalar(0),2,8,0);
                        }
        
                    }
      
                  }
              
              showImages(j,0,2,rois);

              }
            
            }
  
      //imshow( window_name, frame ); }
      imshow("eyes_detect",imgs[0]);   
      imshow("eyes_detect_corner",imgs[1]);

        }

       else{ printf(" --(!) No captured frame -- Break!"); break; }

        int c = waitKey(1);
        if( (char)c == 'c' ) { break; }
      
      }
   
   return 0;
 }

