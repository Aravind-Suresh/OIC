
/*

#include <opencv2/opencv.hpp>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;
int main(int argc, char** argv) {

	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);

	
   	SiftFeatureDetector detector;
    vector<KeyPoint> keypoints;
    detector.detect(img_gray, keypoints);

    // Add results to image and save.
    Mat output;
    drawKeypoints(img_gray, keypoints, output);
    imwrite("sift_result.jpg", output);


    waitKey(0);                   
    return 0;
    }
*/

   #include <stdio.h>
   #include <iostream>
   #include "opencv2/core/core.hpp"
   #include "opencv2/features2d/features2d.hpp"
   #include "opencv2/nonfree/features2d.hpp"
   #include "opencv2/highgui/highgui.hpp"
   #include "opencv2/nonfree/nonfree.hpp"

   using namespace cv;
   using namespace std;

   /** @function main */
   int main( int argc, char** argv )
   {

     Mat img_1 = imread( argv[1], CV_LOAD_IMAGE_GRAYSCALE );
    
     
     //-- Step 1: Detect the keypoints using SURF Detector
     int minHessian = 400;

     SurfFeatureDetector detector( minHessian );

     std::vector<KeyPoint> keypoints_1, keypoints_2;

     detector.detect( img_1, keypoints_1 );
     
     //-- Draw keypoints
     Mat img_keypoints_1; 

     drawKeypoints( img_1, keypoints_1, img_keypoints_1, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
     
     //-- Show detected (drawn) keypoints
     imshow("Keypoints 1", img_keypoints_1 );
     
     waitKey(0);

     return 0;
     }