#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

void CannyThreshold(Mat src_gray, Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
	//blur( src_gray, edge_gray, Size(3,3) );
	GaussianBlur( src_gray, edge_gray, Size(5,5), 2, 2 );
	Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}

void showImages(int l, int h, vector<Mat> imgs) {
	for(int i=l;i<=h;i++) {
		char str[2];
		str[0] = (char)(i+49);
		str[1] = '\0';
		imshow(str, imgs[i]);
	}
}


int main(int argc, char** argv) {

	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_hsv(img_gray.rows, img_gray.cols, CV_8UC1, Scalar::all(0));

	int edgeThresh = 1; 	 	
	int lowThreshold = 30;
	int const max_lowThreshold = 100;
	int ratio = 1;
	int kernel_size = 3;
	int thresh = 200;



	//Storage for all Mat images used during this program execution
	vector<Mat> imgs(25);

	/*
	 *	0 - face detected
	 *	1 - + canny
	 *	2 - + laplacian
	 *	3 - 
	 *	4 - 
	 *	5 - 
	 *	6 - 
	 *	7 - 
	 *	8 - 
	 *	9 - 
	 *
	 *
	 *
	 */

	 for(int i=0;i<imgs.size();i++) {
	 	img_gray.copyTo(imgs[i]);
	 }
	vector<Mat> rois(25);
	/*
	 *	0 - +laplacian
	 *	1 - +canny
	 *	2 - + harris
	 *	3 - 
	 *	4 - 
	 *	5 - 
	 *	6 - 
	 *	7 - 
	 *	8 -
	 *	9 - 
	 *
	 *
	 *
	 */

	 for(int i=0;i<rois.size();i++) {
	 	img_gray.copyTo(rois[i]);
	 }



// Load eyes cascade (.xml file)
    CascadeClassifier eye_cascade;
    eye_cascade.load( "/usr/share/opencv/haarcascades/haarcascade_eye.xml" );
	
	vector<Rect> eyes;
    eye_cascade.detectMultiScale( img_gray, eyes, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE, Size(30, 30) );
    cout<<eyes.size()<<endl;

 	Mat roi_lapl,roi_edges;
    
    // Draw the detected faces
    for( int e= 0;  e< eyes.size(); e++ )
    {
    	//cout<<eyes[e].x<<" "<<eyes[e].y;
    	rectangle( imgs[0], eyes[e], Scalar(255,255,255), 5, 8, 0);
    	Mat roi (img_gray, eyes[e] );
     	
//roi laplacian
     	Laplacian(roi,rois[0], CV_8UC1, 3);
		
//roi canny edge
		CannyThreshold(roi, rois[1], lowThreshold, lowThreshold*ratio, kernel_size);
		
//roi Harris corner
		Mat dst, dst_norm, dst_norm_scaled;
  		dst = Mat::zeros( roi.size(), CV_32FC1 );

  		/// Detector parameters
  		int blockSize = 2;
  		int apertureSize = 3;
 	 	double k = 0.04;

  		/// Detecting corners
 		 cornerHarris( roi, dst, blockSize, apertureSize, k, BORDER_DEFAULT );

		/// Normalizing
		normalize( dst, dst_norm, 0, 255, NORM_MINMAX, CV_32FC1, Mat() );
		convertScaleAbs( dst_norm, dst_norm_scaled );

		
		dst_norm_scaled.copyTo(rois[2]);

		/// Drawing a circle around corners
		for( int j = 0; j < dst_norm.rows ; j++ )
     		{ 
     			for( int i = 0; i < dst_norm.cols; i++ )
          	{
            	
            	if( (int) dst_norm.at<float>(j,i) > thresh )
              {
                circle( rois[2], Point( i, j ), 2,  Scalar(0), 2, 8, 0 );
              }
          	
          	}
     		
     		}

	showImages(0,rois.size(),rois);


	}
     
   	imshow("eyes_detect",imgs[0]);

    waitKey(0);                   
    return 0;
}
