#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

/*
	**************************************************************************************************************
	Function to remove the patch of continuous points of white color, when a vector of points are given.

	parameters :
		src - the source Mat GrayScale image.
		dst - the destination Mat image, returned by reference.
		points - vector containing the known patch points.

	return value :
		As mentioned above, dst is returned by reference.

	**************************************************************************************************************
*/

void deletePatch(Mat src, Mat& dst, vector<Point> points) {

}

/*
	**************************************************************************************************************
	Function to return the coordinates of the center of the eyeball and its radius wrt to the roi image given.

	parameters :
		src - the Mat image(RGB) of the eye --> detected using haar/ by some other means.
		center - center of the eyeball, returned by reference.
		radius - radius of the eyeball, returned by reference.

	return value :
		As mentioned above, center and radius are returned by reference.

	**************************************************************************************************************
*/

void getEyeBallCoordinates(Mat src, Point& center, float& radius) {
	Mat roi_invert, roi_inv_gray, roi_eye_ball;

	roi_invert = Scalar::all(255) - src;
	cvtColor(roi_invert, roi_inv_gray, CV_RGB2GRAY);
	imshow("invert", roi_invert);
	imshow("gray", roi_inv_gray);

	threshold(roi_inv_gray, roi_inv_gray, 220, 255, THRESH_BINARY);
	imshow("thresh", roi_inv_gray);

	Mat img_dt(roi_inv_gray.rows, roi_inv_gray.cols, CV_32F, Scalar::all(0));
	Mat img_out(roi_inv_gray.rows, roi_inv_gray.cols, CV_32F, Scalar::all(0));

	distanceTransform(roi_inv_gray, img_dt, CV_DIST_L2, 3);
	normalize(img_dt, img_dt, 0.1, 1, NORM_MINMAX);

	threshold(img_dt, img_dt, (200/255.0), (255/255.0), THRESH_BINARY);
	
	/*
	int erosion_type[] = {0,1,2};
	int erosion_size = 1;

	Mat erosion_element = getStructuringElement( erosion_type[1],
		Size( 2*erosion_size + 1, 2*erosion_size+1 ),
		Point( erosion_size, erosion_size ) );

	erode(img_dt, img_dt, erosion_element);

	int dilation_type[] = {0,1,2};
	int dilation_size = 3;

	Mat dilation_element = getStructuringElement( dilation_type[0],
		Size( 2*dilation_size + 1, 2*dilation_size+1 ),
		Point( dilation_size, dilation_size ) );

	dilate(img_dt, img_dt, dilation_element);
	*/

	imshow("dt", img_dt);

	/*
	img_dt.convertTo(img_dt, CV_8UC1, 255.0);
	roi_eye_ball = roi_inv_gray - img_dt;
	*/

	//imshow("roi_eye_ball", roi_eye_ball);

}

int main(int argc, char** argv) {
	Mat img_rgb = imread(argv[1], CV_LOAD_IMAGE_COLOR);
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	vector<Mat> imgs(25);
	vector<Mat> rois(2);

	for(int i=0;i<imgs.size();i++) {
	 	img_gray.copyTo(imgs[i]);
	}

	img_rgb.copyTo(rois[0]);

	//Complete the code for getting ROI from the input image

	Point center;
	float radius;

	getEyeBallCoordinates(rois[0], center, radius);

	waitKey(0);
	return 0;
}