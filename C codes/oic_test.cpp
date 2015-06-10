#include<opencv2/opencv.hpp>
#include<iostream>
#include<string.h>
#include<stdlib.h>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
	Mat img_gray = imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);

	vector<Mat> imgs(25);
	for(int i=0;i<imgs.size();i++) {
		img_gray.copyTo(imgs[i]);
	}

	imgs[0] = 
}