
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/opencv/cv_image.h>
#include <iostream>
#include <opencv2/opencv.hpp>

/*#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/image_processing.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/opencv/cv_image.h"
#include <opencv2/core/core_c.h> // shame, but needed for using dlib
#include "dlib/image_processing.h"
#include "dlib/opencv/cv_image.h"
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <cv.h>
#include <highgui.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <opencv2/ml/ml.hpp>
#include <list>*/

using namespace dlib;
using namespace std;
using namespace cv;

void matToArray2d(Mat src, dlib::array2d<uchar>& dst) {
	for(int i=0;i<src.rows;i++) {
		for(int j=0;j<src.cols	;j++) {
			dst[i][j] = src.at<uchar>(i,j);
		}
		if(i==src.rows - 1) {
			cout<<"converted";
		}
	}
}

int main(int argc, char** argv) {
	VideoCapture cap(0);

	frontal_face_detector detector = get_frontal_face_detector();
	dlib::image_window win;
	Mat frame;

	while(true) {
		cap>>frame;
		cvtColor(frame, frame, CV_BGR2GRAY);

		dlib::array2d<uchar> img(frame.rows, frame.cols);
		//img = cv_image<uchar>(frame);
		matToArray2d(frame, img);

		//pyramid_up(img);
		std::vector<dlib::rectangle> dets = detector(img);

		//cout<<dets.size();

		pyramid_down<2> pyr;

		for (int i = 0; i < dets.size(); ++i)
		{

			dlib::rectangle orig_scale=pyr.rect_down(dets[i]);
			cout<<"face number : "<<i<<endl;
			cout<<"     top left "<<orig_scale.left()<<" "<<orig_scale.top()<<endl;
			cout<<"     width "<<orig_scale.width()<<endl;
			cout<<"     height "<<orig_scale.height()<<endl;

		}

		cout << "Number of faces detected: " << dets.size() << endl;
            // Now we show the image on the screen and the face detections as
            // red overlay boxes.
		win.clear_overlay();
		win.set_image(img);
		win.add_overlay(dets, rgb_pixel(255,0,0));

	}
}