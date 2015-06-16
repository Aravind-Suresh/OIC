
#include "dlib/image_processing/frontal_face_detector.h"
#include "dlib/gui_widgets.h"
#include "dlib/image_io.h"
#include "dlib/opencv/cv_image.h"
#include "dlib/image_transforms/assign_image.h"
#include "dlib/image_processing/render_face_detections.h"
#include "dlib/pixel.h"
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

const std::string currentDateTime() {
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = *localtime(&now);
	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return buf;
}


int main(int argc, char** argv) {
	VideoCapture cap(0);

	frontal_face_detector detector = get_frontal_face_detector();
	//dlib::image_window win;
	Mat frame;
	dlib::array2d<rgb_pixel> img;

	shape_predictor sp;
	deserialize("shape_predictor_68_face_landmarks.dat") >> sp;


	pyramid_down<2> pyr;
	std::vector<full_object_detection> shapes;

	int i=0;

	while(true) {
		cap>>frame;

		assign_image(img, dlib::cv_image<bgr_pixel>(frame));

		//pyramid_up(img);
		//std::cout << "time before" << currentDateTime() << std::endl;

		std::vector<dlib::rectangle> dets = detector(img);
		/*if(i%10==0) {
			std::cout << "time" << currentDateTime() << std::endl;
			i++;
		}*/
		//std::cout << "time after" << currentDateTime() << std::endl;

		//cout<<dets.size();

		/*for (int i = 0; i < dets.size(); ++i)
		{

			//dlib::rectangle orig_scale=pyr.rect_down(dets[i]);
			cout<<"face number : "<<i<<endl;
			cout<<"     top left "<<orig_scale.left()<<" "<<orig_scale.top()<<endl;
			cout<<"     width "<<orig_scale.width()<<endl;
			cout<<"     height "<<orig_scale.height()<<endl;

			full_object_detection shape = sp(img, dets[i]);

			cout << "number of parts: "<< shape.num_parts() << endl;
			cout << "pixel position of first part:  " << shape.part(0) << endl;
			cout << "pixel position of second part: " << shape.part(1) << endl;

			shapes.push_back(shape);

		}*/

		cout << "Number of faces detected: " << dets.size() << endl;

		imshow("frame",frame);
		waitKey(1);

        // Now we show the image on the screen and the face detections as
        // red overlay boxes.
		//win.clear_overlay();
		//win.set_image(img);
		//win.add_overlay(dets, rgb_pixel(255,0,0));
		//win.add_overlay(render_face_detections(shapes));

	}
}