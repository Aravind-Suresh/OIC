
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

int main(int argc, char** argv) {

	frontal_face_detector detector = get_frontal_face_detector();
	dlib::image_window win;
	Mat frame = imread(argv[1]);
	dlib::array2d<rgb_pixel> img;

	assign_image(img, dlib::cv_image<bgr_pixel>(frame));

	std::vector<dlib::rectangle> dets = detector(img);

	cout<<dets.size();

	cout << "Number of faces detected: " << dets.size() << endl;

        // Now we show the image on the screen and the face detections as
        // red overlay boxes.
	win.clear_overlay();
	win.set_image(img);
	win.add_overlay(dets, rgb_pixel(255,0,0));
		//win.add_overlay(render_face_detections(shapes));
	cin.get();

}
