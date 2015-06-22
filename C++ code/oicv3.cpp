#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/legacy/compat.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <math.h>
#include <stdlib.h>
#include <string>

double PI = 3.141592653589;
double Rn = 0.5;
double Rm = 0.5;

using namespace dlib;
using namespace std;

double vectorMagnitude(double vec[], int size) {
	double mag = 0;

	for(int i=0;i<size;i++) {
		mag += vec[i]*vec[i];
	}

	return sqrt(mag);
}

cv::Point get_mid_point(cv::Point p1, cv::Point p2) {
	return cv::Point((p1.x + p2.x)/2.0, (p1.y + p2.y)/2.0);
}

double find_sigma(int ln, int lf, double Rn, double theta)
{
	double dz=0;
	double sigma;
	double m1 = ((double)ln*ln)/((double)lf*lf);
	double m2 = (cos(theta))*(cos(theta));

	if (m2 == 1)
	{
		dz = sqrt(	(Rn*Rn)/(m1 + (Rn*Rn))	);
	}
	if (m2>=0 && m2<1)
	{
		dz = sqrt(	((Rn*Rn) - m1 - 2*m2*(Rn*Rn) + sqrt(	((m1-(Rn*Rn))*(m1-(Rn*Rn))) + 4*m1*m2*(Rn*Rn)	))/ (2*(1-m2)*(Rn*Rn))	);
	}
	sigma = acos(dz);
	return sigma;
}

double get_distance(cv::Point p1, cv::Point p2) {
	double x = p1.x - p2.x;
	double y = p1.y - p2.y;

	return sqrt(x*x + y*y);
}

double get_angle_between(cv::Point pt1, cv::Point pt2)
{
	return 360 - cvFastArctan(pt2.y - pt1.y, pt2.x - pt1.x);
}

void draw_facial_normal(cv::Mat img, double normal[3], double sigma, double theta) {

	cv::Point origin = cv::Point(50,50);
	cv::Scalar colour = cv::Scalar(255);

	cv::Point projection_2d;
	projection_2d.x = origin.x + cvRound(60*(normal[0]));
	projection_2d.y = origin.y + cvRound(60*(normal[1]));

	if (normal[0] > 0 && normal[1] < 0)
	{
		cv::ellipse(img, origin, cv::Size(25,std::abs(cvRound(25-sigma*(180/(2*PI))))) , std::abs(180-(theta*(180/PI))), 0, 360, colour, 2,4,0);
	}
	else
	{
		cv::ellipse(img, origin, cv::Size(25,std::abs(cvRound(25-sigma*(180/(2*PI))))) , std::abs(theta*(180/PI)), 0, 360, colour, 2,4,0);
	}

	cv::line(img, origin, projection_2d, colour, 2, 4, 0);
}

void draw_crosshair(cv::Mat img, CvPoint centre, int circle_radius, int line_radius)
{
	cv::Point pt1,pt2,pt3,pt4;
	cv::Scalar colour(255);

	pt1.x = centre.x;
	pt2.x = centre.x;
	pt1.y = centre.y - line_radius;
	pt2.y = centre.y + line_radius;
	pt3.x = centre.x - line_radius;
	pt4.x = centre.x + line_radius;
	pt3.y = centre.y;
	pt4.y = centre.y;


	cv::circle(img, centre, circle_radius, colour, 2, 4, 0);

	cv::line(img, pt1, pt2, colour, 1, 4, 0);
	cv::line(img, pt3, pt4, colour, 1, 4, 0);

}

struct FaceFeatures {
	cv::Point face_centre;

	cv::Point left_eye;
	cv::Point right_eye;
	cv::Point mid_eye;

	cv::Point nose_base;
	cv::Point nose_tip;

	cv::Point mouth;

	void assign(cv::Point c_face_centre, cv::Point c_left_eye, cv::Point c_right_eye, cv::Point c_nose_tip, cv::Point c_mouth) {
		face_centre = c_face_centre;
		left_eye = c_left_eye;
		right_eye = c_right_eye;
		nose_tip = c_nose_tip;
		mouth = c_mouth;

		mid_eye.x = (left_eye.x + right_eye.x)/2.0;
		mid_eye.y = (left_eye.y + right_eye.y)/2.0;

		//Find the nose base along the symmetry axis
		nose_base.x = mouth.x + (mid_eye.x - mouth.x)*Rm;
		nose_base.y = mouth.y - (mouth.y - mid_eye.y)*Rm;
	}
};

struct FaceData {
	double left_eye_nose_distance;
	double right_eye_nose_distance;
	double left_eye_right_eye_distance;
	double nose_mouth_distance;

	double mid_eye_mouth_distance;	//Lf
	double nose_base_nose_tip_distance;	//Ln

	//double mean_face_feature_distance;

/*	double init_left_eye_nose_distance;
	double init_right_eye_nose_distance;
	double init_left_eye_right_eye_distance;
	double init_nose_mouth_distance;

	double init_mean_face_feature_distance;*/

	void assign(FaceFeatures* f) {
		left_eye_nose_distance = get_distance(f->left_eye, f->nose_base);
		right_eye_nose_distance = get_distance(f->right_eye, f->nose_base);
		left_eye_right_eye_distance = get_distance(f->left_eye, f->right_eye);
		nose_mouth_distance = get_distance(f->nose_base, f->mouth);

		mid_eye_mouth_distance = get_distance(f->mid_eye, f->mouth);
		nose_base_nose_tip_distance = get_distance(f->nose_tip, f->nose_base);
	}
};

struct FacePose {
	double theta, tau;
	double sigma, symm_x;

	double normal[3];	//Vector for storing Facial normal

	double yaw, pitch;

	double kalman_pitch, kalman_yaw;
	double kalman_pitch_pre, kalman_yaw_pre;

	void assign(FaceFeatures* f, FaceData* d) {
		symm_x = get_angle_between(f->nose_base, f->mid_eye);
			//symm angle - angle between the symmetry axis and the 'x' axis 
		tau = get_angle_between(f->nose_base, f->nose_tip);
			//tilt angle - angle between normal in image and 'x' axis
		theta = (abs(tau - symm_x)) * (PI/180.0);
			//theta angle - angle between the symmetry axis and the image normal
		sigma = find_sigma(d->nose_base_nose_tip_distance, d->mid_eye_mouth_distance, Rn, theta);
		std::cout<<"symm : "<<symm_x<<" tau : "<<tau<<" theta : "<<theta<<" sigma : "<<sigma<<" ";

		normal[0] = (sin(sigma))*(cos((360 - tau)*(PI/180.0)));
		normal[1] = (sin(sigma))*(sin((360 - tau)*(PI/180.0)));
		normal[2] = -cos(sigma);

		kalman_pitch_pre = pitch;
		pitch = acos(sqrt((normal[0]*normal[0] + normal[2]*normal[2])/(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
		if((f->nose_tip.y - f->nose_base.y) < 0) {
			pitch = -pitch;
		}

		kalman_yaw_pre = yaw;
		yaw = acos((abs(normal[2]))/(sqrt(normal[0]*normal[0] + normal[1]*normal[1] + normal[2]*normal[2])));
		if((f->nose_tip.x - f->nose_base.x) < 0) {
			yaw = -yaw;
		}
	}

};

int main(int argc, char **argv) {
	try
	{
		Rm = std::atoi(argv[1])/100.0;
		Rn = std::atoi(argv[2])/100.0;
		std::cout<<"Rm : "<<Rm<<" Rn : "<<Rn;

		cv::VideoCapture cap(0);
		image_window win;

		FaceFeatures *face_features = new FaceFeatures();
		FaceData *face_data = new FaceData();
		FacePose *face_pose = new FacePose();

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		while(!win.is_closed())
		{
			cv::Mat temp;
			cap >> temp;
			cv::flip(temp, temp, 1);

			cv_image<bgr_pixel> cimg(temp);

			std::vector<rectangle> faces = detector(cimg);

			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			if(shapes.size() == 0) {
				std::cout<<"zero faces";
			}
			else {

				face_features->assign(cv::Point(0,0),
					get_mid_point(cv::Point(shapes[0].part(42).x(), shapes[0].part(42).y()),
						cv::Point(shapes[0].part(45).x(), shapes[0].part(45).y())),
					get_mid_point(cv::Point(shapes[0].part(36).x(), shapes[0].part(36).y()),
						cv::Point(shapes[0].part(39).x(), shapes[0].part(39).y())),
					cv::Point(shapes[0].part(30).x(), shapes[0].part(30).y()), 
					get_mid_point(cv::Point(shapes[0].part(48).x(), shapes[0].part(48).y()),
						cv::Point(shapes[0].part(54).x(), shapes[0].part(54).y())));

				face_data->assign(face_features);

				face_pose->assign(face_features, face_data);

				/*
				cv::circle(temp, face_features->mid_eye, 2, cv::Scalar(255), 2, 4, 0);
				cv::circle(temp, face_features->mouth, 2, cv::Scalar(255), 2, 4, 0);
				cv::circle(temp, face_features->nose_base, 2, cv::Scalar(255), 2, 4, 0);
				cv::circle(temp, face_features->nose_tip, 2, cv::Scalar(255), 2, 4, 0);
				*/

				double del_x = 100*face_pose->normal[0];
				double del_y = 100*face_pose->normal[1];

				cv::line(temp, cv::Point(shapes[0].part(30).x(), shapes[0].part(30).y()),
					cv::Point(shapes[0].part(30).x() + del_x, shapes[0].part(30).y() + del_y), cv::Scalar(0), 3);

				std::cout<<"magnitude : "<<vectorMagnitude(face_pose->normal, 3)<<" ";
				std::cout<<face_pose->normal[0]<<", "<<face_pose->normal[1]<<", "<<face_pose->normal[2];
				std::cout<<"  pitch "<<face_pose->pitch*180.0/PI<<" , yaw  "<<face_pose->yaw*180.0/PI<<std::endl;

				draw_facial_normal(temp, face_pose->normal, face_pose->sigma, face_pose->theta);

			//draw_crosshair(temp, pointer_2d_kalman, 7, 12);
			//cv::circle(temp, pointer_2d, 7, CV_RGB(80,80,80), 2, 4, 0);
			}
			win.clear_overlay();
			win.set_image(cimg);
			win.add_overlay(render_face_detections(shapes));
		}
	}
	catch(serialization_error& e)
	{
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch(exception& e)
	{
		cout << e.what() << endl;
	}
}