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
double Wf = 0.6;
double Cf_left = 10;
double Cf_right = 10;


using namespace dlib;
using namespace std;

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
	float ratio = (((float)(50))/origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
	cv::resize(src, dst, cv::Size(50,(((float)50)/src.cols) * src.rows));
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
	cv::Mat out(mat.rows,mat.cols,CV_64F);

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

double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
	cv::Scalar stdMagnGrad, meanMagnGrad;
	meanStdDev(mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
	cv::Mat mags(matX.rows,matX.cols,CV_64F);
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

cv::Mat floodKillEdges(cv::Mat &mat) {
	cv::rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);

	cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
	std::queue<cv::Point> toDo;
	toDo.push(cv::Point(0,0));
	while (!toDo.empty()) {
		cv::Point p = toDo.front();
		toDo.pop();
		if (mat.at<float>(p) == 0.0f) {
			continue;
		}
    // add in every direction
    cv::Point np(p.x + 1, p.y); // right
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

void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
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
			if (true) {
				Or[cx] += dotProduct * dotProduct * (Wr[cx]);
			} else {
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}

cv::Point findEyeCenter(cv::Mat eye_mat,cv::Rect eye, string debugWindow) {
	cv::Mat eyeROIUnscaled = eye_mat;
	cv::Mat eyeROI;
	scaleToFastSize(eyeROIUnscaled, eyeROI);
  // draw eye region
  //rectangle(face,eye,1234);
  //-- Find the gradient
	cv::Mat gradientX = computeMatXGradient(eyeROI);
	cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
  //-- Normalize and threshold the gradient
  // compute all the magnitudes
	cv::Mat mags = matrixMagnitude(gradientX, gradientY);
  //compute the threshold
	double gradientThresh = computeDynamicThreshold(mags, 50.0);
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
	cv::Mat weight;
	GaussianBlur( eyeROI, weight, cv::Size( 5, 5 ), 0, 0 );
	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x) {
			row[x] = (255 - row[x]);
		}
	}

  //-- Run the algorithm!
	cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
  // for each possible gradient location
  // Note: these loops are reversed from the way the paper does them
  // it evaluates every possible center for each gradient location instead of
  // every possible gradient location for every center.
	//printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);

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
	cv::Mat out;
	outSum.convertTo(out, CV_32F,1.0/numGradients);
  //imshow(debugWindow,out);
  //-- Find the maximum point
	cv::Point maxP;
	double maxVal;
	minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
  //-- Flood fill the edges
	cv::Mat floodClone;
    //double floodThresh = computeDynamicThreshold(out, 1.5);
	double floodThresh = maxVal * 0.97;
	threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);

	cv::Mat mask = floodKillEdges(floodClone);

    // redo max
	minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);


	return unscalePoint(maxP, eye);

}

void preprocessROI(cv::Mat& roi_eye) {
	GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
	equalizeHist( roi_eye, roi_eye );
}

void showImages(int e ,int l, int h, std::vector<cv::Mat> imgs) {
	for(int i=l;i<=h;i++) {
		char str[2];
		str[0] = (char)(i+49+e*(e+2));
		str[1] = '\0';
    //cout<<endl<<str;
		cv::imshow(str, imgs[i]);
	}
}

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

void project_facial_pose(cv::Mat img, double normal[3], double sigma, double theta) {

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
		//std::cout<<"symm : "<<symm_x<<" tau : "<<tau<<" theta : "<<theta<<" sigma : "<<sigma<<" ";

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

void draw_facial_normal(cv::Mat& img, dlib::full_object_detection shape, std::vector<double> normal) {

	double del_x = 100*normal[0];
	double del_y = 100*normal[1];

	cv::line(img, cv::Point(shape.part(30).x(), shape.part(30).y()),
		cv::Point(shape.part(30).x() + del_x, shape.part(30).y() + del_y), cv::Scalar(0), 3);

	//std::cout<<"magnitude : "<<vectorMagnitude(f->normal, 3)<<" ";
	//std::cout<<f->normal[0]<<", "<<f->normal[1]<<", "<<f->normal[2];
	//std::cout<<"  pitch "<<f->pitch*180.0/PI<<" , yaw  "<<f->yaw*180.0/PI<<std::endl;

}

double scalarProduct(std::vector<double> vec1, std::vector<double> vec2) {
	double dot = 0;

	if(vec1.size() != vec2.size()) {
		return 0;
	}

	for(int i=0;i<vec1.size();i++) {
		dot += vec1[i]*vec2[i];
	}

	return dot;
}

cv::Mat get_rotation_matrix_z(double theta) {
	cv::Mat rot_matrix(3,3, CV_64F);

	double sinx = sin(theta);
	double cosx = cos(theta);

	double* col = rot_matrix.ptr<double>(0);
	col[0] = cosx;
	col[1] = sinx;
	col[2] = 0;

	col = rot_matrix.ptr<double>(1);
	col[0] = -sinx;
	col[1] = cosx;
	col[2] = 0;

	col = rot_matrix.ptr<double>(2);
	col[0] = 0;
	col[1] = 0;
	col[2] = 1;

	return rot_matrix;
}

void makeUnitVector(std::vector<double> vec, std::vector<double>& unit_vector) {
	
	double magnitude = 0;

	for(int i=0;i<vec.size();i++) {
		magnitude += vec[i]*vec[i];
	}
	magnitude = sqrt(magnitude);

	for(int i=0;i<vec.size();i++) {
		unit_vector[i] = (((double)(vec[i])/magnitude));
	}
}

void get_rotated_vector(std::vector<double> vec, std::vector<double>& vec_rot) {

	double temp = vec[0];
	temp = temp/sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

	double theta = acos(temp);
	std::cout<<" theta-x : "<<theta<<" ";

	double sinx = sin(theta);
	double cosx = cos(theta);
/*
	//Rotation about the X-axis
	vec_rot[0] = vec[0];
	vec_rot[1] = vec[1]*cosx - vec[2]*sinx;
	vec_rot[2] = vec[1]*sinx + vec[2]*cosx;*/

	vec_rot = vec;
}

void get_reverse_vector(std::vector<double> vec, std::vector<double>& vec_rot) {
	double temp = vec[0];
	temp = temp/sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

	double theta = acos(temp);
	std::cout<<" theta-z : "<<theta<<" ";

	double sinx = sin(theta);
	double cosx = cos(theta);

	//Reverse - rotation about the X-axis
	vec_rot[0] = vec[0];
	vec_rot[1] = vec[1]*cosx + vec[2]*sinx;
	vec_rot[2] = -vec[1]*sinx + vec[2]*cosx;
}

void compute_vector_sum(std::vector<double> vec1, std::vector<double> vec2, std::vector<double>& vec_sum) {
	vec_sum[0] = (vec1[0] + vec2[0]);
	vec_sum[1] = (vec1[1] + vec2[1]);
	vec_sum[2] = (vec1[2] + vec2[2]);
}

void draw_eye_gaze(cv::Point pt, std::vector<double> vec_gaze, cv::Rect roi_eye, cv::Mat& img) {

	double del_x = 70*vec_gaze[0];
	double del_y = 70*vec_gaze[1];

	cv::line(img, cv::Point(pt.x + roi_eye.x, pt.y + roi_eye.y), cv::Point(pt.x + del_x + roi_eye.x, pt.y + del_y + roi_eye.y), cv::Scalar(255, 255, 255), 2);
}

cv::KalmanFilter KF_p_l (6,6,0);
cv::Mat_<float> measurement_p_l (6,1);

void init_kalman_point_p_l(cv::Point pt_pos_l) {
	KF_p_l.statePre.at<float>(0) = pt_pos_l.x;
	KF_p_l.statePre.at<float>(1) = pt_pos_l.y;
	KF_p_l.statePre.at<float>(2) = 0;
	KF_p_l.statePre.at<float>(3) = 0;
	KF_p_l.statePre.at<float>(4) = 0;
	KF_p_l.statePre.at<float>(5) = 0;

	/*KF_p_l.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_p_l.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);*/
	KF_p_l.transitionMatrix = *(cv::Mat_<float>(6,6) << 1,0,1,0,0.5,0,    0,1,0,1,0,0.5,     0,0,1,0,1,0,   0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
	KF_p_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_p_l.measurementMatrix);
	cv::setIdentity(KF_p_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_p_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_p_l.errorCovPost, cv::Scalar::all(.1)); 
}

cv::KalmanFilter KF_p_r (6,6,0);
cv::Mat_<float> measurement_p_r (6,1);

void init_kalman_point_p_r(cv::Point pt_pos_r) {
	KF_p_r.statePre.at<float>(0) = pt_pos_r.x;
	KF_p_r.statePre.at<float>(1) = pt_pos_r.y;
	KF_p_r.statePre.at<float>(2) = 0;
	KF_p_r.statePre.at<float>(3) = 0;
	KF_p_r.statePre.at<float>(4) = 0;
	KF_p_r.statePre.at<float>(5) = 0;

	/*KF_p_r.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_p_r.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);*/
	KF_p_r.transitionMatrix = *(cv::Mat_<float>(6,6) << 1,0,1,0,0.5,0,    0,1,0,1,0,0.5,     0,0,1,0,1,0,   0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
	KF_p_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_p_r.measurementMatrix);
	cv::setIdentity(KF_p_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_p_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_p_r.errorCovPost, cv::Scalar::all(.1)); 
}


cv::Point2f kalman_correct_point_p_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old, cv::Point pt_vel_old) {
	cv::Mat prediction = KF_p_l.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_p_l(0) = pt_pos_l.x;
	measurement_p_l(1) = pt_pos_l.y;
	measurement_p_l(2) = pt_pos_l.x - pt_pos_l_old.x;
	measurement_p_l(3) = pt_pos_l.y - pt_pos_l_old.y;
	measurement_p_l(4) = measurement_p_l(2) - pt_vel_old.x;
	measurement_p_l(5) = measurement_p_l(3) - pt_vel_old.y;

	cv::Mat estimated = KF_p_l.correct(measurement_p_l);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}


cv::Point2f kalman_correct_point_p_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old, cv::Point pt_vel_old) {
	cv::Mat prediction = KF_p_r.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_p_r(0) = pt_pos_r.x;
	measurement_p_r(1) = pt_pos_r.y;
	measurement_p_r(2) = pt_pos_r.x - pt_pos_r_old.x;
	measurement_p_r(3) = pt_pos_r.y - pt_pos_r_old.y;
	measurement_p_r(4) = measurement_p_r(2) - pt_vel_old.x;
	measurement_p_r(5) = measurement_p_r(3) - pt_vel_old.y;

	cv::Mat estimated = KF_p_r.correct(measurement_p_r);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::KalmanFilter KF_e_l (4,4,0);
cv::Mat_<float> measurement_e_l (4,1);

void init_kalman_point_e_l(cv::Point pt_pos_l) {
	KF_e_l.statePre.at<float>(0) = pt_pos_l.x;
	KF_e_l.statePre.at<float>(1) = pt_pos_l.y;
	KF_e_l.statePre.at<float>(2) = 0;
	KF_e_l.statePre.at<float>(3) = 0;

	KF_e_l.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_e_l.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);
	cv::setIdentity(KF_e_l.measurementMatrix);
	cv::setIdentity(KF_e_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_e_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_e_l.errorCovPost, cv::Scalar::all(.1)); 
}

cv::KalmanFilter KF_e_r (4,4,0);
cv::Mat_<float> measurement_e_r (4,1);

void init_kalman_point_e_r(cv::Point pt_pos_r) {
	KF_e_r.statePre.at<float>(0) = pt_pos_r.x;
	KF_e_r.statePre.at<float>(1) = pt_pos_r.y;
	KF_e_r.statePre.at<float>(2) = 0;
	KF_e_r.statePre.at<float>(3) = 0;

	KF_e_r.transitionMatrix = *(cv::Mat_<float>(4,4) << 1,0,1,0,    0,1,0,1,0,     0,0,1,0,   0,0,0,1);
	KF_e_r.processNoiseCov = *(cv::Mat_<float>(4,4) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3);
	cv::setIdentity(KF_e_r.measurementMatrix);
	cv::setIdentity(KF_e_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_e_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_e_r.errorCovPost, cv::Scalar::all(.1)); 
}

cv::Point2f kalman_correct_point_e_l(cv::Point pt_pos_l, cv::Point pt_pos_l_old) {
	cv::Mat prediction = KF_e_l.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_e_l(0) = pt_pos_l.x;
	measurement_e_l(1) = pt_pos_l.y;
	measurement_e_l(2) = pt_pos_l.x - pt_pos_l_old.x;
	measurement_e_l(3) = pt_pos_l.y - pt_pos_l_old.y;

	cv::Mat estimated = KF_e_l.correct(measurement_e_l);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

cv::Point2f kalman_correct_point_e_r(cv::Point pt_pos_r, cv::Point pt_pos_r_old) {
	cv::Mat prediction = KF_e_r.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement_e_r(0) = pt_pos_r.x;
	measurement_e_r(1) = pt_pos_r.y;
	measurement_e_r(2) = pt_pos_r.x - pt_pos_r_old.x;
	measurement_e_r(3) = pt_pos_r.y - pt_pos_r_old.y;

	cv::Mat estimated = KF_e_r.correct(measurement_e_r);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}


cv::KalmanFilter KF_ce_l(6, 6, 0);
cv::Mat_<float> measurement_ce_l(6,1);

void init_kalman_ce_l(std::vector<double> vec) {

	KF_ce_l.statePre.at<float>(0) = vec[0];
	KF_ce_l.statePre.at<float>(1) = vec[1];
	KF_ce_l.statePre.at<float>(2) = vec[2];
	KF_ce_l.statePre.at<float>(3) = 0;
	KF_ce_l.statePre.at<float>(4) = 0;
	KF_ce_l.statePre.at<float>(5) = 0;


	KF_ce_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ce_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ce_l.measurementMatrix);
	cv::setIdentity(KF_ce_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ce_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ce_l.errorCovPost, cv::Scalar::all(.1));  
}


cv::KalmanFilter KF_ce_r(6, 6, 0);
cv::Mat_<float> measurement_ce_r(6,1);

void init_kalman_ce_r(std::vector<double> vec) {

	KF_ce_r.statePre.at<float>(0) = vec[0];
	KF_ce_r.statePre.at<float>(1) = vec[1];
	KF_ce_r.statePre.at<float>(2) = vec[2];
	KF_ce_r.statePre.at<float>(3) = 0;
	KF_ce_r.statePre.at<float>(4) = 0;
	KF_ce_r.statePre.at<float>(5) = 0;


	KF_ce_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ce_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ce_r.measurementMatrix);
	cv::setIdentity(KF_ce_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ce_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ce_r.errorCovPost, cv::Scalar::all(.1));  
}


void kalman_predict_correct_ce_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ce_l.predict();
	measurement_ce_l(0) = vec[0];
	measurement_ce_l(1) = vec[1];
	measurement_ce_l(2) = vec[2];
	measurement_ce_l(3) = vec[0] - old[0];
	measurement_ce_l(4) = vec[1] - old[1];
	measurement_ce_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ce_l.correct(measurement_ce_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

void kalman_predict_correct_ce_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ce_r.predict();
	measurement_ce_r(0) = vec[0];
	measurement_ce_r(1) = vec[1];
	measurement_ce_r(2) = vec[2];
	measurement_ce_r(3) = vec[0] - old[0];
	measurement_ce_r(4) = vec[1] - old[1];
	measurement_ce_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ce_r.correct(measurement_ce_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}


cv::KalmanFilter KF_ep_l(6, 6, 0);
cv::Mat_<float> measurement_ep_l(6,1);

void init_kalman_ep_l(std::vector<double> vec) {

	KF_ep_l.statePre.at<float>(0) = vec[0];
	KF_ep_l.statePre.at<float>(1) = vec[1];
	KF_ep_l.statePre.at<float>(2) = vec[2];
	KF_ep_l.statePre.at<float>(3) = 0;
	KF_ep_l.statePre.at<float>(4) = 0;
	KF_ep_l.statePre.at<float>(5) = 0;


	KF_ep_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ep_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ep_l.measurementMatrix);
	cv::setIdentity(KF_ep_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ep_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ep_l.errorCovPost, cv::Scalar::all(.1));  
}


cv::KalmanFilter KF_ep_r(6, 6, 0);
cv::Mat_<float> measurement_ep_r(6,1);

void init_kalman_ep_r(std::vector<double> vec) {

	KF_ep_r.statePre.at<float>(0) = vec[0];
	KF_ep_r.statePre.at<float>(1) = vec[1];
	KF_ep_r.statePre.at<float>(2) = vec[2];
	KF_ep_r.statePre.at<float>(3) = 0;
	KF_ep_r.statePre.at<float>(4) = 0;
	KF_ep_r.statePre.at<float>(5) = 0;


	KF_ep_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_ep_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_ep_r.measurementMatrix);
	cv::setIdentity(KF_ep_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_ep_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_ep_r.errorCovPost, cv::Scalar::all(.1));  
}



void kalman_predict_correct_ep_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ep_l.predict();
	measurement_ep_l(0) = vec[0];
	measurement_ep_l(1) = vec[1];
	measurement_ep_l(2) = vec[2];
	measurement_ep_l(3) = vec[0] - old[0];
	measurement_ep_l(4) = vec[1] - old[1];
	measurement_ep_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ep_l.correct(measurement_ep_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}


void kalman_predict_correct_ep_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_ep_r.predict();
	measurement_ep_r(0) = vec[0];
	measurement_ep_r(1) = vec[1];
	measurement_ep_r(2) = vec[2];
	measurement_ep_r(3) = vec[0] - old[0];
	measurement_ep_r(4) = vec[1] - old[1];
	measurement_ep_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_ep_r.correct(measurement_ep_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}


cv::KalmanFilter KF_cp_l(6, 6, 0);
cv::Mat_<float> measurement_cp_l(6,1);

void init_kalman_cp_l(std::vector<double> vec) {

	KF_cp_l.statePre.at<float>(0) = vec[0];
	KF_cp_l.statePre.at<float>(1) = vec[1];
	KF_cp_l.statePre.at<float>(2) = vec[2];
	KF_cp_l.statePre.at<float>(3) = 0;
	KF_cp_l.statePre.at<float>(4) = 0;
	KF_cp_l.statePre.at<float>(5) = 0;


	KF_cp_l.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_cp_l.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_cp_l.measurementMatrix);
	cv::setIdentity(KF_cp_l.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_cp_l.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_cp_l.errorCovPost, cv::Scalar::all(.1));  
}


cv::KalmanFilter KF_cp_r(6, 6, 0);
cv::Mat_<float> measurement_cp_r(6,1);

void init_kalman_cp_r(std::vector<double> vec) {

	KF_cp_r.statePre.at<float>(0) = vec[0];
	KF_cp_r.statePre.at<float>(1) = vec[1];
	KF_cp_r.statePre.at<float>(2) = vec[2];
	KF_cp_r.statePre.at<float>(3) = 0;
	KF_cp_r.statePre.at<float>(4) = 0;
	KF_cp_r.statePre.at<float>(5) = 0;


	KF_cp_r.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF_cp_r.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF_cp_r.measurementMatrix);
	cv::setIdentity(KF_cp_r.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF_cp_r.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF_cp_r.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_cp_l(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_cp_l.predict();
	measurement_cp_l(0) = vec[0];
	measurement_cp_l(1) = vec[1];
	measurement_cp_l(2) = vec[2];
	measurement_cp_l(3) = vec[0] - old[0];
	measurement_cp_l(4) = vec[1] - old[1];
	measurement_cp_l(5) = vec[2] - old[2];

	cv::Mat estimated = KF_cp_l.correct(measurement_cp_l);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}


void kalman_predict_correct_cp_r(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF_cp_r.predict();
	measurement_cp_r(0) = vec[0];
	measurement_cp_r(1) = vec[1];
	measurement_cp_r(2) = vec[2];
	measurement_cp_r(3) = vec[0] - old[0];
	measurement_cp_r(4) = vec[1] - old[1];
	measurement_cp_r(5) = vec[2] - old[2];

	cv::Mat estimated = KF_cp_r.correct(measurement_cp_r);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

void retrace_eye_center(cv::Point& pt_e_pos, double normal[3], double mag) {
	pt_e_pos.x = pt_e_pos.x - normal[0]*mag;
	pt_e_pos.y = pt_e_pos.y - normal[1]*mag;
}

void blow_up_rect(cv::Rect& rect, double f) {
	rect.x = rect.x - (rect.width*(f - 1))/2.0;
	rect.y = rect.y - (rect.height*(f - 1))/2.0;
	rect.width = ((double)rect.width)*f;
	rect.height = ((double)rect.height)*f;
}

/*double get_z_component(double Cf_left, cv::Point pt_p_kalman, cv::Point pt_e_kalman, std::vector<double> vec_ce_kalman) {	
	makeUnitVector(vec_ce_kalman, vec_ce_kalman);
	double x, y, z, mag = Cf_left*13.101, z_comp;
	x = pt_e_kalman.x - vec_ce_kalman[0]*mag;
	y = pt_e_kalman.y - vec_ce_kalman[1]*mag;
	z = -vec_ce_kalman[2]*mag;

	z_comp = sqrt(pow(8*Cf_left, 2) - pow(pt_p_kalman.x - x, 2) + pow(pt_p_kalman.y - y, 2)) - z;
	return z_comp;
}*/

int main(int argc, char** argv) {
	try	{
		Rm = std::atoi(argv[1])/100.0;
		Rn = std::atoi(argv[2])/100.0;

		//Wf = std::atoi(argv[3])/100.0;

		//Nf = std::atoi(argv[3])/100.0;
		std::cout<<"Rm : "<<Rm<<" Rn : "<<Rn<<endl;

		cv::VideoCapture cap(0);
		image_window win;

		FaceFeatures *face_features = new FaceFeatures();
		FaceData *face_data = new FaceData();
		FacePose *face_pose = new FacePose();

		frontal_face_detector detector = get_frontal_face_detector();
		shape_predictor pose_model;
		deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

		std::vector<double> vec_ce_pos_l(3), vec_ce_vel_l(3), vec_ce_pos_l_old(3), vec_ce_vel_l_old(3), vec_ce_kalman_l(3);
		std::vector<double> vec_ep_pos_l(3), vec_ep_vel_l(3), vec_ep_pos_l_old(3), vec_ep_vel_l_old(3), vec_ep_kalman_l(3);
		std::vector<double> vec_cp_pos_l(3), vec_cp_vel_l(3), vec_cp_pos_l_old(3), vec_cp_vel_l_old(3), vec_cp_kalman_l(3);

		std::vector<double> vec_ce_pos_r(3), vec_ce_vel_r(3), vec_ce_pos_r_old(3), vec_ce_vel_r_old(3), vec_ce_kalman_r(3);
		std::vector<double> vec_ep_pos_r(3), vec_ep_vel_r(3), vec_ep_pos_r_old(3), vec_ep_vel_r_old(3), vec_ep_kalman_r(3);
		std::vector<double> vec_cp_pos_r(3), vec_cp_vel_r(3), vec_cp_pos_r_old(3), vec_cp_vel_r_old(3), vec_cp_kalman_r(3);

		std::vector<double> center_eye_proj(3);
		std::vector<double> vec_cp_kalman_avg(3);

		//TODO : Initialize all vectors to [0, 0, 0];

		vec_ce_pos_l[0] = 0;vec_ce_pos_l[1] = 0;vec_ce_pos_l[2] = 0;
		vec_ce_pos_l_old[0] = 0;vec_ce_pos_l_old[1] = 0;vec_ce_pos_l_old[2] = 0;

		vec_ce_pos_r[0] = 0;vec_ce_pos_r[1] = 0;vec_ce_pos_r[2] = 0;
		vec_ce_pos_r_old[0] = 0;vec_ce_pos_r_old[1] = 0;vec_ce_pos_r_old[2] = 0;


		vec_ep_pos_l[0] = 0;vec_ep_pos_l[1] = 0;vec_ep_pos_l[2] = 0;
		vec_ep_pos_l_old[0] = 0;vec_ep_pos_l_old[1] = 0;vec_ep_pos_l_old[2] = 0;

		vec_ep_pos_r[0] = 0;vec_ep_pos_r[1] = 0;vec_ep_pos_r[2] = 0;
		vec_ep_pos_r_old[0] = 0;vec_ep_pos_r_old[1] = 0;vec_ep_pos_r_old[2] = 0;


		vec_cp_pos_l[0] = 0;vec_cp_pos_l[1] = 0;vec_cp_pos_l[2] = 0;
		vec_cp_pos_l_old[0] = 0;vec_cp_pos_l_old[1] = 0;vec_cp_pos_l_old[2] = 0;

		vec_cp_pos_r[0] = 0;vec_cp_pos_r[1] = 0;vec_cp_pos_r[2] = 0;
		vec_cp_pos_r_old[0] = 0;vec_cp_pos_r_old[1] = 0;vec_cp_pos_r_old[2] = 0;


		cv::Point pt_p_pos_l(0,0), pt_p_vel_l(0,0), pt_p_pos_l_old(0,0), pt_p_kalman_l(0,0), pt_p_vel_l_old(0,0);
		cv::Point pt_e_pos_l(0,0), pt_e_vel_l(0,0), pt_e_pos_l_old(0,0), pt_e_kalman_l(0,0);

		cv::Point pt_p_pos_r(0,0), pt_p_vel_r(0,0), pt_p_pos_r_old(0,0), pt_p_kalman_r(0,0), pt_p_vel_r_old(0,0);
		cv::Point pt_e_pos_r(0,0), pt_e_vel_r(0,0), pt_e_pos_r_old(0,0), pt_e_kalman_r(0,0);

		cv::Rect rect1, rect2;

		cv::Mat frame, temp, temp2, roi1,roi2;
		int k_pt_e_l = 0, k_pt_p_l = 0, k_vec_ce_l = 0, k_vec_cp_l = 0, k_vec_ep_l = 0;
		int k_pt_e_r = 0, k_pt_p_r = 0, k_vec_ce_r = 0, k_vec_cp_r = 0, k_vec_ep_r = 0;

		while(!win.is_closed()) {
			cap>>frame;
			cv::flip(frame, frame, 1);
			cv::cvtColor(frame, frame, CV_BGR2GRAY);

			cv_image<unsigned char> cimg(frame);

			std::vector<rectangle> faces = detector(cimg);

			std::vector<full_object_detection> shapes;
			for (unsigned long i = 0; i < faces.size(); ++i)
				shapes.push_back(pose_model(cimg, faces[i]));

			if(shapes.size() == 0) {
				std::cout<<"zero faces"<<std::endl;
				k_pt_p_l=0;
				k_pt_e_l=0;
				k_vec_ce_l=0;
				k_vec_ep_l=0;
				k_pt_p_r=0;
				k_pt_e_r=0;
				k_vec_ce_r=0;
				k_vec_ep_r=0;

			}
			else {

				//TODO : Initialize the variables used in the Kalman filter

				pt_p_pos_l_old = pt_p_pos_l;
				pt_p_vel_l_old = pt_p_vel_l;
				pt_e_pos_l_old = pt_e_pos_l;

				pt_p_pos_r_old = pt_p_pos_r;
				pt_p_vel_r_old = pt_p_vel_r;
				pt_e_pos_r_old = pt_e_pos_r;


				vec_ce_pos_l_old = vec_ce_pos_l;
				vec_ep_pos_l_old = vec_ep_pos_l;
				vec_cp_pos_l_old = vec_cp_pos_l;

				vec_ce_pos_r_old = vec_ce_pos_r;
				vec_ep_pos_r_old = vec_ep_pos_r;
				vec_cp_pos_r_old = vec_cp_pos_r;

				dlib::full_object_detection shape = shapes[0];

				face_features->assign(cv::Point(0,0),
					get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()),
						cv::Point(shape.part(45).x(), shape.part(45).y())),
					get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()),
						cv::Point(shape.part(39).x(), shape.part(39).y())),
					cv::Point(shape.part(30).x(), shape.part(30).y()), 
					get_mid_point(cv::Point(shape.part(48).x(), shape.part(48).y()),
						cv::Point(shape.part(54).x(), shape.part(54).y())));

				face_data->assign(face_features);

				face_pose->assign(face_features, face_data);
				Cf_left = get_distance(cv::Point(shape.part(42).x(), shape.part(42).y()),
					cv::Point(shape.part(45).x(), shape.part(45).y()));
				Cf_right = get_distance(cv::Point(shape.part(36).x(), shape.part(36).y()),
					cv::Point(shape.part(39).x(), shape.part(39).y()));


				Cf_left = (Cf_left)/(14.0);
				Cf_right = (Cf_right)/(14.0);

				std::vector<cv::Point> vec_pts_left_eye(0), vec_pts_right_eye(0);
				
				for(int j=42;j<=47;j++) {
					vec_pts_left_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
				}

				for(int j=36;j<=41;j++) {
					vec_pts_right_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
				}

				rect1 = cv::boundingRect(vec_pts_left_eye);
				rect2 = cv::boundingRect(vec_pts_right_eye);

				blow_up_rect(rect1, 2.0);
				blow_up_rect(rect2, 2.0);


				//cv::Rect rect2(cv::Point(shape.part(22).x(), shape.part(22).y()), cv::Point(shape.part(26).x(), rect1.y + rect1.height));
				cv::rectangle(frame, rect1, cv::Scalar(255, 255, 255), 1, 8, 0);
				cv::rectangle(frame, rect2, cv::Scalar(255, 255, 255), 1, 8, 0);
				
				roi1 = frame(rect1);
				roi2 = frame(rect2);
				//TODO : Compute current values and correct values using Kalman filter

				pt_e_pos_l = get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()),cv::Point(shape.part(45).x(), shape.part(45).y()));
				pt_e_pos_r = get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()),cv::Point(shape.part(39).x(), shape.part(39).y()));
				

				//cv::Point(cv::Point((shape.part(23).x() + rect1.x + rect1.width)*0.5, shape.part(23).y()*(1.0-Wf) + Wf*(rect1.y + rect1.height)));
				cv::circle(frame, pt_e_pos_l, 1, cv::Scalar(255,0,0), 1, 4, 0);
				cv::circle(frame, pt_e_pos_r, 1, cv::Scalar(255,0,0), 1, 4, 0);

				//retrace_eye_center(pt_e_pos, face_pose->normal, Cf_left);
				//cv::circle(frame, pt_e_pos, 1, cv::Scalar(127,0,0), 1, 4, 0);

				pt_e_pos_l.x -= rect1.x;
				pt_e_pos_l.y -= rect1.y;
				pt_e_vel_l.x = pt_e_pos_l.x - pt_e_pos_l_old.x;
				pt_e_vel_l.y = pt_e_pos_l.y - pt_e_pos_l_old.y;

				pt_e_pos_r.x -= rect2.x;
				pt_e_pos_r.y -= rect2.y;
				pt_e_vel_r.x = pt_e_pos_r.x - pt_e_pos_r_old.x;
				pt_e_vel_r.y = pt_e_pos_r.y - pt_e_pos_r_old.y;

				if(k_pt_e_l == 0) {
					pt_e_pos_l_old.x = 0;
					pt_e_pos_l_old.y = 0;
					init_kalman_point_e_l(pt_e_pos_l);
					++k_pt_e_l;
				}

				if(k_pt_e_r == 0) {
					pt_e_pos_r_old.x = 0;
					pt_e_pos_r_old.y = 0;
					init_kalman_point_e_r(pt_e_pos_r);
					++k_pt_e_r;
				}

				pt_e_kalman_l = kalman_correct_point_e_l(pt_e_pos_l, pt_e_pos_l_old);
				pt_e_kalman_r = kalman_correct_point_e_r(pt_e_pos_r, pt_e_pos_r_old);

				std::cout<<"Point E - l "<<pt_e_kalman_l.x<<" "<<pt_e_kalman_l.y<<endl;
				std::cout<<"Point E - l "<<pt_e_kalman_r.x<<" "<<pt_e_kalman_r.y<<endl;

				pt_p_pos_l = findEyeCenter(roi1, rect1, "");
				pt_p_vel_l.x = pt_p_pos_l.x - pt_p_pos_l_old.x;
				pt_p_vel_l.y = pt_p_pos_l.y - pt_p_pos_l_old.y;

				pt_p_pos_r = findEyeCenter(roi2, rect2, "");
				pt_p_vel_r.x = pt_p_pos_r.x - pt_p_pos_r_old.x;
				pt_p_vel_r.y = pt_p_pos_r.y - pt_p_pos_r_old.y;

				if(k_pt_p_l == 0) {
					pt_p_pos_l_old.x = 0;
					pt_p_pos_l_old.y = 0;
					init_kalman_point_p_l(pt_p_pos_l);
					++k_pt_p_l;
				}

				if(k_pt_p_r == 0) {
					pt_p_pos_r_old.x = 0;
					pt_p_pos_r_old.y = 0;
					init_kalman_point_p_r(pt_p_pos_r);
					++k_pt_p_r;
				}

				pt_p_kalman_l = kalman_correct_point_p_l(pt_p_pos_l, pt_p_pos_l_old, pt_p_vel_l);
				pt_p_kalman_r = kalman_correct_point_p_r(pt_p_pos_r, pt_p_pos_r_old, pt_p_vel_r);


				if(!floodShouldPushPoint(pt_p_kalman_l, roi1)) {
					k_pt_p_l=0;
					k_pt_e_l=0;
					k_vec_ce_l=0;
					k_vec_ep_l=0;
				}

				if(!floodShouldPushPoint(pt_p_kalman_r, roi1)) {
					k_pt_p_r=0;
					k_pt_e_r=0;
					k_vec_ce_r=0;
					k_vec_ep_r=0;
				}

				std::cout<<"Point P "<<pt_p_kalman_l.x<<" "<<pt_p_kalman_l.y<<endl;
				std::cout<<"Point P "<<pt_p_kalman_r.x<<" "<<pt_p_kalman_r.y<<endl;
			
				vec_ce_pos_l[0] = face_pose->normal[0];
				vec_ce_pos_l[1] = face_pose->normal[1];
				vec_ce_pos_l[2] = face_pose->normal[2];
				
				vec_ce_pos_r[0] = face_pose->normal[0];
				vec_ce_pos_r[1] = face_pose->normal[1];
				vec_ce_pos_r[2] = face_pose->normal[2];


				vec_ce_vel_l[0] = vec_ce_pos_l[0] - vec_ce_pos_l_old[0];
				vec_ce_vel_l[1] = vec_ce_pos_l[1] - vec_ce_pos_l_old[1];
				vec_ce_vel_l[2] = vec_ce_pos_l[2] - vec_ce_pos_l_old[2];

				vec_ce_vel_r[0] = vec_ce_pos_r[0] - vec_ce_pos_r_old[0];
				vec_ce_vel_r[1] = vec_ce_pos_r[1] - vec_ce_pos_r_old[1];
				vec_ce_vel_r[2] = vec_ce_pos_r[2] - vec_ce_pos_r_old[2];

				//std::cout<<"vel "<<vec_ce_vel_r[0] <<" "<<vec_ce_vel_r[1] <<" "<<vec_ce_vel_r[2]<<" "<<sqrt(vec_ce_vel_r[0]*vec_ce_vel_r[0] +vec_ce_vel_r[1]*vec_ce_vel_r[1]+ vec_ce_vel_r[2]*vec_ce_vel_r[2]) <<endl;
				if(k_vec_ce_l == 0) {
					vec_ce_pos_l_old[0] = 0;vec_ce_pos_l_old[1] = 0;vec_ce_pos_l_old[2] = 0;
					init_kalman_ce_l(vec_ce_pos_l);
					++k_vec_ce_l;
				}

				if(k_vec_ce_r == 0) {
					vec_ce_pos_r_old[0] = 0;vec_ce_pos_r_old[1] = 0;vec_ce_pos_r_old[2] = 0;
					init_kalman_ce_r(vec_ce_pos_r);
					++k_vec_ce_r;
				}

				kalman_predict_correct_ce_l(vec_ce_pos_l, vec_ce_pos_l_old, vec_ce_kalman_l);
				kalman_predict_correct_ce_r(vec_ce_pos_r, vec_ce_pos_r_old, vec_ce_kalman_r);
		
				makeUnitVector(vec_ce_pos_l, vec_ce_pos_l);
				makeUnitVector(vec_ce_kalman_l, vec_ce_kalman_l);
				std::cout<<"Vector CE "<<vec_ce_kalman_l[0]<<" "<<vec_ce_kalman_l[1]<<" "<<vec_ce_kalman_l[2]<<endl;

				makeUnitVector(vec_ce_pos_r, vec_ce_pos_r);
				makeUnitVector(vec_ce_kalman_r, vec_ce_kalman_r);
				std::cout<<"Vector CE "<<vec_ce_kalman_r[0]<<" "<<vec_ce_kalman_r[1]<<" "<<vec_ce_kalman_r[2]<<endl;


				vec_ep_pos_l[0] = pt_p_kalman_l.x - pt_e_kalman_l.x;
				vec_ep_pos_l[1] = pt_p_kalman_l.y - pt_e_kalman_l.y;
				vec_ep_pos_l[2] = 0.0;

				vec_ep_pos_r[0] = pt_p_kalman_r.x - pt_e_kalman_r.x;
				vec_ep_pos_r[1] = pt_p_kalman_r.y - pt_e_kalman_r.y;
				vec_ep_pos_r[2] = 0.0;

				vec_ep_pos_l[0] = pt_p_pos_l.x - pt_e_pos_l.x;
				vec_ep_pos_l[1] = pt_p_pos_l.y - pt_e_pos_l.y;
				vec_ep_pos_l[2] = 0.0;

				vec_ep_pos_r[0] = pt_p_pos_r.x - pt_e_pos_r.x;
				vec_ep_pos_r[1] = pt_p_pos_r.y - pt_e_pos_r.y;
				vec_ep_pos_r[2] = 0.0;

				if(k_vec_ep_l == 0) {
					vec_ep_pos_l_old[0] = 0;
					vec_ep_pos_l_old[1] = 0;
					vec_ep_pos_l_old[2] = 0;
					init_kalman_ep_l(vec_ep_pos_l);
					++k_vec_ep_l;
				}
	
				if(k_vec_ep_r == 0) {
					vec_ep_pos_r_old[0] = 0;
					vec_ep_pos_r_old[1] = 0;
					vec_ep_pos_r_old[2] = 0;
					init_kalman_ep_r(vec_ep_pos_r);
					++k_vec_ep_r;
				}
	
				kalman_predict_correct_ep_l(vec_ep_pos_l, vec_ep_pos_l_old, vec_ep_kalman_l);
				kalman_predict_correct_ep_r(vec_ep_pos_r, vec_ep_pos_r_old, vec_ep_kalman_r);

				vec_cp_pos_l[0] = (13.101*Cf_left*vec_ce_pos_l[0]) + vec_ep_pos_l[0];
				vec_cp_pos_l[1] = (13.101*Cf_left*vec_ce_pos_l[1]) + vec_ep_pos_l[1];
				vec_cp_pos_l[2] = (13.101*Cf_left*vec_ce_pos_l[2]) + vec_ep_pos_l[2];
	
				vec_cp_pos_r[0] = (13.101*Cf_right*vec_ce_pos_r[0]) + 3.0*vec_ep_pos_r[0];
				vec_cp_pos_r[1] = (13.101*Cf_right*vec_ce_pos_r[1]) + 3.0*vec_ep_pos_r[1];
				vec_cp_pos_r[2] = (13.101*Cf_right*vec_ce_pos_r[2]) + 3.0*vec_ep_pos_r[2];
	

				vec_cp_vel_l[0] = vec_cp_pos_l[0] - vec_cp_pos_l_old[0];
				vec_cp_vel_l[1] = vec_cp_pos_l[1] - vec_cp_pos_l_old[1];
				vec_cp_vel_l[2] = vec_cp_pos_l[2] - vec_cp_pos_l_old[2];
		
				vec_cp_vel_r[0] = vec_cp_pos_r[0] - vec_cp_pos_r_old[0];
				vec_cp_vel_r[1] = vec_cp_pos_r[1] - vec_cp_pos_r_old[1];
				vec_cp_vel_r[2] = vec_cp_pos_r[2] - vec_cp_pos_r_old[2];

	
				if(k_vec_cp_l == 0) {
					vec_cp_pos_l_old[0] = 0;
					vec_cp_pos_l_old[1] = 0;
					vec_cp_pos_l_old[2] = 0;
					init_kalman_cp_l(vec_cp_pos_l);
					++k_vec_cp_l;
				}
		
				if(k_vec_cp_r == 0) {
					vec_cp_pos_r_old[0] = 0;
					vec_cp_pos_r_old[1] = 0;
					vec_cp_pos_r_old[2] = 0;
					init_kalman_cp_r(vec_cp_pos_r);
					++k_vec_cp_r;
				}

				kalman_predict_correct_cp_l(vec_cp_pos_l, vec_cp_pos_l_old, vec_cp_kalman_l);
				kalman_predict_correct_cp_r(vec_cp_pos_r, vec_cp_pos_r_old, vec_cp_kalman_r);
	
				makeUnitVector(vec_cp_kalman_l, vec_cp_kalman_l);
				makeUnitVector(vec_cp_kalman_r, vec_cp_kalman_r);

				std::cout<<"Vector CP "<<vec_cp_kalman_l[0]<<" "<<vec_cp_kalman_l[1]<<" "<<vec_cp_kalman_l[2]<<endl;
				std::cout<<"Vector CP "<<vec_cp_kalman_r[0]<<" "<<vec_cp_kalman_r[1]<<" "<<vec_cp_kalman_r[2]<<endl;

/*
				vec_cp_kalman[0] = vec_ce_kalman[0] + vec_ep_kalman[0];
				vec_cp_kalman[1] = vec_ce_kalman[1] + vec_ep_kalman[1];
				vec_cp_kalman[2] = vec_ce_kalman[2] + vec_ep_kalman[2];
*/
				makeUnitVector(vec_cp_kalman_l, vec_cp_kalman_l);
				makeUnitVector(vec_cp_kalman_r, vec_cp_kalman_r);


				/*if(!floodShouldPushPoint(pt_p_kalman, roi1)) {
					init_kalman_point_p(pt_p_pos);
				}*/
				vec_cp_kalman_avg[0] = (vec_cp_kalman_l[0] + vec_cp_kalman_r[0])/2.0;
				vec_cp_kalman_avg[1] = (vec_cp_kalman_l[1] + vec_cp_kalman_r[1])/2.0;
				vec_cp_kalman_avg[2] = (vec_cp_kalman_l[2] + vec_cp_kalman_r[2])/2.0;

				//cv::circle(roi1, pt_p_kalman, 1, cv::Scalar(255,255,0), 1, 4, 0);
/*
				draw_eye_gaze(pt_p_kalman_l, vec_cp_kalman_l, rect1, frame);				
				draw_eye_gaze(pt_p_kalman_r, vec_cp_kalman_r, rect2, frame);
*/				

				draw_eye_gaze(pt_p_kalman_l, vec_cp_kalman_avg, rect1, frame);				
				draw_eye_gaze(pt_p_kalman_r, vec_cp_kalman_avg, rect2, frame);

				draw_facial_normal(frame, shape, vec_ce_kalman_l);
			}
			win.clear_overlay();
			win.set_image(cimg);
			//win.add_overlay(render_face_detections(shapes));
		}
	}
	catch(serialization_error& e) {
		cout << "You need dlib's default face landmarking model file to run this example." << endl;
		cout << "You can get it from the following URL: " << endl;
		cout << "   http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2" << endl;
		cout << endl << e.what() << endl;
	}
	catch(exception& e) {
		cout << e.what() << endl;
	}
}