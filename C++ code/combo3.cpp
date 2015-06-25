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
double Cf_left = 10;

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

void draw_facial_normal(cv::Mat& img, dlib::full_object_detection shape, FacePose* f) {

	double del_x = 100*f->normal[0];
	double del_y = 100*f->normal[1];

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

	//Reducing the size so as to fit it properly
	double del_x = 70*vec_gaze[0];
	double del_y = 70*vec_gaze[1];

	cv::line(img, cv::Point(pt.x + roi_eye.x, pt.y + roi_eye.y), cv::Point(pt.x + del_x + roi_eye.x, pt.y + del_y + roi_eye.y), cv::Scalar(255, 255, 255), 2);
}

cv::KalmanFilter KF3(6, 6, 0);
cv::Mat_<float> state3(6, 1);
cv::Mat_<float> measurement3(6,1);

void init_kalman_3(std::vector<double> vec) {

	KF3.statePre.at<float>(0) = vec[0];
	KF3.statePre.at<float>(1) = vec[1];
	KF3.statePre.at<float>(2) = vec[2];
	KF3.statePre.at<float>(3) = 0;
	KF3.statePre.at<float>(4) = 0;
	KF3.statePre.at<float>(5) = 0;

	//TODO : change this matrix
	KF3.transitionMatrix = *(cv::Mat_<float>(6, 6) << 1,0,0,1,0,0, 0,1,0,0,1,0, 0,0,1,0,0,1, 0,0,0,1,0,0, 0,0,0,0,1,0, 0,0,0,0,0,1);
	KF3.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);

	cv::setIdentity(KF3.measurementMatrix);
	cv::setIdentity(KF3.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF3.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF3.errorCovPost, cv::Scalar::all(.1));  
}

void kalman_predict_correct_3(std::vector<double> vec, std::vector<double> old, std::vector<double>& vec_pred) {
	cv::Mat prediction = KF3.predict();
	measurement3(0) = vec[0];
	measurement3(1) = vec[1];
	measurement3(2) = vec[2];
	measurement3(3) = vec[0] - old[0];
	measurement3(4) = vec[1] - old[1];
	measurement3(5) = vec[2] - old[2];

	cv::Mat estimated = KF3.correct(measurement3);
	vec_pred[0] = estimated.at<float>(0);
	vec_pred[1] = estimated.at<float>(1);
	vec_pred[2] = estimated.at<float>(2);
}

cv::KalmanFilter KF (6,6,0);
cv::Mat_<float> state (6,1); 
cv::Mat_<float> measurement (6,1);

void init_kalman(double x, double y)
{

	KF.statePre.at<float>(0) = x;
	KF.statePre.at<float>(1) = y;
	KF.statePre.at<float>(2) = 0;
	KF.statePre.at<float>(3) = 0;
	KF.statePre.at<float>(4) = 0;
	KF.statePre.at<float>(5) = 0;

	KF.transitionMatrix = *(cv::Mat_<float>(6,6) << 1,0,1,0,0.5,0,    0,1,0,1,0,0.5,     0,0,1,0,1,0,   0,0,0,1,0,1,  0,0,0,0,1,0,  0,0,0,0,0,1);
	KF.processNoiseCov = *(cv::Mat_<float>(6,6) << 0.2,0,0.2,0,  0,0.2,0,0.2,  0,0,0.3,0,   
		0,0,0,0.3, 0.2,0,0.2,0,  0,0.2,0,0.2);
	cv::setIdentity(KF.measurementMatrix);
	cv::setIdentity(KF.processNoiseCov,cv::Scalar::all(1e-4));
	cv::setIdentity(KF.measurementNoiseCov,cv::Scalar::all(1e-1));
	cv::setIdentity(KF.errorCovPost, cv::Scalar::all(.1));  
}

cv::Point2f kalman_predict_correct(double x, double y, double vx, double vy, double ax, double ay)
{
	cv::Mat prediction = KF.predict();
	cv::Point2f predictPt (prediction.at<float>(0), prediction.at<float>(1));   
	measurement(0) = x;
	measurement(1) = y;
	measurement(2) = vx;
	measurement(3) = vy;
	measurement(4) = ax;
	measurement(5) = ay;

	cv::Mat estimated = KF.correct(measurement);
	cv::Point2f statePt (estimated.at<float>(0), estimated.at<float>(1));
	return statePt;
}

int main(int argc, char **argv) {
	try	{
		Rm = std::atoi(argv[1])/100.0;
		Rn = std::atoi(argv[2])/100.0;
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

		std::vector<double> vec_normal(3), vec_pupil_left_proj(3), vec_pupil_left(3), vec_pupil_right(3);

		std::vector<double> vec_temp1(3), vec_temp2(3);
		std::vector<double> sm_le_gaze(3);

		vec_pupil_left[0] = 0;
		vec_pupil_left[1] = 0;
		vec_pupil_left[2] = 0;

		cv::Point pt_temp1, pupil_left_eye(0,0), vt_temp1;

		/////// Kalman Filter initiated ////////
		measurement.setTo(cv::Scalar(0));
		int k_i=0;

		while(!win.is_closed())	{
			cv::Mat temp, temp2, roi_left_eye_temp, roi_right_eye_temp;
			cap >> temp;
			temp.copyTo(temp2);
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

				draw_facial_normal(temp, shape, face_pose);
				project_facial_pose(temp, face_pose->normal, face_pose->sigma, face_pose->theta);

				std::vector<cv::Point> vec_pts_left_eye(0), vec_pts_right_eye(0);

				for(int j=42;j<=47;j++) {
					vec_pts_left_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
				}

				for(int j=36;j<=41;j++) {
					vec_pts_right_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
				}

				cv::Point rect_center_left_eye = get_mid_point(cv::Point(shape.part(42).x(), shape.part(42).y()), cv::Point(shape.part(45).x(), shape.part(45).y()));
				/*double conv_x = get_distance(cv::Point(shape.part(42).x(), shape.part(42).y()), cv::Point(shape.part(45).x(), shape.part(45).y()));
				conv_x = conv_x/2.0;*/
                cv::Rect roi_left_eye_rect/*(rect_center_left_eye.x - conv_x, rect_center_left_eye.y - conv_x, 2*conv_x, 2*conv_x);*/= cv::boundingRect(vec_pts_left_eye);

				cv::Mat roi_left_eye = temp(roi_left_eye_rect);
				cv::cvtColor(roi_left_eye, roi_left_eye_temp, CV_BGR2GRAY);

				preprocessROI(roi_left_eye_temp);

				vt_temp1.x = pupil_left_eye.x - pt_temp1.x;
				vt_temp1.y = pupil_left_eye.y - pt_temp1.y;

				pt_temp1 = pupil_left_eye;

				pupil_left_eye = findEyeCenter(roi_left_eye_temp, roi_left_eye_rect,"");
                //cv::circle( roi_left_eye, pupil_left_eye, 2, cv::Scalar(0, 255, 0), -1, 8, 0 );

				cv::Point rect_center_right_eye = get_mid_point(cv::Point(shape.part(36).x(), shape.part(36).y()), cv::Point(shape.part(39).x(), shape.part(39).y()));
                cv::Rect roi_right_eye_rect/*(rect_center_right_eye.x - 15, rect_center_right_eye.y - 18, 35, 30);*/ = cv::boundingRect(vec_pts_right_eye);

				cv::Mat roi_right_eye = temp(roi_right_eye_rect);
				cv::cvtColor(roi_right_eye, roi_right_eye_temp, CV_BGR2GRAY);

				preprocessROI(roi_right_eye_temp);
				cv::Point pupil_right_eye = findEyeCenter(roi_right_eye_temp, roi_left_eye_rect, "");
                //std::cout<<pupil_right_eye.x<<" "<<pupil_right_eye.y<<endl;
				cv::circle( roi_right_eye, pupil_right_eye, 9, cv::Scalar(0, 255, 0), -1, 8, 0 );

				Cf_left = get_distance(cv::Point(shape.part(42).x(), shape.part(42).y()),
					cv::Point(shape.part(45).x(), shape.part(45).y()));

				Cf_left = (12.5*Cf_left)/14.0;
				std::cout<<"Cf_left : "<<Cf_left<<" ";

				vec_normal[0] = (face_pose->normal[0]*Cf_left);
				vec_normal[1] = (face_pose->normal[1]*Cf_left);
				vec_normal[2] = (face_pose->normal[2]*Cf_left);

                //std::cout<<vec_normal[0]<<endl;

				vec_pupil_left_proj[0] = (pupil_left_eye.x - (rect_center_left_eye.x - roi_left_eye_rect.x));
				vec_pupil_left_proj[1] = (pupil_left_eye.y - (rect_center_left_eye.y - roi_left_eye_rect.y));
				vec_pupil_left_proj[2] = (0);

				vec_temp1 = vec_pupil_left;

                //get_rotated_vector(vec_pupil_left_proj, vec_pupil_left);
				vec_pupil_left[0] = vec_pupil_left_proj[0];
				vec_pupil_left[1] = vec_pupil_left_proj[1];
				vec_pupil_left[2] = vec_pupil_left_proj[2];

                //Make this unit vector and then apply weight only to Normal
                //makeUnitVector(vec_pupil_left, vec_pupil_left);

				std::cout<<"\nbefore : "<<vec_pupil_left[0]<<", "<<vec_pupil_left[1]<<", "<<vec_pupil_left[2]<<endl;
				std::cout<<"normal : "<<vec_normal[0]<<", "<<vec_normal[1]<<", "<<vec_normal[2]<<endl;

				compute_vector_sum(vec_normal, vec_pupil_left, vec_pupil_left);
/*
                vec_pupil_left[0] += vec_normal[0];
                vec_pupil_left[1] += vec_normal[1];
                vec_pupil_left[2] += vec_normal[2];*/

                //Make vec_gaze a unit vector before this step
                makeUnitVector(vec_pupil_left, vec_pupil_left);
                std::cout<<"final : "<<vec_pupil_left[0]<<", "<<vec_pupil_left[1]<<", "<<vec_pupil_left[2]<<endl;
                /*std::vector<double> vec_temp(3);
                get_reverse_vector(vec_pupil_left, vec_temp);
                makeUnitVector(vec_temp, vec_temp);
                draw_eye_gaze(pupil_left_eye, vec_temp, roi_left_eye_rect, temp);
*/                
                //draw_eye_gaze(pupil_left_eye, vec_pupil_left, roi_left_eye_rect, temp);	




                //////// Process the output gaze point using Kalman filter /////////
                if(k_i==0) {
                	
                	init_kalman(pupil_left_eye.x, pupil_left_eye.y);
                	init_kalman_3(vec_pupil_left);
                	std::cout<<"Kalman filter initiated for left eye"<<std::endl;
                	k_i++;
                }
                double vt_x_nw = pupil_left_eye.x - pt_temp1.x;
                double vt_y_nw = pupil_left_eye.y - pt_temp1.y;
                cv::Point2f sm_le = kalman_predict_correct(pupil_left_eye.x, pupil_left_eye.y, vt_x_nw, vt_y_nw, vt_x_nw - vt_temp1.x, vt_y_nw - vt_temp1.y);

                kalman_predict_correct_3(vec_pupil_left, vec_temp1, sm_le_gaze);
                std::cout<<"\nsm_le_gaze : "<<sm_le_gaze[0]<<", "<<sm_le_gaze[1]<<", "<<sm_le_gaze[2]<<std::endl;


                cv::circle(roi_left_eye, sm_le, 1, cv::Scalar(0, 255, 0), 1, 4, 0);
                /*roi_left_eye.at<cv::Vec3b>(sm_le.x, sm_le.y)[0] = 255;
                roi_left_eye.at<cv::Vec3b>(sm_le.x, sm_le.y)[1] = 255;
                roi_left_eye.at<cv::Vec3b>(sm_le.x, sm_le.y)[2] = 255;
                */
                //draw_eye_gaze(pupil_left_eye, vec_pupil_left, roi_left_eye_rect, temp);
                draw_eye_gaze(sm_le, sm_le_gaze, roi_left_eye_rect, temp);


            }
            //cv::waitKey(0);
            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));
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