#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>
#include <math.h>
#include <stdlib.h>
#include <queue>

using namespace dlib;
using namespace std;

double gradientThresh;

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
}
return mask;
}

double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
  cv::Scalar stdMagnGrad, meanMagnGrad;
  meanStdDev(mat, meanMagnGrad, stdMagnGrad);
  double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
  return stdDevFactor * stdDev + meanMagnGrad[0];
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

void getVectorGradient(cv::Mat roi_eye, int x, int y, std::vector<double>& grad) {
    int rows = roi_eye.rows;
    int cols = roi_eye.cols;

    if(x > 0 && x < cols && y > 0 && y < rows) {
        grad.push_back((roi_eye.at<uchar>(x+1, y) - roi_eye.at<uchar>(x-1, y)) / 2.0);
        grad.push_back((roi_eye.at<uchar>(x, y+1) - roi_eye.at<uchar>(x, y-1)) / 2.0);
    }
    else {
        grad.push_back(0);
        grad.push_back(0);
    }
}

void makeUnitVector(std::vector<double> vec, double magnitude, std::vector<double>& unit_vector) {
    for(int i=0;i<vec.size();i++) {
        unit_vector.push_back(((double)(vec[i])/magnitude));
    }
}

double vectorMagnitude(std::vector<double> vec) {
    double mag = 0;

    for(int i=0;i<vec.size();i++) {
        mag += vec[i]*vec[i];
    }

    return sqrt(mag);
}

double getAccumulatorScore(cv::Mat roi_eye, cv::Point c) {
    double score = 0;
    int rows = roi_eye.rows;
    int cols = roi_eye.cols;

    cv::Mat grad_storage = cv::Mat::zeros(rows, cols, CV_64F);

    for(int i=0;i<rows;i++) {
        double *grad_storage_row = grad_storage.ptr<double>(i);
        for(int j=0;j<cols;j++) {
            cv::Point xi = cv::Point(j,i);
            if(c.x == j && c.y == i) {
                continue;
            }
            else {
                std::vector<double> gi;
                getVectorGradient(roi_eye, xi.x, xi.y, gi);
                grad_storage_row[j] = vectorMagnitude(gi);
            }
        }
    }

    gradientThresh = computeDynamicThreshold(grad_storage, 50.0);

    for(int i=0;i<rows;i++) {
        for(int j=0;j<cols;j++) {
            if(c.x == j && c.y == i) {
                continue;
            }
            else {
                cv::Point xi = cv::Point(j,i);
                std::vector<double> gi;

                getVectorGradient(roi_eye, xi.x, xi.y, gi);

                if(vectorMagnitude(gi) > gradientThresh) {
                    std::vector<double> di, di_unit;

                    di.push_back(xi.x - c.x);
                    di.push_back(xi.y - c.y);

                    double mag = vectorMagnitude(di);

                    makeUnitVector(di, mag, di_unit);
                    double dot_product = std::max(0.0, scalarProduct(di_unit, gi));
                    score += ((255 - (int)(roi_eye.at<uchar>(j,i)))*dot_product)/255.0;
                }
            }
        }
    }

    //std::cout<<score<<" ";

    return ((double)(score)/(rows*cols));
}

cv::Point getPupilCoordinates(cv::Mat roi_eye) {
    int rows = roi_eye.rows;
    int cols = roi_eye.cols;

    //std::cout<<"roi_eye dim : "<<rows<<","<<cols<<std::endl;

    cv::Mat mask(rows, cols, CV_64F);
    cv::Mat roi_eye_clone;
    cv::Mat grad_storage;

    //Canny( roi_eye, roi_eye_clone, 30, 30, 3 );
    roi_eye.copyTo(roi_eye_clone);

    for(int i=0;i<rows;i++) {

        double *roi_eye_row = roi_eye.ptr<double>(i);
        double *mask_row = mask.ptr<double>(i);

        for(int j=0;j<cols;j++) {
            cv::Point c = cv::Point(i,j);
            mask_row[j] = getAccumulatorScore(roi_eye_clone, c);
        }
    }

    double minVal, maxVal = 0;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(mask, NULL, &maxVal, NULL, &maxLoc, cv::Mat());

    double numGradients = (rows*cols);
    cv::Mat mask_convert;
    mask.convertTo(mask_convert, CV_32F,1.0/numGradients);

    cv::Mat floodClone;
    double floodThresh = maxVal * 0.97;
    cv::threshold(mask_convert, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
    
    cv::Mat mask_flood = floodKillEdges(floodClone);
    
    cv::minMaxLoc(mask_flood, NULL,&maxVal,NULL,&maxLoc,mask_flood);

    std::cout<<maxVal<<"\t";
    return maxLoc;
}

void preprocessROI(cv::Mat& roi_eye) {
    //GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
}

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        image_window win, win_points, win_2, win_4;

        frontal_face_detector detector = get_frontal_face_detector();
        shape_predictor pose_model;
        deserialize("shape_predictor_68_face_landmarks.dat") >> pose_model;

        std::vector<cv::Mat> imgs(10);
        cv::Mat temp, temp2;

        while(!win.is_closed())
        {
            cap >> temp;
            cv::flip(temp, temp, 1);

            temp.copyTo(temp2);
            for(int i=0;i<imgs.size();i++) {
                temp2.copyTo(imgs[i]);
            }

            cvtColor(imgs[0], imgs[0], CV_BGR2GRAY);
            cvtColor(imgs[1], imgs[1], CV_BGR2GRAY);
            cvtColor(imgs[2], imgs[2], CV_BGR2GRAY);
            cvtColor(imgs[4], imgs[4], CV_BGR2GRAY);
            cvtColor(imgs[5], imgs[5], CV_BGR2GRAY);

            cv_image<bgr_pixel> cimg(temp);

            std::vector<rectangle> faces = detector(cimg);

            std::vector<full_object_detection> shapes;

            for (unsigned long i = 0; i < faces.size(); ++i) {
                full_object_detection shape = pose_model(cimg, faces[i]);
                shapes.push_back(shape);

                for(unsigned long int j=0;j<shape.num_parts();j++) {
                    std::stringstream ss;
                    ss << j;
                    std::string text = ss.str();
                    int fontFace = cv::FONT_HERSHEY_SCRIPT_SIMPLEX;
                    double fontScale = 0.3;
                    int thickness = 1;

                    cv::Point point(shape.part(j).x(), shape.part(j).y());

                    cv::putText(temp2, text, point, fontFace, fontScale, cv::Scalar::all(255), thickness,8);
                }

                std::vector<std::vector<cv::Point> > vec_lines(0);
                std::vector<cv::Point> pose_track_shape(0);

                pose_track_shape.push_back(cv::Point(shape.part(0).x(), shape.part(0).y()));
                pose_track_shape.push_back(cv::Point(shape.part(21).x(), shape.part(21).y()));
                pose_track_shape.push_back(cv::Point(shape.part(22).x(), shape.part(22).y()));
                pose_track_shape.push_back(cv::Point(shape.part(16).x(), shape.part(16).y()));
                pose_track_shape.push_back(cv::Point(shape.part(33).x(), shape.part(33).y()));

                vec_lines.push_back(pose_track_shape);

                std::vector<cv::Point> vec_pts_left_eye(0), vec_pts_right_eye(0);
                std::vector<std::vector<cv::Point> > vec_vec_rois;

                for(int j=42;j<=47;j++) {
                    vec_pts_left_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
                }

                for(int j=36;j<=41;j++) {
                    vec_pts_right_eye.push_back(cv::Point(shape.part(j).x(), shape.part(j).y()));
                }

                cv::approxPolyDP(vec_pts_left_eye, vec_pts_left_eye, 0.1, true);
                imgs[1] = cv::Scalar::all(0);

                vec_vec_rois.push_back(vec_pts_left_eye);
                vec_vec_rois.push_back(vec_pts_right_eye);

                cv::drawContours(imgs[1], vec_vec_rois, -1, cv::Scalar(255), CV_FILLED);
                cv::bitwise_and(imgs[4], imgs[1], imgs[2]);

                cv::Mat roi_left_eye = imgs[5](cv::boundingRect(vec_pts_left_eye));
                //std::cout<<"roi_left_eye dim : "<<roi_left_eye.rows<<","<<roi_left_eye.cols<<std::endl;

                //preprocessROI(roi_left_eye);

                //cv::Point pupil_left_eye = getPupilCoordinates(roi_left_eye);

                //cv::circle( roi_left_eye, pupil_left_eye, 3, cv::Scalar(255), -1, 8, 0 );
                //std::cout<<"Left Pupil@ : "<<pupil_left_eye.x<<","<<pupil_left_eye.y<<std::endl;


                

            }

            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));

            win_points.clear_overlay();
            win_points.set_image(cv_image<bgr_pixel>(temp2));

            win_2.clear_overlay();
            win_2.set_image(cv_image<unsigned char>(imgs[2]));
/*
            win_3.clear_overlay();
            win_3.set_image(cv_image<bgr_pixel>(imgs[3]));*/

            // win_4.clear_overlay();
            // win_4.set_image(cv_image<unsigned char>(imgs[5]));
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
