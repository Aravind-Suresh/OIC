#include <dlib/opencv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/gui_widgets.h>

using namespace dlib;
using namespace std;

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
        grad.push_back(roi_eye.at<uchar>(x+1, y) - roi_eye.at<uchar>(x-1, y));
        grad.push_back(roi_eye.at<uchar>(x, y+1) - roi_eye.at<uchar>(x, y-1));
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

    return mag;
}

double getAccumulatorScore(cv::Mat roi_eye, cv::Point c) {
    double score = 0;
    int rows = roi_eye.rows;
    int cols = roi_eye.cols;

    for(int i=0;i<cols;i++) {
        for(int j=0;j<rows;j++) {
            if(c.x == i && c.y == j) {
                continue;
            }
            else {
                cv::Point xi = cv::Point(i,j);
                std::vector<double> di, di_unit, gi;

                di.push_back(xi.x - c.x);
                di.push_back(xi.y - c.y);

                double mag = vectorMagnitude(di);

                makeUnitVector(di, mag, di_unit);
                getVectorGradient(roi_eye, xi.x, xi.y, gi);

                double dot_product = scalarProduct(di_unit, gi);
                score += dot_product;
            }
        }
    }

    return ((double)(score)/(rows*cols));
}

cv::Point getPupilCoordinates(cv::Mat roi_eye) {
    int rows = roi_eye.rows;
    int cols = roi_eye.cols;
    cv::Mat mask(rows, cols, CV_64F);

    for(int i=0;i<rows;i++) {

        double *roi_eye_row = roi_eye.ptr<double>(i);
        double *mask_row = mask.ptr<double>(i);

        for(int j=0;j<cols;j++) {
            cv::Point c = cv::Point(i,j);
            roi_eye_row[j] = getAccumulatorScore(roi_eye, c);
        }
    }

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;

    cv::minMaxLoc(mask, &minVal, &maxVal, &minLoc, &maxLoc, cv::Mat());

    return maxLoc;
}

int main()
{
    try
    {
        cv::VideoCapture cap(0);
        image_window win, win_points, win_2, win_3, win_4;

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

                cv::Point pupil_left_eye = getPupilCoordinates(roi_left_eye);

                cv::circle( roi_left_eye, pupil_left_eye, 3, cv::Scalar( 255, 0, 0 ), -1, 8, 0 );

            }

            win.clear_overlay();
            win.set_image(cimg);
            win.add_overlay(render_face_detections(shapes));

            win_points.clear_overlay();
            win_points.set_image(cv_image<bgr_pixel>(temp2));

            win_2.clear_overlay();
            win_2.set_image(cv_image<unsigned char>(imgs[2]));

            win_3.clear_overlay();
            win_3.set_image(cv_image<bgr_pixel>(imgs[3]));

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


