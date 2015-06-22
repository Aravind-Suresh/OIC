

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

double PI=3.14159265359;
void CannyThreshold(cv::Mat src_gray, cv::Mat& edge_gray, int lowThreshold, int highThreshold, int kernel_size)
{
    //blur( src_gray, edge_gray, Size(3,3) );
    cv::GaussianBlur( src_gray, edge_gray, cv::Size(5,5), 2, 2 );
    cv::Canny( edge_gray, edge_gray, lowThreshold, highThreshold, kernel_size );
}
void getVerticalEdges(cv::Mat grad_x,cv::Mat grad_y,cv::vector<cv::Point>& edges )
{
    for(int i=0;i<grad_x.rows;i++)
        for(int j=0;j<grad_x.cols;j++)
            {

                double ang = atan2(grad_y.at<uchar>(j,i), grad_x.at<uchar>(j,i)) * 180 / PI;
                if (ang < 0)
                    ang = 360 + ang;

                if(ang>80 && ang<100)
                    edges.push_back(cv::Point(j,i));
            }

}

bool comparatorContourAreas ( cv::vector<cv::Point> c1, cv::vector<cv::Point> c2 ) {
    double i = fabs( contourArea(cv::Mat(c1)) );
    double j = fabs( contourArea(cv::Mat(c2)) );
    return ( i < j );
}   

void preprocessROI(cv::Mat& roi_eye) {
    GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
    equalizeHist( roi_eye, roi_eye );
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

        cv::vector<cv::vector<cv::Point> > contours;
        cv::vector<cv::Vec4i> hierarchy;

        cv::vector<int> row(5,1);
        cv::vector<cv::vector<int> > kernelOpen(5,row);

        int morph_size = 2;
        cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );

        int lowThreshold = 30;
        int ratio = 1;
        int kernel_size = 3;

        

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
                // std::cout<<"roi_left_eye dim : "<<roi_left_eye.rows<<","<<roi_left_eye.cols<<std::endl;

                // preprocessROI(roi_left_eye);

                
                // std::cout<<"Left Pupil@ : "<<pupil_left_eye.x<<","<<pupil_left_eye.y<<std::endl;


                // cv::Mat roi_left_eye_otsu;
                // cv::Mat roi_left_eye_otsu_open;
                // cv::Mat roi_left_eye_temp2 (roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));;
                // cv::Mat roi_left_eye_temp3 (roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));;
                
                cv::Mat roi_left_eye_dt(roi_left_eye.rows, roi_left_eye.cols, CV_32F, cv::Scalar::all(0));
                cv::Mat roi_left_eye_edge(roi_left_eye.rows, roi_left_eye.cols, CV_32F, cv::Scalar::all(0));
                cv::Mat roi_left_eye_dt_thresh (roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));;
                cv::Mat roi_left_eye_out(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));
                cv::Mat roi_left_eye_temp1(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));

                roi_left_eye.copyTo(roi_left_eye_temp1);
                preprocessROI(roi_left_eye_temp1);
 






//compute otsu threshold of eye roi

                cv::threshold(roi_left_eye_temp1, roi_left_eye_out, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
                //cv::threshold(roi_left_eye, roi_left_eye_out, 25, 255,CV_THRESH_BINARY );

                // cv::distanceTransform(roi_left_eye, roi_left_eye_dt, CV_DIST_L2, 3);
                // cv::normalize(roi_left_eye_dt, roi_left_eye_dt, 0.1, 1, cv::NORM_MINMAX);

                // cv::threshold(roi_left_eye_dt, roi_left_eye_dt_thresh, (20/255.0), 255, 1);

                // roi_left_eye_dt_thresh.convertTo(roi_left_eye_out, CV_8UC1);
                // cv::imshow("dt_thresh",roi_left_eye_out);

//compute opening of otsu image                
                //cv::morphologyEx(roi_left_eye_otsu, roi_left_eye_otsu_open, CV_MOP_OPEN, kernelOpen, cv::Point(0,0), 1, cv::BORDER_CONSTANT);
                morphologyEx( roi_left_eye_out, roi_left_eye_out, CV_MOP_OPEN, element );

                
//compute the canny threshold

                //roi_left_eye_out.copyTo(roi_left_eye);
                CannyThreshold(roi_left_eye_out, roi_left_eye_edge, lowThreshold, lowThreshold*ratio, kernel_size);

//compute the x,y derivatives
                cv::Mat grad_x, grad_y;
                cv::Mat abs_grad_x, abs_grad_y;
                int scale = 1;
                int delta = 0;
                int ddepth = CV_16S;

            /// Gradient X
                cv::Sobel( roi_left_eye_edge, grad_x, ddepth, 1, 0, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::convertScaleAbs( grad_x, abs_grad_x );

            /// Gradient Y
                cv::Sobel( roi_left_eye_edge, grad_y, ddepth, 0, 1, 3, scale, delta, cv::BORDER_DEFAULT );
                cv::convertScaleAbs( grad_y, abs_grad_y );

//get only almost vertical edges
                cv::vector<cv::Point> vert_edges;
                getVerticalEdges(abs_grad_x,abs_grad_y,vert_edges);

//compute largest contour
                // cv::findContours(roi_left_eye_out, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
                // sort(contours.begin(),contours.end(),comparatorContourAreas);

                //cv::drawContours( roi_left_eye, contours, contours.size()-1,  cv::Scalar (255, 0, 0 ), CV_FILLED, 8, hierarchy );
                //cv::drawContours( roi_left_eye, contours, contours.size()-1,  cv::Scalar (255, 0, 0 ), CV_RETR_LIST, 8, hierarchy );

                if(vert_edges.size()>=0)
                {
                cout<<vert_edges.size()<<endl;

//least fit the contours
                if(vert_edges.size()>4)
                { 

                cv::RotatedRect rRect = cv::fitEllipse( vert_edges);
                cv::Point2f vertices[4];
                rRect.points(vertices);
                
                cv::Point pupil_left_eye = cv::Point((vertices[0].x + vertices[1].x + vertices[2].x + vertices[3].x)/4,(vertices[0].y + vertices[1].y + vertices[2].y + vertices[3].y)/4);

                cv::ellipse(roi_left_eye, rRect, cv::Scalar(255), 1, 8);
                cv::circle( roi_left_eye, pupil_left_eye, 1, cv::Scalar(255), -1, 8, 0 );
                // std::cout<<"Left Pupil@ : "<<pupil_left_eye.x<<","<<pupil_left_eye.y<<std::endl;
                }

                }

                

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

            win_4.clear_overlay();
            win_4.set_image(cv_image<unsigned char>(imgs[5]));
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