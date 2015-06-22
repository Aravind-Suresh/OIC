

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


void otsuThreshold(cv::Mat input, cv::Mat& output , bool invertParam)
{
    if(invertParam==true)
    cv::threshold(input, output, 0, 255, CV_THRESH_BINARY_INV + CV_THRESH_OTSU);
    
    else
    cv::threshold(input, output, 0, 255, CV_THRESH_BINARY+ CV_THRESH_OTSU);
       
}

void preprocessROI(cv::Mat& roi_eye) {
    GaussianBlur(roi_eye, roi_eye, cv::Size(3,3), 0, 0);
    equalizeHist( roi_eye, roi_eye );
}


bool comparatorContourAreas ( std::vector<cv::Point> c1, std::vector<cv::Point> c2 ) {
    double i = fabs( contourArea(cv::Mat(c1)) );
    double j = fabs( contourArea(cv::Mat(c2)) );
    return ( i < j );
}   


void distanceTransformThreshold(cv::Mat input, cv::Mat& input_dt , cv::Mat& output, double dt_thresholdValue )
{
    cv::Mat temp(input.rows, input.cols, CV_32F, cv::Scalar::all(0));

    cv::distanceTransform(input, input_dt, CV_DIST_L2, 3);
    cv::normalize(input_dt, input_dt, 0.1, 1, cv::NORM_MINMAX);

    cv::threshold(input_dt, temp , dt_thresholdValue , 255, 1);

    temp.convertTo(output, CV_8UC1);
                
}

void MorphOpening(cv::Mat input, cv::Mat& output, cv::Mat element)
{
    morphologyEx( input, output, CV_MOP_OPEN, element );

}

void findAndSortContours(cv::Mat input, std::vector<std::vector<cv::Point> >& contours, std::vector<cv::Vec4i>& hierarchy)
{

    cv::findContours(input, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
    sort(contours.begin(),contours.end(),comparatorContourAreas);

}

cv::Point findAndDrawPupil(cv::Mat& roi_left_eye , cv::Rect eye )
{
                
    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    std::vector<int> row(5,1);
    std::vector<std::vector<int> > kernelOpen(5,row);

    int morph_size = 2;
    cv::Mat element = getStructuringElement( cv::MORPH_RECT, cv::Size( 2*morph_size + 1, 2*morph_size+1 ), cv::Point( morph_size, morph_size ) );


    cv::Mat roi_left_eye_temp1(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));
    
    cv::Mat roi_left_eye_otsu;
    cv::Mat roi_left_eye_out(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));
    
    // cv::Mat roi_left_eye_dt(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));
    // cv::Mat roi_left_eye_dtThresh(roi_left_eye.rows, roi_left_eye.cols, CV_8UC1, cv::Scalar::all(0));
    
    roi_left_eye.copyTo(roi_left_eye_temp1);
    preprocessROI(roi_left_eye_temp1);
 

//compute otsu threshold of eye roi

    otsuThreshold(roi_left_eye_temp1,roi_left_eye_otsu,true);
                
//compute distance transform thresholding of eye roi

    //distanceTransformThreshold(roi_left_eye_temp1,roi_left_eye_dt,roi_left_eye_dtThresh);
 
                
//compute opening of otsu image                
    
    MorphOpening( roi_left_eye_otsu, roi_left_eye_out, element );
                
//compute largest contour
                
    findAndSortContours(roi_left_eye_out, contours, hierarchy);
    //cv::drawContours( roi_left_eye, contours, contours.size()-1,  cv::Scalar (255, 0, 0 ), CV_FILLED, 8, hierarchy );
                



    std::vector<cv::Point> largestContour = contours[contours.size()-1];
    
    if(contours.size()>=0)
            {
                cout<<(largestContour).size()<<endl;

//least fit the contours
                if((largestContour).size()>4)
                { 

                cv::RotatedRect rRect = fitEllipse( largestContour);
                cv::Point2f vertices[4];
                rRect.points(vertices);
                
                cv::Point pupil_left_eye = cv::Point((vertices[0].x + vertices[1].x + vertices[2].x + vertices[3].x)/4,(vertices[0].y + vertices[1].y + vertices[2].y + vertices[3].y)/4);

                cv::ellipse(roi_left_eye, rRect, cv::Scalar(255), 1, 8);
                cv::circle( roi_left_eye, pupil_left_eye, 1, cv::Scalar(255), -1, 8, 0 );
                // std::cout<<"Left Pupil@ : "<<pupil_left_eye.x<<","<<pupil_left_eye.y<<std::endl;
                
                return pupil_left_eye;

                }

            }
    return cv::Point(0,0);

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
                // std::cout<<"roi_left_eye dim : "<<roi_left_eye.rows<<","<<roi_left_eye.cols<<std::endl;

                                
                cv::Point pupil_left_eye = findAndDrawPupil(roi_left_eye,cv::boundingRect(vec_pts_left_eye));
                             

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