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

 /** Function Headers */

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
  printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
  
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


 /** @function main */

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

                cv::Rect roi_left_eye_rect = cv::boundingRect(vec_pts_left_eye);

                cv::Mat roi_left_eye = imgs[5](cv::boundingRect(vec_pts_left_eye));
                // std::cout<<"roi_left_eye dim : "<<roi_left_eye.rows<<","<<roi_left_eye.cols<<std::endl;
                cv::Mat roi_left_eye_temp;
                roi_left_eye.copyTo(roi_left_eye_temp);
                preprocessROI(roi_left_eye_temp);

                cv::Point pupil_left_eye = findEyeCenter(roi_left_eye_temp,cv::boundingRect(vec_pts_left_eye),"hello");

                //const Point face_tl_corner = faces[i].tl_corner();

                //cv::Point center= cv::Point(faces[i].tl_corner().x() + roi_left_eye_rect.x+pupil_left_eye.x,faces[i].tl_corner().y() + roi_left_eye_rect.y+pupil_left_eye.y);
                
                cv::circle( roi_left_eye, pupil_left_eye, 3, cv::Scalar(255), -1, 8, 0 );
                // std::cout<<"Left Pupil@ : "<<pupil_left_eye.x<<","<<pupil_left_eye.y<<std::endl;


                

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

