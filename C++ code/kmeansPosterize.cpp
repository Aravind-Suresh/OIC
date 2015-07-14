#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char** argv) {
    Mat src, src_hsv;

    src = imread(argv[1]);
    imshow("original", src);
    cvtColor(src,src_hsv,CV_RGB2HSV);


    //blur(src_hsv, src_hsv, Size(15,15));
    imshow("blurred", src_hsv);

    Mat p = Mat::zeros(src.cols*src.rows, 5, CV_32F);
    Mat bestLabels, centers, clustered;
    vector<Mat> hsv;
    cv::split(src_hsv, hsv);
    //equalizeHist( hsv, hsv );
    // i think there is a better way to split pixel bgr color
    for(int i=0; i<src.cols*src.rows; i++) {
        // p.at<float>(i,0) = (i/src.cols) / src.rows;
        // p.at<float>(i,1) = (i%src.cols) / src.cols;
        p.at<float>(i,0) = hsv[0].data[i] / 255.0 ;
        //p.at<float>(i,3) = hsv[1].data[i] / 255.0;
        //p.at<float>(i,4) = hsv[2].data[i] / 255.0;
    }

    int K = 3;
    cv::kmeans(p, K, bestLabels,
        TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),
        3, KMEANS_PP_CENTERS, centers);

    int colors[K];
    for(int i=0; i<K; i++) {
        colors[i] = 255/(i+1);
    }
    // i think there is a better way to do this mayebe some Mat::reshape?
    clustered = Mat(src.rows, src.cols, CV_32F);
    for(int i=0; i<src.cols*src.rows; i++) {
        clustered.at<float>(i/src.cols, i%src.cols) = (float)(colors[bestLabels.at<int>(0,i)]);
        cout << bestLabels.at<int>(0,i) << " ";
//              colors[bestLabels.at<int>(0,i)] << " " << 
//              clustered.at<float>(i/src.cols, i%src.cols) << " " <<
//              endl;
    }

    clustered.convertTo(clustered, CV_8U);
    imshow("clustered", clustered);

    waitKey();
    return 0;
}