#include <iostream>
#include <stdio.h>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "facedetection.h"

namespace facedetection {
std::string cascadeName =
		"/home/greg/ws/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";

int processImage(cv::Mat& img, std::vector<cv::Rect>& faces) {
	cv::CascadeClassifier cascade;

	if (!cascade.load(cascadeName)) {
		std::cerr << "ERROR: Could not load classifier cascade" << std::endl;
		return -1;
	}

	double t = 0;
	double scale = 1;

	cv::Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale),
	CV_8UC1);

	cv::cvtColor(img, gray, CV_BGR2GRAY);
	cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
	cv::equalizeHist(smallImg, smallImg);

	t = (double) cvGetTickCount();
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0
	//|CV_HAAR_FIND_BIGGEST_OBJECT
	//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE, cv::Size(30, 30));
	t = (double) cvGetTickCount() - t;
	printf("detection time = %g ms\n",
			t / ((double) cvGetTickFrequency() * 1000.));

	return 0;
}

} /* namespace facedetection */

