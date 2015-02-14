#ifndef FACEDETECTION_H
#define FACEDETECTION_H

#include "opencv2/imgproc/imgproc.hpp"

namespace facedetection
{
	int processImage(cv::Mat& img, std::vector<cv::Rect>& faces);
}

#endif /* #ifndef FACEDETECTION_H */
