#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <boost/algorithm/string/trim.hpp>

#include "pipeline.h"
#include "facedetection.h"
#include "textdetection.h"

#include "stdio.h"

namespace pipeline {

static void vectorAtoi(std::vector<int>&numbers, std::vector<std::string>&text)
{
	for (std::vector<std::string>::iterator it = text.begin(); it != text.end();
					it++) {
		boost::algorithm::trim(*it);
		numbers.push_back(atoi(it->c_str()));
	}
}

int Pipeline::processImage(
		cv::Mat& img,
		std::string svmModel,
		std::vector<int>& bibNumbers) {
#if 0
	int res;
	const double scale = 1;
	std::vector<cv::Rect> faces;
	const static cv::Scalar colors[] = { CV_RGB(0, 0, 255), CV_RGB(0, 128, 255),
			CV_RGB(0, 255, 255), CV_RGB(0, 255, 0), CV_RGB(255, 128, 0), CV_RGB(
					255, 255, 0), CV_RGB(255, 0, 0), CV_RGB(255, 0, 255) };

	res = facedetection::processImage(img, faces);
	if (res < 0) {
		std::cerr << "ERROR: Could not proceed to face detection" << std::endl;
		return -1;
	}


	int i = 0;
	for (std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end();
			r++, i++) {
		cv::Mat smallImgROI;
		std::vector < cv::Rect > nestedObjects;
		cv::Point center;
		cv::Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double) r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3) {
			center.x = cvRound((r->x + r->width * 0.5) * scale);
			center.y = cvRound((r->y + r->height * 0.5) * scale);
			radius = cvRound((r->width + r->height) * 0.25 * scale);
			circle(img, center, radius, color, 3, 8, 0);

		} else
		rectangle(img,
				cvPoint(cvRound(r->x * scale), cvRound(r->y * scale)),
				cvPoint(cvRound((r->x + r->width - 1) * scale),
						cvRound((r->y + r->height - 1) * scale)), color, 3,
				8, 0);

		cv::Rect roi = cv::Rect(
				cvPoint(cvRound((r->x - 0.66 * r->width) * scale),
						cvRound((r->y + 1.5 * r->height) * scale)),
				cvPoint(cvRound((r->x + 1.66 * r->width) * scale),
						cvRound((r->y + 5 * r->height) * scale)));

		roi.x = std::max(roi.x, 0);
		roi.y = std::max(roi.y, 0);
		roi.x = std::min(roi.x, img.cols);
		roi.y = std::min(roi.y, img.rows);
		roi.width = std::min(roi.width, img.cols - roi.x);
		roi.height = std::min(roi.height, img.rows - roi.y);

		//rectangle( img, roi, color, 3, 8, 0);

		cv::Mat subImage(img, roi);
		IplImage ipl_img = subImage;
		if ( //(i==10) &&
				(1)) {
			std::vector<std::string> text;
			const struct TextDetectionParams params = {
					1, /* darkOnLight */
					20, /* maxStrokeLength */
					11, /* minCharacterHeight */
					4, /* maxImgWidthToTextRatio */
					15, /* maxAngle */
					0, /* topBorder: don't discard anything */
					0,  /* bottomBorder: don't discard anything */
					3, /* min chain length */
			};
			textDetector.detect(&ipl_img, params, text);
			vectorAtoi(bibNumbers, text);
			char filename[100];
			sprintf(filename, "torso-%d.png", i);
			cv::imwrite(filename, subImage);
		}
	}
#else
	IplImage ipl_img = img;
	std::vector<std::string> text;
	struct TextDetectionParams params = {
						1, /* darkOnLight */
						15, /* maxStrokeLength */
						11, /* minCharacterHeight */
						100, /* maxImgWidthToTextRatio */
						45, /* maxAngle */
						img.rows * 10/100, /* topBorder: discard top 10% */
						img.rows * 5/100,  /* bottomBorder: discard bottom 5% */
						3, /* min chain len */
						0, /* verify with SVM model up to this chain len */
						0, /* height needs to be this large to verify with model */
				};

	if (!svmModel.empty())
	{
		/* lower min chain len */
		params.minChainLen = 2;
		/* verify with SVM model up to this chain len */
		params.modelVerifLenCrit = 2;
		/* height needs to be this large to verify with model */
		params.modelVerifMinHeight = 15;
	}

	std::vector<Chain> chains;
	std::vector<std::pair<Point2d, Point2d> > compBB;
	std::vector<std::pair<CvPoint, CvPoint> > chainBB;
	textDetector.detect(&ipl_img, params, chains, compBB, chainBB);
	textRecognizer.recognize(&ipl_img, params, svmModel, chains, compBB, chainBB, text);
	vectorAtoi(bibNumbers, text);
#endif
	cv::imwrite("face-detection.png", img);

	return 0;

}

} /* namespace pipeline */

