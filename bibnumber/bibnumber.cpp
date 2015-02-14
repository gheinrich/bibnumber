#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "textdetection.h"

#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

using namespace std;
using namespace cv;

static void help() {
	cout << "\nThis program extracts bib numbers from images.\n"
			"Usage:\n"
			"./bibnumber [image file|folder path|csv ground truth file]\n\n"
			<< endl;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
		CascadeClassifier& nestedCascade, double scale, bool tryflip);

string cascadeName =
		"/home/greg/ws/opencv/data/haarcascades/haarcascade_frontalface_alt.xml";
string nestedCascadeName =
		"../../data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

int main(int argc, const char** argv) {
	CvCapture* capture = 0;
	Mat frame, frameCopy, image;
	const string scaleOpt = "--scale=";
	size_t scaleOptLen = scaleOpt.length();
	const string cascadeOpt = "--cascade=";
	size_t cascadeOptLen = cascadeOpt.length();
	const string nestedCascadeOpt = "--nested-cascade";
	size_t nestedCascadeOptLen = nestedCascadeOpt.length();
	const string tryFlipOpt = "--try-flip";
	size_t tryFlipOptLen = tryFlipOpt.length();
	string inputName;
	bool tryflip = false;

	CascadeClassifier cascade, nestedCascade;
	double scale = 1;

	for (int i = 1; i < argc; i++) {
		cout << "Processing " << i << " " << argv[i] << endl;
		if (cascadeOpt.compare(0, cascadeOptLen, argv[i], cascadeOptLen) == 0) {
			cascadeName.assign(argv[i] + cascadeOptLen);
			cout << "  from which we have cascadeName= " << cascadeName << endl;
		} else if (nestedCascadeOpt.compare(0, nestedCascadeOptLen, argv[i],
				nestedCascadeOptLen) == 0) {
			if (argv[i][nestedCascadeOpt.length()] == '=')
				nestedCascadeName.assign(
						argv[i] + nestedCascadeOpt.length() + 1);
			if (!nestedCascade.load(nestedCascadeName))
				cerr
						<< "WARNING: Could not load classifier cascade for nested objects"
						<< endl;
		} else if (scaleOpt.compare(0, scaleOptLen, argv[i], scaleOptLen)
				== 0) {
			if (!sscanf(argv[i] + scaleOpt.length(), "%lf", &scale)
					|| scale < 1)
				scale = 1;
			cout << " from which we read scale = " << scale << endl;
		} else if (tryFlipOpt.compare(0, tryFlipOptLen, argv[i], tryFlipOptLen)
				== 0) {
			tryflip = true;
			cout
					<< " will try to flip image horizontally to detect assymetric objects\n";
		} else if (argv[i][0] == '-') {
			cerr << "WARNING: Unknown option %s" << argv[i] << endl;
		} else
			inputName.assign(argv[i]);
	}

	if (!cascade.load(cascadeName)) {
		cerr << "ERROR: Could not load classifier cascade" << endl;
		help();
		return -1;
	}

	if ((inputName.empty()) || (!inputName.size())) {
		cerr << "ERROR: Missing parameter" << endl;
		help();
		return -1;
	}

	image = imread(inputName, 1);
	if (image.empty()) {
		cerr << "ERROR:Failed to open image file" << endl;
		help();
		return -1;
	}

	detectAndDraw(image, cascade, nestedCascade, scale, tryflip);

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
		CascadeClassifier& nestedCascade, double scale, bool tryflip) {
	int i = 0;
	double t = 0;
	vector<Rect> faces, faces2;
	const static Scalar colors[] = { CV_RGB(0, 0, 255), CV_RGB(0, 128, 255),
			CV_RGB(0, 255, 255), CV_RGB(0, 255, 0), CV_RGB(255, 128, 0), CV_RGB(
					255, 255, 0), CV_RGB(255, 0, 0), CV_RGB(255, 0, 255) };
	Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale),
	CV_8UC1);

	cvtColor(img, gray, CV_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);

	t = (double) cvGetTickCount();
	cascade.detectMultiScale(smallImg, faces, 1.1, 2, 0
	//|CV_HAAR_FIND_BIGGEST_OBJECT
	//|CV_HAAR_DO_ROUGH_SEARCH
			| CV_HAAR_SCALE_IMAGE, Size(30, 30));
	if (tryflip) {
		flip(smallImg, smallImg, 1);
		cascade.detectMultiScale(smallImg, faces2, 1.1, 2, 0
		//|CV_HAAR_FIND_BIGGEST_OBJECT
		//|CV_HAAR_DO_ROUGH_SEARCH
				| CV_HAAR_SCALE_IMAGE, Size(20, 20));
		for (vector<Rect>::const_iterator r = faces2.begin(); r != faces2.end();
				r++) {
			faces.push_back(
					Rect(smallImg.cols - r->x - r->width, r->y, r->width,
							r->height));
		}
	}
	t = (double) cvGetTickCount() - t;
	printf("detection time = %g ms\n",
			t / ((double) cvGetTickFrequency() * 1000.));
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end();
			r++, i++) {
		Mat smallImgROI;
		vector<Rect> nestedObjects;
		Point center;
		Scalar color = colors[i % 8];
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

		Rect roi = cv::Rect(
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

		Mat subImage(img, roi);
		IplImage ipl_img = subImage;
		if (1) {
			Mat output = textDetection(&ipl_img, 1);
			char filename[100];
			sprintf(filename, "torso-%d.png", i);
			cv::imwrite(filename, subImage);
		}
	}
	cv::imwrite("face-detection.png", img);
}
