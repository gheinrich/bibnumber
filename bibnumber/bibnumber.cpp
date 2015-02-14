#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"


#include "pipeline.h"

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
		inputName.assign(argv[i]);
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

	//detectAndDraw(image, cascade, nestedCascade, scale, tryflip);
	pipeline::processImage(image);

	return 0;
}

void detectAndDraw(Mat& img, CascadeClassifier& cascade,
		CascadeClassifier& nestedCascade, double scale, bool tryflip) {

}
