#ifndef PIPELINE_H
#define PIPELINE_H

#include "opencv2/imgproc/imgproc.hpp"
#include "textdetection.h"

namespace pipeline
{
	class Pipeline {
	public:
		int processImage(cv::Mat& img, std::vector<int>& bibNumbers);
	private:
		textdetection::TextDetector textDetector;
	};

}

#endif /* #ifndef PIPELINE_H */

