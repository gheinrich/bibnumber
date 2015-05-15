#ifndef TRAIN_H
#define TRAIN_H

#include <string>

#include "opencv2/imgproc/imgproc.hpp"

namespace train
{
	cv::Mat hogVisualizeSingleBlock(cv::Mat& origImg,
		std::vector<float>& descriptorValues, cv::Size winSize,
		cv::Size cellSize, int scaleFactor, double viz_factor);
	cv::Mat hogVisualizeStdBlkSize(cv::Mat& origImg,
			std::vector<float>& descriptorValues, cv::Size winSize,
			cv::Size cellSize, int scaleFactor, double viz_factor);
	int process(std::string trainDir, std::string inputDir);
}

#endif /* #ifndef TRAIN_H */
