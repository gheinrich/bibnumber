#include <iostream>
#include <string>
#include <ctime>
#include <cstdlib>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml/ml.hpp>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#include "train.h"
#include "batch.h"

namespace fs = boost::filesystem;

class LinearSVM: public CvSVM {
public:
	void getSupportVector(std::vector<float>& support_vector) const;
};

void LinearSVM::getSupportVector(std::vector<float>& support_vector) const {

	int sv_count = get_support_vector_count();
	const CvSVMDecisionFunc* df = decision_func;
	const double* alphas = df[0].alpha;
	double rho = df[0].rho;
	int var_count = get_var_count();
	support_vector.resize(var_count, 0);
	for (unsigned int r = 0; r < (unsigned) sv_count; r++) {
		float myalpha = alphas[r];
		const float* v = get_support_vector(r);
		for (int j = 0; j < var_count; j++, v++) {
			support_vector[j] += (-myalpha) * (*v);
		}
	}
	support_vector.push_back(rho);
}


/**
 * Compute HOG feature descriptor from input image
 * @param filename file name of image
 * @param descriptor HOG feature descriptor
 * @param hog instance of cv::HOGDescriptor
 */
static void computeHOGDescriptor(const std::string filename,
		std::vector<float>& descriptor, cv::HOGDescriptor& hog) {
	cv::Mat imageMat = cv::imread(filename, 1);
	cv::Mat resizedMat;

	// resize to HOGDescriptor dimensions
	cv::resize(imageMat, resizedMat, hog.winSize, 0, 0);
	hog.compute(resizedMat, descriptor);
}

static inline double square(double x) {
	return x * x;
}

namespace train {

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
cv::Mat hogVisualizeStdBlkSize(cv::Mat& origImg,
		std::vector<float>& descriptorValues, cv::Size winSize,
		cv::Size cellSize, int scaleFactor, double viz_factor) {
	cv::Mat visual_image;
	cv::resize(origImg, visual_image,
			cv::Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float) gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;
	float*** gradientStrengths = new float**[cells_in_y_dir];
	int** cellUpdateCounter = new int*[cells_in_y_dir];
	for (int y = 0; y < cells_in_y_dir; y++) {
		gradientStrengths[y] = new float*[cells_in_x_dir];
		cellUpdateCounter[y] = new int[cells_in_x_dir];
		for (int x = 0; x < cells_in_x_dir; x++) {
			gradientStrengths[y][x] = new float[gradientBinSize];
			cellUpdateCounter[y][x] = 0;

			for (int bin = 0; bin < gradientBinSize; bin++)
				gradientStrengths[y][x][bin] = 0.0;
		}
	}

	// nr of blocks = nr of cells - 1
	// since there is a new block on each cell (overlapping blocks!) but the last one
	int blocks_in_x_dir = cells_in_x_dir - 1;
	int blocks_in_y_dir = cells_in_y_dir - 1;

	// compute gradient strengths per cell
	int descriptorDataIdx = 0;
	for (int blockx = 0; blockx < blocks_in_x_dir; blockx++) {
		for (int blocky = 0; blocky < blocks_in_y_dir; blocky++) {
			// 4 cells per block ...
			for (int cellNr = 0; cellNr < 4; cellNr++) {
				// compute corresponding cell nr
				int cellx = blockx;
				int celly = blocky;
				if (cellNr == 1)
					celly++;
				if (cellNr == 2)
					cellx++;
				if (cellNr == 3) {
					cellx++;
					celly++;
				}

				for (int bin = 0; bin < gradientBinSize; bin++) {
					float gradientStrength = descriptorValues[descriptorDataIdx];
					descriptorDataIdx++;

					gradientStrengths[celly][cellx][bin] += gradientStrength;

				} // for (all bins)

				// note: overlapping blocks lead to multiple updates of this sum!
				// we therefore keep track how often a cell was updated,
				// to compute average gradient strengths
				cellUpdateCounter[celly][cellx]++;

			} // for (all cells)

		} // for (all block x pos)
	} // for (all block y pos)

	// compute average gradient strengths
	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {

			float NrUpdatesForThisCell = (float) cellUpdateCounter[celly][cellx];

			// compute average gradient strenghts for each gradient bin direction
			for (int bin = 0; bin < gradientBinSize; bin++) {
				gradientStrengths[celly][cellx][bin] /= NrUpdatesForThisCell;
			}
		}
	}

	// draw cells
	for (int celly = 0; celly < cells_in_y_dir; celly++) {
		for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height / 2;

			cv::rectangle(visual_image,
					cv::Point(drawX * scaleFactor, drawY * scaleFactor),
					cv::Point((drawX + cellSize.width) * scaleFactor,
							(drawY + cellSize.height) * scaleFactor),
					CV_RGB(100, 100, 100), 1);

			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++) {
				float currentGradStrength = gradientStrengths[celly][cellx][bin];

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
										  // to see the lines better

				// compute line coordinates
				float x1 = mx
						- dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my
						- dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx
						+ dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my
						+ dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				cv::line(visual_image,
						cv::Point(x1 * scaleFactor, y1 * scaleFactor),
						cv::Point(x2 * scaleFactor, y2 * scaleFactor),
						CV_RGB(0, 0, 255), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)

	// don't forget to free memory allocated by helper data structures!
	for (int y = 0; y < cells_in_y_dir; y++) {
		for (int x = 0; x < cells_in_x_dir; x++) {
			delete[] gradientStrengths[y][x];
		}
		delete[] gradientStrengths[y];
		delete[] cellUpdateCounter[y];
	}
	delete[] gradientStrengths;
	delete[] cellUpdateCounter;

	return visual_image;

}

// HOGDescriptor visual_imagealizer
// adapted for arbitrary size of feature sets and training images
cv::Mat hogVisualizeSingleBlock(cv::Mat& origImg,
		std::vector<float>& descriptorValues, cv::Size winSize,
		cv::Size cellSize, int scaleFactor, double viz_factor) {
	cv::Mat visual_image;
	cv::resize(origImg, visual_image,
			cv::Size(origImg.cols * scaleFactor, origImg.rows * scaleFactor));

	int gradientBinSize = 9;
	// dividing 180° into 9 bins, how large (in rad) is one bin?
	float radRangeForOneBin = 3.14 / (float) gradientBinSize;

	// prepare data structure: 9 orientation / gradient strenghts for each cell
	int cells_in_x_dir = winSize.width / cellSize.width;
	int cells_in_y_dir = winSize.height / cellSize.height;

	// draw cells
	int idx = 0;
	for (int cellx = 0; cellx < cells_in_x_dir; cellx++) {
	for (int celly = 0; celly < cells_in_y_dir; celly++) {

			int drawX = cellx * cellSize.width;
			int drawY = celly * cellSize.height;

			int mx = drawX + cellSize.width / 2;
			int my = drawY + cellSize.height / 2;

			cv::rectangle(visual_image,
					cv::Point(drawX * scaleFactor, drawY * scaleFactor),
					cv::Point((drawX + cellSize.width) * scaleFactor,
							(drawY + cellSize.height) * scaleFactor),
					CV_RGB(100, 100, 100), 1);

			//std::cout << std::endl << "x=" << cellx << " y=" << celly;
			// draw in each cell all 9 gradient strengths
			for (int bin = 0; bin < gradientBinSize; bin++) {
				float currentGradStrength = descriptorValues[idx++];
				//std::cout << " bin=" << bin << ":" << currentGradStrength;

				// no line to draw?
				if (currentGradStrength == 0)
					continue;

				float currRad = bin * radRangeForOneBin + radRangeForOneBin / 2;

				float dirVecX = cos(currRad);
				float dirVecY = sin(currRad);
				float maxVecLen = cellSize.width / 2;
				float scale = viz_factor; // just a visual_imagealization scale,
										  // to see the lines better

				// compute line coordinates
				float x1 = mx
						- dirVecX * currentGradStrength * maxVecLen * scale;
				float y1 = my
						- dirVecY * currentGradStrength * maxVecLen * scale;
				float x2 = mx
						+ dirVecX * currentGradStrength * maxVecLen * scale;
				float y2 = my
						+ dirVecY * currentGradStrength * maxVecLen * scale;

				// draw gradient visual_imagealization
				cv::line(visual_image,
						cv::Point(x1 * scaleFactor, y1 * scaleFactor),
						cv::Point(x2 * scaleFactor, y2 * scaleFactor),
						CV_RGB(0, 0, 255), 1);

			} // for (all bins)

		} // for (cellx)
	} // for (celly)

	return visual_image;

}

int process(std::string trainDir, std::string inputDir) {

	if ((!fs::is_directory(trainDir)) || (!fs::is_directory(inputDir))) {
		std::cerr << "Invalid parameters (not directories as expeceted)";
		return -1;
	}

	cv::HOGDescriptor hog(cv::Size(128, 64), /* windows size */
	cv::Size(16, 16), /* block size */
	cv::Size(8, 8), /* block stride */
	cv::Size(8, 8), /* cell size */
	9 /* nbins */
	);

	/* find positive image files names */
	std::cout << "Training from positives in " << trainDir << " image data in "
			<< inputDir << std::endl;
	std::vector<fs::path> positiveImgFiles = batch::getImageFiles(trainDir);

	/* find full image files names */
	std::cout << "Training from full images in " << trainDir
			<< " image data in " << inputDir << std::endl;
	std::vector<fs::path> fullImgFiles = batch::getImageFiles(inputDir);

	const int nRandomNegativesPerImage = 5;
	unsigned int nPositives = positiveImgFiles.size();
	unsigned int nNegatives = fullImgFiles.size() * nRandomNegativesPerImage;
	unsigned int rows = nPositives + nNegatives; /* one row per training example */
	unsigned int cols = hog.getDescriptorSize(); /* one column per descriptor field */

	cv::Mat trainingData(rows, cols, CV_32FC1);

	/* compute features for positive examples */
	cv::Mat aggregateDescriptor(1, cols, CV_32FC1, cv::Scalar(0));
	for (unsigned i = 0; i < nPositives; i++) {
		std::cout << "Opening positive example " << positiveImgFiles[i].string()
				<< std::endl;
		std::vector<float> descriptor;
		computeHOGDescriptor(positiveImgFiles[i].string().c_str(), descriptor,
				hog);
		trainingData.row(i) = cv::Mat(descriptor).t();
		aggregateDescriptor += trainingData.row(i);
	}

	aggregateDescriptor /= nPositives;
	const float*p = aggregateDescriptor.ptr<float>(0);
	std::vector<float> vec(p, p+aggregateDescriptor.cols);
	cv::Mat zMat = cv::Mat::zeros(hog.winSize,CV_32FC1);
	cv::Mat visualImage = hogVisualizeStdBlkSize(zMat, vec,
				hog.winSize, hog.cellSize, 5, 2.5);
	cv::imwrite("hog-viz.png", visualImage);

	/* compute features for negative examples */
	std::srand(std::time(0)); // use current time as seed for random generator
	for (unsigned i = 0, end = fullImgFiles.size(); i < end; i++) {
		std::cout << "Opening full image " << fullImgFiles[i].string()
				<< std::endl;
		cv::Mat imageMat = cv::imread(fullImgFiles[i].string().c_str(), 1);
		for (unsigned int j = 0; j < nRandomNegativesPerImage; j++) {
			std::vector<float> descriptor;
			int x = std::rand() % (imageMat.cols - hog.winSize.width);
			int y = std::rand() % (imageMat.rows - hog.winSize.height);
			int idx = nPositives + j + i * nRandomNegativesPerImage;
			cv::Rect roi = cv::Rect(cv::Point(x, y), hog.winSize);
			std::cout << "Sampling random patch #" << idx << " from " << roi
					<< std::endl;
			hog.compute(imageMat(roi), descriptor);
			trainingData.row(idx) = cv::Mat(descriptor).t();
		}
	}

	std::cout << "descriptors :" << trainingData.size() << std::endl;
	cv::Mat labels(rows, 1, CV_32FC1, cv::Scalar(-1.0));
	labels.rowRange(0, nPositives) = cv::Scalar(1.0);

	LinearSVM svm;
	CvSVMParams params;
	params.svm_type = CvSVM::C_SVC;
	params.kernel_type = CvSVM::LINEAR;
	params.term_crit = cvTermCriteria( CV_TERMCRIT_ITER, 10000, 1e-6);

	svm.train(trainingData, labels, cv::Mat(), cv::Mat(), params);
	svm.save("svm.xml");

	for (unsigned int i = 0; i < rows; i++) {
		/* show measure of distance to average positive feature */
		double distance=0;
		for(unsigned int j=0;j<cols;j++)
		{
			distance += square(trainingData.at<float>(i,j) - aggregateDescriptor.at<float>(0,j));
		}
		float prediction = svm.predict(trainingData.row(i));
		std::cout << i << " dist=" << distance <<" prediction=" << prediction << std::endl;
	}

	for (unsigned i = 0, end = fullImgFiles.size(); i < end; i++) {
		std::cout << "Opening full image " << fullImgFiles[i].string()
				<< std::endl;
		cv::Mat imageMat = cv::imread(fullImgFiles[i].string().c_str(), 1);
		std::vector<float> descriptor;
		int x = std::rand() % (imageMat.cols - hog.winSize.width);
		int y = std::rand() % (imageMat.rows - hog.winSize.height);
		cv::Rect roi = cv::Rect(cv::Point(x, y), hog.winSize);
		std::cout << "Sampling random patch from " << roi << std::endl;
		hog.compute(imageMat(roi), descriptor);
		float prediction = svm.predict(cv::Mat(descriptor).t());
		std::cout << i << " prediction=" << prediction << std::endl;
		if (prediction > 0.5) {
			char *filename;
			asprintf(&filename, "positive-%d.png", i);
			cv::imwrite(filename, imageMat(roi));
			free(filename);
		}
	}

#if 0
	{
		std::cout << "Opening full image " << fullImgFiles[0].string()
		<< std::endl;
		cv::Mat imageMat = cv::imread(fullImgFiles[0].string().c_str(), 1);
		std::vector<float> support_vector;
		svm.getSupportVector(support_vector);
		std::vector<cv::Rect> locations;
		hog.setSVMDetector(support_vector);
		hog.detectMultiScale(imageMat, locations, 0.0, cv::Size(), cv::Size(), 1.01);
		for (unsigned int i = 0; i < locations.size(); ++i) {
			cv::rectangle(imageMat, locations[i], cv::Scalar(64, 255, 64), 3);
		}
		cv::imwrite("detection.png", imageMat);
	}
#endif

#if 1
	{
		std::cout << "Opening full image " << fullImgFiles[0].string()
				<< std::endl;
		//cv::Mat imageMat = cv::imread(fullImgFiles[0].string().c_str(), 1);
		cv::Mat imageMat = cv::imread("../samples/prom.jpg", 1);

		for (unsigned int j = 0; j < 1000; j++) {
			std::vector<float> descriptor;
			int x = std::rand() % (imageMat.cols - hog.winSize.width);
			int y = std::rand() % (imageMat.rows - hog.winSize.height);
			cv::Rect roi = cv::Rect(cv::Point(x, y), hog.winSize);
			std::cout << "Sampling random patch from " << roi
					<< std::endl;
			hog.compute(imageMat(roi), descriptor);
			float prediction = svm.predict(cv::Mat(descriptor).t());
			if (prediction > 0.5) {
				std::cout << "detection!" << std::endl;
				cv::rectangle(imageMat, roi, cv::Scalar(64, 255, 64),
						3);
			}
			else
			{
				cv::rectangle(imageMat, roi, cv::Scalar(0, 0, 0),
										1);
			}
		}
		cv::imwrite("detection.png", imageMat);

	}
#endif
	return 0;
}

}
