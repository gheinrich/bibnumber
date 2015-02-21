#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "batch.h"
#include "pipeline.h"
#include "debug.h"

namespace fs = boost::filesystem;

class CSVRow {
public:
	std::string const& operator[](std::size_t index) const {
		return m_data[index];
	}
	std::size_t size() const {
		return m_data.size();
	}
	void readNextRow(std::istream& str) {
		std::string line;
		std::getline(str, line);

		std::stringstream lineStream(line);
		std::string cell;

		m_data.clear();
		while (std::getline(lineStream, cell, ';')) {
			m_data.push_back(cell);
		}
	}
private:
	std::vector<std::string> m_data;
};

std::istream& operator>>(std::istream& str, CSVRow& data) {
	data.readNextRow(str);
	return str;
}

#if 0
int batch(const char *path) {
	if (boost::algorithm::ends_with(path, ".csv")) {
		std::ifstream file(path);

		CSVRow row;
		while (file >> row) {
			std::cout << "4th Element(" << row[3] << ")\n";
		}
	}

	return 0;
}
#endif

static int processSingleImage(std::string fileName,
		std::vector<int>& bibNumbers) {
	int res;

	std::cout << "Processing file " << fileName << std::endl;

	/* open image */
	cv::Mat image = cv::imread(fileName, 1);
	if (image.empty()) {
		std::cerr << "ERROR:Failed to open image file" << std::endl;
		return -1;
	}

	/* process image */
	res = pipeline::processImage(image, bibNumbers);
	if (res < 0) {
		std::cerr << "ERROR: Could not process image" << std::endl;
		return -1;
	}

	/* remove duplicates */
	std::sort( bibNumbers.begin(), bibNumbers.end() );
	bibNumbers.erase( std::unique( bibNumbers.begin(), bibNumbers.end() ),
							bibNumbers.end() );

	/* display result */
	std::cout << "Read: [";
	for (std::vector<int>::iterator it = bibNumbers.begin();
			it != bibNumbers.end(); ++it) {
		std::cout << " " << *it;
	}
	std::cout << "]" << std::endl;

	return res;
}

static int exists(std::vector<int> arr, int item) {
	return std::find(arr.begin(), arr.end(), item) != arr.end();
}

namespace batch {

int process(std::string inputName) {
	int res;

	if (!fs::exists(inputName)) {
		std::cerr << "ERROR: Not found: " << inputName << std::endl;
		return -1;
	}

	if (fs::is_regular_file(inputName)) {
		if ((boost::algorithm::ends_with(inputName, ".jpg"))
				|| (boost::algorithm::ends_with(inputName, ".png"))) {
			std::vector<int> bibNumbers;
			res = processSingleImage(inputName, bibNumbers);
		} else if (boost::algorithm::ends_with(inputName, ".csv")) {

			int true_positives = 0;
			int false_positives = 0;
			int relevant = 0;

			/* set debug mask to minimum */
			debug::set_debug_mask(DBG_NONE);

			std::ifstream file(inputName.c_str());
			fs::path pathname(inputName);
			fs::path dirname = pathname.parent_path();

			CSVRow row;
			while (file >> row) {
				std::string filename = row[0];
				std::vector<int> groundTruthNumbers;
				std::vector<int> bibNumbers;

				fs::path file(filename);
				fs::path full_path = dirname / file;

				processSingleImage(full_path.string(), bibNumbers);

				for (unsigned int i = 1; i < row.size(); i++)
					groundTruthNumbers.push_back(atoi(row[i].c_str()));
				relevant += groundTruthNumbers.size();

				for (unsigned int i = 0; i < bibNumbers.size(); i++) {
					if (exists(groundTruthNumbers, bibNumbers[i])) {
						std::cout << "Match " << bibNumbers[i] << std::endl;
						true_positives++;
					} else {
						std::cout << "Mismatch " << bibNumbers[i] << std::endl;
						false_positives++;
					}
				}

				for (unsigned int i = 0; i < groundTruthNumbers.size(); i++) {
					if (!exists(bibNumbers, groundTruthNumbers[i])) {
						std::cout << "Missed " << groundTruthNumbers[i] << std::endl;
					}
				}

			}

			std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
			std::cout.precision(2);

			float precision = (float) true_positives
					/ (float) (true_positives + false_positives);
			float recall = (float) true_positives / (float) (relevant);
			float fscore = 2 * precision * recall / (precision + recall);

			std::cout << "precision=" << true_positives << "/"
					<< true_positives + false_positives << "=" << precision
					<< std::endl;
			std::cout << "recall=" << true_positives << "/" << relevant << "="
					<< recall << std::endl;
			std::cout << "F-score=" << fscore << std::endl;

		}
	} else if (fs::is_directory(inputName)) {
		std::cout << "Processing directory " << inputName << std::endl;
		return -1;
	} else {
		std::cerr << "ERROR: unknown path type " << inputName << std::endl;
		return -1;
	}

	return res;
}

} /* namespace batch */

