#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "batch.h"
#include "pipeline.h"

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
		while (std::getline(lineStream, cell, ',')) {
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

int processSingleImage(std::string fileName) {
	int res;

	std::cout << "Processing file " << fileName << std::endl;

	std::vector<std::string> bibNumbers;
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

	/* display result */
	for (std::vector<std::string>::iterator it = bibNumbers.begin();
			it != bibNumbers.end(); ++it) {
		std::string s = *it;
		boost::algorithm::trim(s);
		if (s.size() > 0) {
			std::cout << "Read: " << s << std::endl;
		}
	}
}

namespace batch {

int process(std::string inputName) {
	int res;

	if ( (boost::algorithm::ends_with(inputName,".jpg"))
			|| (boost::algorithm::ends_with(inputName,".png")) )
	{
		res = processSingleImage(inputName);
	}

	return res;
}

} /* namespace batch */

