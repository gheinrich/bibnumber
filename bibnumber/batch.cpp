#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string/predicate.hpp>

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

namespace batch {

int process(std::string inputName) {
	cv::Mat image = cv::imread(inputName, 1);
	if (image.empty()) {
		std::cerr << "ERROR:Failed to open image file" << std::endl;
		return -1;
	}

	//detectAndDraw(image, cascade, nestedCascade, scale, tryflip);
	pipeline::processImage(image);
	return 0;
}

} /* namespace batch */

