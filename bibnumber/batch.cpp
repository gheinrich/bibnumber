#include <iterator>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/bimap.hpp>
#include <boost/bimap/set_of.hpp>
#include <boost/bimap/multiset_of.hpp>

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "batch.h"
#include "pipeline.h"
#include "log.h"

namespace bimaps = boost::bimaps;
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

static int processSingleImage(
		std::string fileName,
		std::string svmModel,
		pipeline::Pipeline &pipeline,
		std::vector<int>& bibNumbers)
{
	int res;

	std::cout << "Processing file " << fileName << std::endl;

	/* open image */
	cv::Mat image = cv::imread(fileName, 1);
	if (image.empty()) {
		std::cerr << "ERROR:Failed to open image file" << std::endl;
		return -1;
	}

	/* process image */
	res = pipeline.processImage(image, svmModel, bibNumbers);
	if (res < 0) {
		std::cerr << "ERROR: Could not process image" << std::endl;
		return -1;
	}

	/* remove duplicates */
	std::sort(bibNumbers.begin(), bibNumbers.end());
	bibNumbers.erase(std::unique(bibNumbers.begin(), bibNumbers.end()),
			bibNumbers.end());

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

bool isImageFile(std::string name) {
	std::string lower_case(name);
	std::transform(lower_case.begin(), lower_case.end(), lower_case.begin(),
			::tolower);

	if ((boost::algorithm::ends_with(lower_case, ".jpg"))
			|| (boost::algorithm::ends_with(lower_case, ".png")))
		return true;
	else
		return false;

}

std::vector<fs::path> getImageFiles(std::string dir)
{
	std::vector<fs::path> paths; // vector of paths in directory
	std::copy(fs::directory_iterator(dir), fs::directory_iterator(),
				back_inserter(paths));
	// sort, since directory iteration
	// is not ordered on some file systems
	sort(paths.begin(), paths.end());

	std::vector<fs::path> imgFiles;;
	for (std::vector<fs::path>::const_iterator it(paths.begin());
				it != paths.end(); ++it) {
		if (batch::isImageFile(it->string())) {
			imgFiles.push_back(*it);
		}
	}
	return imgFiles;
}

int process(std::string inputName, std::string svmModel) {
	int res;

	std::string resultFileName("out.csv");

	if (!fs::exists(inputName)) {
		std::cerr << "ERROR: Not found: " << inputName << std::endl;
		return -1;
	}

	pipeline::Pipeline pipeline;

	if (fs::is_regular_file(inputName)) {
		/* convert name to lower case to make extension checks easier */
		std::string name(inputName);
		std::transform(name.begin(), name.end(), name.begin(), ::tolower);

		if (isImageFile(inputName)) {
			std::vector<int> bibNumbers;
			res = processSingleImage(inputName, svmModel, pipeline, bibNumbers);
		} else if (boost::algorithm::ends_with(name, ".csv")) {

			int true_positives = 0;
			int false_positives = 0;
			int relevant = 0;

			/* set log mask to minimum */
			biblog::set_log_mask(LOG_NONE);

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

				processSingleImage(full_path.string(), svmModel, pipeline, bibNumbers);

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
						std::cout << "Missed " << groundTruthNumbers[i]
								<< std::endl;
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

		fs::path outPath = inputName / fs::path(resultFileName);
		std::cout << "Processing directory " << inputName << " into "
				<< outPath.string() << std::endl;

		std::ofstream outFile;
		outFile.open(outPath.c_str());

		/* set log mask to minimum */
		biblog::set_log_mask(LOG_NONE);

		std::vector<fs::path> img_paths; // vector of image paths

		typedef boost::bimap<bimaps::multiset_of<std::string>,
				bimaps::multiset_of<int> > imgTagBimap;

		imgTagBimap tags;

		/* find images in directory */
		img_paths = getImageFiles(inputName);

		/* process images */
		for (int i = 0, j=img_paths.size(); i<j ; i++) {
			std::vector<int> bibNumbers;

			std::cout << std::endl << "[" << i+1 << "/" << j << "] ";
			res = processSingleImage(img_paths[i].string(), svmModel, pipeline, bibNumbers);

			for (unsigned int k = 0; k < bibNumbers.size(); k++) {
				tags.insert(
						imgTagBimap::value_type(img_paths[i].string(), bibNumbers[k]));
			}
		}

		/* save results to .csv file */
		std::cout << "Saving results to " << outPath.string() << std::endl;
		int current_bib = 0;
		for (imgTagBimap::right_const_iterator it = tags.right.begin(), iend =
				tags.right.end(); it != iend; it++) {
			if (it->first != current_bib) {
				current_bib = it->first;
				outFile << std::endl << current_bib << ",";
			}
			outFile << it->second << ",";
		}
		outFile.close();

		return -1;
	} else {
		std::cerr << "ERROR: unknown path type " << inputName << std::endl;
		return -1;
	}

	return res;
}

} /* namespace batch */

