#ifndef BATCH_H
#define BATCH_H

#include <string>
#include <boost/filesystem.hpp>

namespace batch
{
	bool isImageFile(std::string name);
	std::vector<boost::filesystem::path> getImageFiles(std::string dir);
	int process(std::string inputName, std::string svmModel);
}

#endif /* #ifndef BATCH_H */
