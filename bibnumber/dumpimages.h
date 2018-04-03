
#ifndef BIBNUMBER_DUMPIMAGES_H
#define BIBNUMBER_DUMPIMAGES_H
#include <iostream>
#include <string>
#include <boost/filesystem.hpp>

namespace dumpimages
{
    class dumpFiles {
    public:
        bool dumpImages(std::string bibNumber, boost::filesystem::path imageName);
        std::string NOT_FOUND;

        dumpFiles(){
            NOT_FOUND = "NOT_FOUND";
        }
    };
}
#endif //BIBNUMBER_DUMPIMAGES_H
