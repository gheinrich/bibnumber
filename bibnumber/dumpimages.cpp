
#include "dumpimages.h"
namespace fs = boost::filesystem;

namespace dumpimages {

    bool dumpFiles::dumpImages(std::string bibNumber, fs::path imageName)
    {
        fs::path pathname(imageName);
        fs::path dirname = pathname.parent_path();
        fs::path file(imageName);
        fs::path full_path = dirname / file;
        std::string bibPath = dirname.string() + "/"+ bibNumber;
        if(!fs::exists(bibPath))
        {
            if(boost::filesystem::create_directory(bibPath))
            {
                std::cout << "Folder successfully created: " + bibNumber << std::endl;
            }
        }


        fs::path bibPathDir(bibPath);
        try {
            fs::copy_file(pathname, bibPathDir / pathname.filename());
        }
        catch(fs::filesystem_error const & e)
        {
            std:: cerr << e.what() << '\n';
            return false;
        }
        return true;
    }
} /* namespace dumpimages*/