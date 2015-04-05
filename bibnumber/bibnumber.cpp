#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>
#include <string.h>

#include "batch.h"
#include "train.h"

using namespace std;

static void help() {
	cout << "\nThis program extracts bib numbers from images.\n"
			"Usage:\n"
			"./bibnumber [-train dir] [-model svmModel.xml] image_file|folder_path|csv_ground_truth_file\n\n"
			<< endl;
}

int main(int argc, const char** argv) {
	string inputName;
	string trainDir;
	string svmModel;
	int train = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i],"-train"))
		{
			if ( (i>=(argc-1)) )
			{
				cerr << "ERROR: missing parameter for -train" << endl;
				help();
				return -1;
			}
			train = 1;
			trainDir.assign(argv[++i]);
		}
		else if (!strcmp(argv[i],"-model"))
		{
			if ( (i>=(argc-1)) )
			{
				cerr << "ERROR: missing parameter for -model" << endl;
				help();
				return -1;
			}
			svmModel.assign(argv[++i]);
		}
		else
		{
			inputName.assign(argv[i]);
		}
	}

	if ((inputName.empty()) || (!inputName.size())) {
		cerr << "ERROR: Missing parameter" << endl;
		help();
		return -1;
	}

	if (train)
	{
		train::process(trainDir, inputName);
	}
	else
	{
		batch::process(inputName, svmModel);
	}

	return 0;
}
