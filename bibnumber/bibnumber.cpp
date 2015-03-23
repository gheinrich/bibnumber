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
			"./bibnumber -train dir image_file|folder_path|csv_ground_truth_file\n\n"
			<< endl;
}

int main(int argc, const char** argv) {
	string inputName;
	string trainDir;
	int train = 0;

	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i],"-train"))
		{
			if ( (i!=1) || (argc<3) )
			{
				cerr << "ERROR: invalid parameters for -train" << endl;
				help();
				return -1;
			}
			train = 1;
			trainDir.assign(argv[++i]);
		}
		else
			inputName.assign(argv[i]);
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
		batch::process(inputName);
	}

	return 0;
}
