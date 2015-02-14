#include <cctype>
#include <iostream>
#include <iterator>
#include <stdio.h>

#include "batch.h"

using namespace std;

static void help() {
	cout << "\nThis program extracts bib numbers from images.\n"
			"Usage:\n"
			"./bibnumber [image file|folder path|csv ground truth file]\n\n"
			<< endl;
}

int main(int argc, const char** argv) {
	string inputName;

	for (int i = 1; i < argc; i++) {
		inputName.assign(argv[i]);
	}

	if ((inputName.empty()) || (!inputName.size())) {
		cerr << "ERROR: Missing parameter" << endl;
		help();
		return -1;
	}

	batch::process(inputName);

	return 0;
}
