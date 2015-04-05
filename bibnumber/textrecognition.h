#ifndef TEXTREC_H
#define TEXTREC_H

#include <tesseract/baseapi.h>

#include "opencv2/imgproc/imgproc.hpp"

#include "textdetection.h"

namespace textrecognition
{
	class TextRecognizer {
	public:
		TextRecognizer(void);
		~TextRecognizer(void);
		int recognize (IplImage *input,
	   	               const struct TextDetectionParams &params,
	   	               std::string svmModel,
		               std::vector<Chain> &chains,
			           std::vector<std::pair<Point2d, Point2d> > &compBB,
			           std::vector<std::pair<CvPoint, CvPoint> > &chainBB,
			           std::vector<std::string>& text);
	private:
		tesseract::TessBaseAPI tess;
		int dsid; /* digit sequence id */
		int bsid; /* bib sequence id */
	};

}

#endif /* #ifndef TEXTREC_H */

