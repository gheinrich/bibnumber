#include <boost/algorithm/string/trim.hpp>

#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <tesseract/genericvector.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>

#include "textrecognition.h"
#include "log.h"
#include "stdio.h"

#define PI 3.14159265

static bool is_number(const std::string& s) {
	std::string::const_iterator it = s.begin();
	while (it != s.end() && std::isdigit(*it))
		++it;
	return !s.empty() && it == s.end();
}

static double absd(double x) {
	return x > 0 ? x : -x;
}

static cv::Rect getBoundingBox(std::vector<cv::Point> vec, cv::Size clip) {
	int minx = clip.width - 1, miny = clip.height - 1, maxx = 0, maxy = 0;
	for (std::vector<cv::Point>::iterator it = vec.begin(); it != vec.end();
			it++) {
		if (it->x < minx)
			minx = std::max(it->x, 0);
		if (it->y < miny)
			miny = std::max(it->y, 0);
		if (it->x > maxx)
			maxx = std::min(it->x, clip.width - 1);
		if (it->y > maxy)
			maxy = std::min(it->y, clip.height - 1);
	}
	return cv::Rect(cv::Point(minx, miny), cv::Point(maxx, maxy));
}

namespace textrecognition {

TextRecognizer::TextRecognizer() {
	GenericVector<STRING> pars_keys;
	GenericVector<STRING> pars_vals;
	/*pars_keys.push_back("load_system_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_freq_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_punc_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_number_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_unambig_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_bigram_dawg");
	 pars_vals.push_back("F");
	 pars_keys.push_back("load_fixed_length_dawgs");
	 pars_vals.push_back("F");*/
	tess.Init(NULL, "eng", tesseract::OEM_DEFAULT, NULL, 0, &pars_keys,
			&pars_vals, false);
#if 0
	tess.SetVariable("tessedit_char_whitelist", "0123456789");
#endif
	tess.SetVariable("tessedit_write_images", "true");
	tess.SetPageSegMode(tesseract::PSM_SINGLE_WORD);

	/* initialize sequence ids */
	bsid = 0;
	dsid = 0;
}

TextRecognizer::~TextRecognizer(void) {
	tess.Clear();
	tess.End();
}

int TextRecognizer::recognize(IplImage *input,
		const struct TextDetectionParams &params, std::vector<Chain> &chains,
		std::vector<std::pair<Point2d, Point2d> > &compBB,
		std::vector<std::pair<CvPoint, CvPoint> > &chainBB,
		std::vector<std::string>& text) {

	// Convert to grayscale
	IplImage * grayImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCvtColor(input, grayImage, CV_RGB2GRAY);

	for (unsigned int i = 0; i < chainBB.size(); i++) {
		cv::Point center = cv::Point(
				(chainBB[i].first.x + chainBB[i].second.x / 2),
				(chainBB[i].first.y + chainBB[i].second.y / 2));

		/* work out if total width of chain is large enough */
		if (chainBB[i].second.x - chainBB[i].first.x
				< input->width / params.maxImgWidthToTextRatio) {
			LOGL(LOG_TXT_ORIENT,
					"Reject chain #" << i << " width=" << (chainBB[i].second.x - chainBB[i].first.x) << "<" << (input->width / params.maxImgWidthToTextRatio));
			continue;
		}

		/* eliminate chains with components of lower height than required minimum */
		int minHeight = chainBB[i].second.y - chainBB[i].first.y;
		for (unsigned j = 0; j < chains[i].components.size(); j++) {
			minHeight = std::min(minHeight,
					compBB[chains[i].components[j]].second.y
							- compBB[chains[i].components[j]].first.y);
		}
		if (minHeight < params.minCharacterheight) {
			LOGL(LOG_CHAINS,
					"Reject chain # " << i << " minHeight=" << minHeight << "<" << params.minCharacterheight);
			continue;
		}

		/* invert direction if angle is in 3rd/4th quadrants */
		if (chains[i].direction.x < 0) {
			chains[i].direction.x = -chains[i].direction.x;
			chains[i].direction.y = -chains[i].direction.y;
		}
		/* work out chain angle */
		double theta_deg = 180
				* atan2(chains[i].direction.y, chains[i].direction.x) / PI;

		if (absd(theta_deg) > params.maxAngle) {
			LOGL(LOG_TXT_ORIENT,
					"Chain angle " << theta_deg << " exceeds max " << params.maxAngle);
			continue;
		}
		if ((chainBB.size() == 2) && (absd(theta_deg) > 5))
			continue;
		LOGL(LOG_TXT_ORIENT,
				"Chain #" << i << " Angle: " << theta_deg << " degrees");

		/* create copy of input image including only the selected components */
		cv::Mat inputMat = cv::Mat(input);
		cv::Mat grayMat = cv::Mat(grayImage);
		cv::Mat componentsImg = cv::Mat::zeros(grayMat.rows, grayMat.cols,
				grayMat.type());

		std::vector<cv::Point> compCoords;

		for (unsigned int j = 0; j < chains[i].components.size(); j++) {
			int component_id = chains[i].components[j];
			cv::Rect roi = cv::Rect(compBB[component_id].first.x,
					compBB[component_id].first.y,
					compBB[component_id].second.x
							- compBB[component_id].first.x,
					compBB[component_id].second.y
							- compBB[component_id].first.y);
			cv::Mat componentRoi = grayMat(roi);

			compCoords.push_back(
					cv::Point(compBB[component_id].first.x,
							compBB[component_id].first.y));
			compCoords.push_back(
					cv::Point(compBB[component_id].second.x,
							compBB[component_id].second.y));
			compCoords.push_back(
					cv::Point(compBB[component_id].first.x,
							compBB[component_id].second.y));
			compCoords.push_back(
					cv::Point(compBB[component_id].second.x,
							compBB[component_id].first.y));

			cv::Mat thresholded;
			cv::threshold(componentRoi, thresholded, 0 // the value doesn't matter for Otsu thresholding
					, 255 // we could choose any non-zero value. 255 (white) makes it easy to see the binary image
					, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);

#if 0
			cv::Moments mu = cv::moments(thresholded, true);
			std::cout << "mu02=" << mu.mu02 << " mu11=" << mu.mu11 << " skew="
					<< mu.mu11 / mu.mu02 << std::endl;
#endif
			cv::imwrite("thresholded.png", thresholded);

			cv::threshold(componentRoi, componentsImg(roi), 0 // the value doesn't matter for Otsu thresholding
					, 255 // we could choose any non-zero value. 255 (white) makes it easy to see the binary image
					, cv::THRESH_OTSU | cv::THRESH_BINARY_INV);
		}
		cv::imwrite("bib-components.png", componentsImg);

		cv::Mat rotMatrix = cv::getRotationMatrix2D(center, theta_deg, 1.0);

		cv::Mat rotatedMat = cv::Mat::zeros(grayMat.rows, grayMat.cols,
				grayMat.type());
		cv::warpAffine(componentsImg, rotatedMat, rotMatrix, rotatedMat.size());
		cv::imwrite("bib-rotated.png", rotatedMat);

		/* rotate each component coordinates */
		const int border = 3;
		cv::transform(compCoords, compCoords, rotMatrix);
		/* find bounding box of rotated components */
		cv::Rect roi = getBoundingBox(compCoords,
				cv::Size(input->width, input->height));
		/* ROI area can be null if outside of clipping area */
		if ((roi.width == 0) || (roi.height == 0))
			continue;
		LOGL(LOG_TEXTREC, "ROI = " << roi);
		cv::Mat mat = cv::Mat::zeros(roi.height + 2 * border,
				roi.width + 2 * border, grayMat.type());
		cv::Mat tmp = rotatedMat(roi);
		/* copy bounded box from rotated mat to new mat with borders - borders are needed
		 * to improve OCR success rate
		 */
		tmp.copyTo(
				mat(
						cv::Rect(cv::Point(border, border),
								cv::Point(roi.width + border,
										roi.height + border))));

#if 1
		/* resize image to improve OCR success rate */
		float upscale = 3.0;
		cv::resize(mat, mat, cvSize(0, 0), upscale, upscale);
		/* erode text to get rid of thin joints */
		int s = (int) (0.05 * mat.rows); /* 5% of up-scaled size) */
		cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE,
				cv::Size(2 * s + 1, 2 * s + 1), cv::Point(s, s));
		cv::erode(mat, mat, elem);
#endif
		cv::imwrite("bib-tess-input.png", mat);

		// Pass it to Tesseract API
		tess.SetImage((uchar*) mat.data, mat.cols, mat.rows, 1, mat.step1());
		// Get the text
		char* out = tess.GetUTF8Text();
		do {
			if (strlen(out) == 0) {
				break;
			}
			std::string s_out(out);
			boost::algorithm::trim(s_out);

			if (s_out.size() != chains[i].components.size()) {
				LOGL(LOG_TEXTREC,
						"Text size mismatch: expected " << chains[i].components.size() << " digits, got '" << s_out << "' (" << s_out.size() << " digits)");
				break;
			}
			/* if first character is a '0' we have a partially occluded number */
			if (s_out[0] == '0') {
				LOGL(LOG_TEXTREC, "Text begins with '0' (partially occluded)");
				break;
			}
			if (!is_number(s_out)) {
				LOGL(LOG_TEXTREC, "Text is not a number ('" << s_out << "')");
				break;
			}
			text.push_back(s_out);
			LOGL(LOG_TEXTREC, "Mat text: " << s_out);

#if 0
			/* save all individual digits for subsequent learning */
			for (unsigned int j = 0; j < chains[i].components.size(); j++) {
				int component_id = chains[i].components[j];
				/* enforce 3x height/width aspect ratio */
				int midy = (compBB[component_id].first.y
						+ compBB[component_id].second.y) / 2;
				int width = compBB[component_id].second.x
						- compBB[component_id].first.x;
				cv::Rect roi = cv::Rect(compBB[component_id].first.x,
						midy - 3 * width / 2, width, 3 * width);
				cv::Mat digitMat = grayMat(roi);
				char *filename;
				asprintf(&filename, "digit-%c-%04d.png", out[j], this->dsid++);
				cv::imwrite(filename, digitMat);
				free(filename);
			}
#endif

			/* save whole bib image */

			/* only if orientation is ~horizontal */
			if (abs(theta_deg) < 6) {
				/* adjust width to size of 6 digits */
				int width = 6 * (chainBB[i].second.x - chainBB[i].first.x)
						/ s_out.size();
				/* adjust to 2 width/height aspect ratio */
				int height = width / 2;
				int midx = (chainBB[i].first.x + chainBB[i].second.x) / 2;
				int midy = (chainBB[i].first.y + chainBB[i].second.y) / 2;
				cv::Rect roi = cv::Rect(midx - width/2, midy - height/2,
						width, height);
				if ( (roi.x>=0) && (roi.y>=0) && (roi.x+roi.width<inputMat.cols)
						&& (roi.y+roi.height<inputMat.rows) )
				{
					cv::Mat bibMat = inputMat(roi);
					char *filename;
					asprintf(&filename, "bib-%05d-%04d.png",
							this->bsid++, atoi(out));
					cv::imwrite(filename, bibMat);
					free(filename);
				}
			}

		} while (0);
		free(out);
	}

	cvReleaseImage(&grayImage);

	return 0;

}

} /* namespace textrecognition */

