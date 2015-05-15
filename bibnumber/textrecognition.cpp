#include <boost/algorithm/string/trim.hpp>

#include <tesseract/baseapi.h>
#include <tesseract/strngs.h>
#include <tesseract/genericvector.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/ml/ml.hpp>

#include "train.h"

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

static inline double square(double x) {
	return x * x;
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

cv::Scalar getMSSIM(const cv::Mat& i1, const cv::Mat& i2) {
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	cv::Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	cv::Mat I2_2 = I2.mul(I2);        // I2^2
	cv::Mat I1_2 = I1.mul(I1);        // I1^2
	cv::Mat I1_I2 = I1.mul(I2);        // I1 * I2

	/***********************PRELIMINARY COMPUTING ******************************/

	cv::Mat mu1, mu2;   //
	cv::GaussianBlur(I1, mu1, cv::Size(11, 11), 1.5);
	cv::GaussianBlur(I2, mu2, cv::Size(11, 11), 1.5);

	cv::Mat mu1_2 = mu1.mul(mu1);
	cv::Mat mu2_2 = mu2.mul(mu2);
	cv::Mat mu1_mu2 = mu1.mul(mu2);

	cv::Mat sigma1_2, sigma2_2, sigma12;

	cv::GaussianBlur(I1_2, sigma1_2, cv::Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	cv::GaussianBlur(I2_2, sigma2_2, cv::Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	cv::GaussianBlur(I1_I2, sigma12, cv::Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	cv::Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);   // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	cv::Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	cv::Scalar mssim = mean(ssim_map); // mssim = average of ssim map
	return mssim;
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
		const struct TextDetectionParams &params, std::string svmModel,
		std::vector<Chain> &chains,
		std::vector<std::pair<Point2d, Point2d> > &compBB,
		std::vector<std::pair<CvPoint, CvPoint> > &chainBB,
		std::vector<std::string>& text) {

	// Convert to grayscale
	IplImage * grayImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCvtColor(input, grayImage, CV_RGB2GRAY);

	for (unsigned int i = 0; i < chainBB.size(); i++) {
		cv::Point center = cv::Point(
				(chainBB[i].first.x + chainBB[i].second.x) / 2,
				(chainBB[i].first.y + chainBB[i].second.y) / 2);

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

		/* resize image to improve OCR success rate */
		float upscale = 3.0;
		cv::resize(mat, mat, cvSize(0, 0), upscale, upscale);
		/* erode text to get rid of thin joints */
		int s = (int) (0.05 * mat.rows); /* 5% of up-scaled size) */
		cv::Mat elem = cv::getStructuringElement(cv::MORPH_ELLIPSE,
				cv::Size(2 * s + 1, 2 * s + 1), cv::Point(s, s));
		cv::erode(mat, mat, elem);
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

			/* adjust width to size of 6 digits */
			int charWidth = (chainBB[i].second.x - chainBB[i].first.x)
					/ s_out.size();
			int width = 6 * charWidth;
			/* adjust to 2 width/height aspect ratio */
			int height = width / 2;
			int midx = center.x;
			int midy = center.y;

			cv::Rect roi = cv::Rect(midx - width / 2, midy - height / 2, width,
					height);
			if ((roi.x >= 0) && (roi.y >= 0)
					&& (roi.x + roi.width < inputMat.cols)
					&& (roi.y + roi.height < inputMat.rows)) {
				cv::Mat bibMat = inputMat(roi);

				if (s_out.size() <= (unsigned) params.modelVerifLenCrit) {

					if (svmModel.empty()) {
						LOGL(LOG_TEXTREC, "Reject " << s_out << " on no model");
						break;
					}

					if (minHeight < params.modelVerifMinHeight) {
						LOGL(LOG_TEXTREC,
								"Reject " << s_out << " on small height");
						break;
					}

					/* if we have an SVM Model, predict */

					CvSVM svm;
					cv::HOGDescriptor hog(cv::Size(128, 64), /* windows size */
					cv::Size(16, 16), /* block size */
					cv::Size(8, 8), /* block stride */
					cv::Size(8, 8), /* cell size */
					9 /* nbins */
					);
					std::vector<float> descriptor;

					/* resize to HOGDescriptor dimensions */
					cv::Mat resizedMat;
					cv::resize(bibMat, resizedMat, hog.winSize, 0, 0);
					hog.compute(resizedMat, descriptor);

					/* load SVM model */
					svm.load(svmModel.c_str());
					float prediction = svm.predict(cv::Mat(descriptor).t());
					LOGL(LOG_SVM, "Prediction=" << prediction);
					if (prediction < 0.5) {
						LOGL(LOG_TEXTREC,
								"Reject " << s_out << " on low SVM prediction");
						break;
					}
				}

				/* symmetry check */
				if (   //(i == 4) &&
						(1)) {
					cv::Mat inputRotated = cv::Mat::zeros(inputMat.rows,
							inputMat.cols, inputMat.type());
					cv::warpAffine(inputMat, inputRotated, rotMatrix,
							inputRotated.size());

					int minOffset = 0;
					double min = 1e6;
					//width = 12 * charWidth;
					for (int offset = -50; offset < 30; offset += 2) {

						/* resize to HOGDescriptor dimensions */
						cv::Mat straightMat;
						cv::Mat flippedMat;

						/* extract shifted ROI */
						cv::Rect roi = cv::Rect(midx - width / 2 + offset,
								midy - height / 2, width, height);

						if ((roi.x >= 0) && (roi.y >= 0)
								&& (roi.x + roi.width < inputMat.cols)
								&& (roi.y + roi.height < inputMat.rows)) {
							straightMat = inputRotated(roi);
							cv::flip(straightMat, flippedMat, 1);
							cv::Scalar mssimV = getMSSIM(straightMat,
									flippedMat);
							double avgMssim = (mssimV.val[0] + mssimV.val[1]
									+ mssimV.val[2]) * 100 / 3;
							double dist = 1 / (avgMssim + 1);
							LOGL(LOG_SYMM_CHECK, "offset=" << offset << " dist=" << dist);
							if (dist < min) {
								min = dist;
								minOffset = offset;
								cv::imwrite("symm-max.png", straightMat);
								cv::Mat visualImage;
							}
						}
					}

					LOGL(LOG_SYMM_CHECK, "MinOffset = " << minOffset
							<< " charWidth=" << charWidth);

					if (absd(minOffset) > charWidth / 3) {
						LOGL(LOG_TEXTREC,
								"Reject " << s_out << " on asymmetry");
						std::cout << "Reject " << s_out << " on asymmetry"
								<< std::endl;
						break;
					}
				}

				/* save for training only if orientation is ~horizontal */
				if (abs(theta_deg) < 7) {
					char *filename;
					asprintf(&filename, "bib-%05d-%04d.png", this->bsid++,
							atoi(out));
					cv::imwrite(filename, bibMat);
					free(filename);
				}

			} else {
				LOGL(LOG_TEXTREC, "Reject as ROI outside boundaries");
				break;
			}

			/* all fine, add this bib number */
			text.push_back(s_out);
			LOGL(LOG_TEXTREC, "Bib number: '" << s_out << "'");

		} while (0);
		free(out);
	}

	cvReleaseImage(&grayImage);

	return 0;

}

} /* namespace textrecognition */

