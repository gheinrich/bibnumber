/*
 Copyright 2012 Andrew Perrault and Saurav Kumar.

 This file is part of DetectText.

 DetectText is free software: you can redistribute it and/or modify
 it under the terms of the GNU General Public License as published by
 the Free Software Foundation, either version 3 of the License, or
 (at your option) any later version.

 DetectText is distributed in the hope that it will be useful,
 but WITHOUT ANY WARRANTY; without even the implied warranty of
 MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 GNU General Public License for more details.

 You should have received a copy of the GNU General Public License
 along with DetectText.  If not, see <http://www.gnu.org/licenses/>.
 */
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/graph_traits.hpp>
#include <boost/graph/connected_components.hpp>
#include <boost/unordered_map.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/algorithm/string/trim.hpp>
#include <cassert>
#include <cmath>
#include <iostream>
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/cxcore.h>
#include <math.h>
#include <time.h>
#include <utility>
#include <algorithm>
#include <vector>
#include "textdetection.h"

#include "log.h"

#define PI 3.14159265

#define COM_MAX_MEDIAN_RATIO (3.0)
#define COM_MAX_DIM_RATIO (2.0)
#define COM_MAX_DIST_RATIO (2.0)
#define COM_MAX_ASPECT_RATIO (2.5)

static inline int square(int x) {
	return x * x;
}

static int inline ratio_within(float ratio, float max_ratio) {
	return ((ratio < max_ratio) && (ratio > 1 / max_ratio));
}

std::vector<std::pair<CvPoint, CvPoint> > findBoundingBoxes(
		std::vector<Chain> & chains,
		std::vector<std::pair<Point2d, Point2d> > & compBB, IplImage * output) {
	std::vector<std::pair<CvPoint, CvPoint> > bb;
	bb.reserve(chains.size());
	for (std::vector<Chain>::iterator chainit = chains.begin();
			chainit != chains.end(); chainit++) {
		int minx = output->width;
		int miny = output->height;
		int maxx = 0;
		int maxy = 0;
		for (std::vector<int>::const_iterator cit = chainit->components.begin();
				cit != chainit->components.end(); cit++) {
			miny = std::min(miny, compBB[*cit].first.y);
			minx = std::min(minx, compBB[*cit].first.x);
			maxy = std::max(maxy, compBB[*cit].second.y);
			maxx = std::max(maxx, compBB[*cit].second.x);
		}
		CvPoint p0 = cvPoint(minx, miny);
		CvPoint p1 = cvPoint(maxx, maxy);
		std::pair<CvPoint, CvPoint> pair(p0, p1);
		bb.push_back(pair);
	}
	return bb;
}

std::vector<std::pair<CvPoint, CvPoint> > findBoundingBoxes(
		std::vector<std::vector<Point2d> > & components, IplImage * output) {
	std::vector<std::pair<CvPoint, CvPoint> > bb;
	bb.reserve(components.size());
	for (std::vector<std::vector<Point2d> >::iterator compit =
			components.begin(); compit != components.end(); compit++) {
		int minx = output->width;
		int miny = output->height;
		int maxx = 0;
		int maxy = 0;
		for (std::vector<Point2d>::iterator it = compit->begin();
				it != compit->end(); it++) {
			miny = std::min(miny, it->y);
			minx = std::min(minx, it->x);
			maxy = std::max(maxy, it->y);
			maxx = std::max(maxx, it->x);
		}
		CvPoint p0 = cvPoint(minx, miny);
		CvPoint p1 = cvPoint(maxx, maxy);
		std::pair<CvPoint, CvPoint> pair(p0, p1);
		bb.push_back(pair);
	}
	return bb;
}

void normalizeImage(IplImage * input, IplImage * output) {
	assert(input->depth == IPL_DEPTH_32F);
	assert(input->nChannels == 1);
	assert(output->depth == IPL_DEPTH_32F);
	assert(output->nChannels == 1);
	float maxVal = 0;
	float minVal = 1e100;
	for (int row = 0; row < input->height; row++) {
		const float* ptr = (const float*) (input->imageData
				+ row * input->widthStep);
		for (int col = 0; col < input->width; col++) {
			if (*ptr < 0) {
			} else {
				maxVal = std::max(*ptr, maxVal);
				minVal = std::min(*ptr, minVal);
			}
			ptr++;
		}
	}

	float difference = maxVal - minVal;
	for (int row = 0; row < input->height; row++) {
		const float* ptrin = (const float*) (input->imageData
				+ row * input->widthStep);\
		float* ptrout = (float*) (output->imageData + row * output->widthStep);\
		for (int col = 0; col < input->width; col++) {
			if (*ptrin < 0) {
				*ptrout = 1;
			} else {
				*ptrout = ((*ptrin) - minVal) / difference;
			}
			ptrout++;
			ptrin++;
		}
	}
}

void renderComponents(IplImage * SWTImage,
		std::vector<std::vector<Point2d> > & components, IplImage * output) {
	cvZero(output);
	for (std::vector<std::vector<Point2d> >::iterator it = components.begin();
			it != components.end(); it++) {
		for (std::vector<Point2d>::iterator pit = it->begin(); pit != it->end();
				pit++) {
			CV_IMAGE_ELEM(output, float, pit->y, pit->x) = CV_IMAGE_ELEM(
					SWTImage, float, pit->y, pit->x);
		}
	}
	for (int row = 0; row < output->height; row++) {
		float* ptr = (float*) (output->imageData + row * output->widthStep);
		for (int col = 0; col < output->width; col++) {
			if (*ptr == 0) {
				*ptr = -1;
			}
			ptr++;
		}
	}
	float maxVal = 0;
	float minVal = 1e100;
	for (int row = 0; row < output->height; row++) {
		const float* ptr = (const float*) (output->imageData
				+ row * output->widthStep);
		for (int col = 0; col < output->width; col++) {
			if (*ptr == 0) {
			} else {
				maxVal = std::max(*ptr, maxVal);
				minVal = std::min(*ptr, minVal);
			}
			ptr++;
		}
	}
	float difference = maxVal - minVal;
	for (int row = 0; row < output->height; row++) {
		float* ptr = (float*) (output->imageData + row * output->widthStep);\
		for (int col = 0; col < output->width; col++) {
			if (*ptr < 1) {
				*ptr = 1;
			} else {
				*ptr = ((*ptr) - minVal) / difference;
			}
			ptr++;
		}
	}

}

void renderComponentsWithBoxes(IplImage * SWTImage,
		std::vector<std::vector<Point2d> > & components,
		std::vector<std::pair<Point2d, Point2d> > & compBB, IplImage * output) {
	IplImage * outTemp = cvCreateImage(cvGetSize(output), IPL_DEPTH_32F, 1);

	renderComponents(SWTImage, components, outTemp);
	std::vector<std::pair<CvPoint, CvPoint> > bb;
	bb.reserve(compBB.size());
	for (std::vector<std::pair<Point2d, Point2d> >::iterator it =
			compBB.begin(); it != compBB.end(); it++) {
		CvPoint p0 = cvPoint(it->first.x, it->first.y);
		CvPoint p1 = cvPoint(it->second.x, it->second.y);
		std::pair<CvPoint, CvPoint> pair(p0, p1);
		bb.push_back(pair);
	}

	IplImage * out = cvCreateImage(cvGetSize(output), IPL_DEPTH_8U, 1);
	cvConvertScale(outTemp, out, 255, 0);
	cvCvtColor(out, output, CV_GRAY2RGB);
	cvReleaseImage ( &outTemp );
	cvReleaseImage ( &out );

	int count = 0;
	for (std::vector<std::pair<CvPoint, CvPoint> >::iterator it = bb.begin();
			it != bb.end(); it++) {
		CvScalar c;
		if (count % 3 == 0)
			c = cvScalar(255, 0, 0);
		else if (count % 3 == 1)
			c = cvScalar(0, 255, 0);
		else
			c = cvScalar(0, 0, 255);

		char *txt;
		asprintf(&txt, "%d", count);
		cv::Mat tmp_mat = cv::Mat(output);
		cv::rectangle(tmp_mat, it->first, it->second, c);
		cv::putText(tmp_mat, txt, it->first, cv::FONT_HERSHEY_SIMPLEX, 0.3, c);
		free(txt);
		count++;
	}
}

void renderChainsWithBoxes(IplImage * SWTImage,
		std::vector<std::vector<Point2d> > & components,
		std::vector<Chain> & chains,
		std::vector<std::pair<Point2d, Point2d> > & compBB,
		std::vector<std::pair<CvPoint, CvPoint> > & bb,
		IplImage * output) {
	// keep track of included components
	std::vector<bool> included;
	included.reserve(components.size());
	for (unsigned int i = 0; i != components.size(); i++) {
		included.push_back(false);
	}
	for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end();
			it++) {
		for (std::vector<int>::iterator cit = it->components.begin();
				cit != it->components.end(); cit++) {
			included[*cit] = true;
		}
	}
	std::vector<std::vector<Point2d> > componentsRed;
	for (unsigned int i = 0; i != components.size(); i++) {
		if (included[i]) {
			componentsRed.push_back(components[i]);
		}
	}
	IplImage * outTemp = cvCreateImage(cvGetSize(output), IPL_DEPTH_32F, 1);

	LOGL(LOG_CHAINS, componentsRed.size() << " components after chaining");

	renderComponents(SWTImage, componentsRed, outTemp);

	bb = findBoundingBoxes(chains, compBB, outTemp);

	IplImage * out = cvCreateImage(cvGetSize(output), IPL_DEPTH_8U, 1);
	cvConvertScale(outTemp, out, 255, 0);
	cvCvtColor(out, output, CV_GRAY2RGB);
	cvReleaseImage(&out);
	cvReleaseImage(&outTemp);
}

void renderChains(IplImage * SWTImage,
		std::vector<std::vector<Point2d> > & components,
		std::vector<Chain> & chains, IplImage * output) {
	// keep track of included components
	std::vector<bool> included;
	included.reserve(components.size());
	for (unsigned int i = 0; i != components.size(); i++) {
		included.push_back(false);
	}
	for (std::vector<Chain>::iterator it = chains.begin(); it != chains.end();
			it++) {
		for (std::vector<int>::iterator cit = it->components.begin();
				cit != it->components.end(); cit++) {
			included[*cit] = true;
		}
	}
	std::vector<std::vector<Point2d> > componentsRed;
	for (unsigned int i = 0; i != components.size(); i++) {
		if (included[i]) {
			componentsRed.push_back(components[i]);
		}
	}
	LOGL(LOG_CHAINS, componentsRed.size() << " components after chaining");
	IplImage * outTemp = cvCreateImage(cvGetSize(output), IPL_DEPTH_32F, 1);
	renderComponents(SWTImage, componentsRed, outTemp);
	cvConvertScale(outTemp, output, 255, 0);
	cvReleaseImage(&outTemp);
}

namespace textdetection {

TextDetector::TextDetector()
{
}

TextDetector::~TextDetector(void)
{
}

void TextDetector::detect(IplImage * input,
		const struct TextDetectionParams &params,
		std::vector<Chain> &chains,
		std::vector<std::pair<Point2d, Point2d> > &compBB,
		std::vector<std::pair<CvPoint, CvPoint> > &chainBB) {
	assert(input->depth == IPL_DEPTH_8U);
	assert(input->nChannels == 3);
	// Convert to grayscale
	IplImage * grayImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCvtColor(input, grayImage, CV_RGB2GRAY);
	// Create Canny Image
	double threshold_low = 175;
	double threshold_high = 320;
	IplImage * edgeImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvCanny(grayImage, edgeImage, threshold_low, threshold_high, 3);
	cvSaveImage("canny.png", edgeImage);

	// Create gradient X, gradient Y
	IplImage * gaussianImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_32F,
			1);
	cvConvertScale(grayImage, gaussianImage, 1. / 255., 0);
	cvSmooth(gaussianImage, gaussianImage, CV_GAUSSIAN, 5, 5);
	IplImage * gradientX = cvCreateImage(cvGetSize(input), IPL_DEPTH_32F, 1);
	IplImage * gradientY = cvCreateImage(cvGetSize(input), IPL_DEPTH_32F, 1);
	cvSobel(gaussianImage, gradientX, 1, 0, CV_SCHARR);
	cvSobel(gaussianImage, gradientY, 0, 1, CV_SCHARR);
	cvSmooth(gradientX, gradientX, 3, 3);
	cvSmooth(gradientY, gradientY, 3, 3);
	cvReleaseImage(&gaussianImage);

	// Calculate SWT and return ray vectors
	std::vector<Ray> rays;
	IplImage * SWTImage = cvCreateImage(cvGetSize(input), IPL_DEPTH_32F, 1);
	for (int row = 0; row < input->height; row++) {
		float* ptr = (float*) (SWTImage->imageData + row * SWTImage->widthStep);
		for (int col = 0; col < input->width; col++) {
			*ptr++ = -1;
		}
	}
	strokeWidthTransform(edgeImage, gradientX, gradientY, params, SWTImage,
			rays);
	cvSaveImage("SWT_0.png", SWTImage);
	SWTMedianFilter(SWTImage, rays);
	cvSaveImage("SWT_1.png", SWTImage);

	IplImage * output2 = cvCreateImage(cvGetSize(input), IPL_DEPTH_32F, 1);
	normalizeImage(SWTImage, output2);
	cvSaveImage("SWT_2.png", output2);
	IplImage * saveSWT = cvCreateImage(cvGetSize(input), IPL_DEPTH_8U, 1);
	cvConvertScale(output2, saveSWT, 255, 0);
	cvSaveImage("SWT.png", saveSWT);
	cvReleaseImage(&output2);
	cvReleaseImage(&saveSWT);

	// Calculate legally connected components from SWT and gradient image.
	// return type is a vector of vectors, where each outer vector is a component and
	// the inner vector contains the (y,x) of each pixel in that component.
	std::vector<std::vector<Point2d> > components =
			findLegallyConnectedComponents(SWTImage, rays);

	// Filter the components
	std::vector<std::vector<Point2d> > validComponents;
	std::vector<Point2dFloat> compCenters;
	std::vector<float> compMedians;
	std::vector<Point2d> compDimensions;
	filterComponents(SWTImage, components, validComponents, compCenters,
			compMedians, compDimensions, compBB, params);

	IplImage * output3 = cvCreateImage(cvGetSize(input), 8U, 3);
	renderComponentsWithBoxes(SWTImage, validComponents, compBB, output3);
	cvSaveImage("components.png", output3);
	cvReleaseImage ( &output3 );

	// Make chains of components
	chains = makeChains(input, validComponents, compCenters, compMedians,
			compDimensions, params);

	IplImage * output = cvCreateImage(cvGetSize(grayImage), IPL_DEPTH_8U, 3);
	renderChainsWithBoxes(SWTImage, validComponents, chains, compBB, chainBB, output);
	cvSaveImage("text-boxes.png", output);



	cvReleaseImage(&output);
	cvReleaseImage(&gradientX);
	cvReleaseImage(&gradientY);
	cvReleaseImage(&SWTImage);
	cvReleaseImage(&edgeImage);
	cvReleaseImage(&grayImage);
	return;
}

} /* namespace textdetection */

void strokeWidthTransform(IplImage * edgeImage, IplImage * gradientX,
		IplImage * gradientY, const struct TextDetectionParams &params,
		IplImage * SWTImage, std::vector<Ray> & rays) {
	// First pass
	float prec = .05;
	for (int row = 0; row < edgeImage->height; row++) {
		const uchar* ptr = (const uchar*) (edgeImage->imageData
				+ row * edgeImage->widthStep);
		for (int col = 0; col < edgeImage->width; col++) {
			if (*ptr > 0) {
				Ray r;

				Point2d p;
				p.x = col;
				p.y = row;
				r.p = p;
				std::vector<Point2d> points;
				points.push_back(p);

				float curX = (float) col + 0.5;
				float curY = (float) row + 0.5;
				int curPixX = col;
				int curPixY = row;
				float G_x = CV_IMAGE_ELEM(gradientX, float, row, col);
				float G_y = CV_IMAGE_ELEM(gradientY, float, row, col);
				// normalize gradient
				float mag = sqrt((G_x * G_x) + (G_y * G_y));
				if (params.darkOnLight) {
					G_x = -G_x / mag;
					G_y = -G_y / mag;
				} else {
					G_x = G_x / mag;
					G_y = G_y / mag;

				}
				while (true) {
					curX += G_x * prec;
					curY += G_y * prec;
					if ((int) (floor(curX)) != curPixX
							|| (int) (floor(curY)) != curPixY) {
						curPixX = (int) (floor(curX));
						curPixY = (int) (floor(curY));
						// check if pixel is outside boundary of image
						if (curPixX < 0 || (curPixX >= SWTImage->width)
								|| curPixY < 0
								|| (curPixY >= SWTImage->height)) {
							break;
						}
						Point2d pnew;
						pnew.x = curPixX;
						pnew.y = curPixY;
						points.push_back(pnew);

						if (CV_IMAGE_ELEM(edgeImage, uchar, curPixY, curPixX)
								> 0) {
							r.q = pnew;
							// dot product
							float G_xt = CV_IMAGE_ELEM(gradientX, float,
									curPixY, curPixX);
							float G_yt = CV_IMAGE_ELEM(gradientY, float,
									curPixY, curPixX);
							mag = sqrt((G_xt * G_xt) + (G_yt * G_yt));
							if (params.darkOnLight) {
								G_xt = -G_xt / mag;
								G_yt = -G_yt / mag;
							} else {
								G_xt = G_xt / mag;
								G_yt = G_yt / mag;

							}

							if (acos(G_x * -G_xt + G_y * -G_yt) < PI / 2.0) {
								float length =
										sqrt(
												((float) r.q.x - (float) r.p.x)
														* ((float) r.q.x
																- (float) r.p.x)
														+ ((float) r.q.y
																- (float) r.p.y)
																* ((float) r.q.y
																		- (float) r.p.y));
								if (length > params.maxStrokeLength)
									break;

								for (std::vector<Point2d>::iterator pit =
										points.begin(); pit != points.end();
										pit++) {
									if (CV_IMAGE_ELEM(SWTImage, float, pit->y,
											pit->x) < 0) {
										CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) =
												length;
									} else {
										CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) =
												std::min(length,
														CV_IMAGE_ELEM(SWTImage,
																float, pit->y,
																pit->x));
									}
								}
								r.points = points;
								rays.push_back(r);
							}
							break;
						}
					}
				}
			}
			ptr++;
		}
	}

}

void SWTMedianFilter(IplImage * SWTImage, std::vector<Ray> & rays) {
	for (std::vector<Ray>::iterator rit = rays.begin(); rit != rays.end();
			rit++) {
		for (std::vector<Point2d>::iterator pit = rit->points.begin();
				pit != rit->points.end(); pit++) {
			pit->SWT = CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x);
		}
		std::sort(rit->points.begin(), rit->points.end(), &Point2dSort);
		float median = (rit->points[rit->points.size() / 2]).SWT;
		for (std::vector<Point2d>::iterator pit = rit->points.begin();
				pit != rit->points.end(); pit++) {
			CV_IMAGE_ELEM(SWTImage, float, pit->y, pit->x) = std::min(pit->SWT,
					median);
		}
	}

}

bool Point2dSort(const Point2d &lhs, const Point2d &rhs) {
	return lhs.SWT < rhs.SWT;
}

std::vector<std::vector<Point2d> > findLegallyConnectedComponents(
		IplImage * SWTImage, std::vector<Ray> &rays) {
	boost::unordered_map<int, int> map;
	boost::unordered_map<int, Point2d> revmap;

	typedef boost::adjacency_list<boost::vecS, boost::vecS, boost::undirectedS> Graph;
	int num_vertices = 0;
	// Number vertices for graph.  Associate each point with number
	for (int row = 0; row < SWTImage->height; row++) {
		float * ptr = (float*) (SWTImage->imageData + row * SWTImage->widthStep);
		for (int col = 0; col < SWTImage->width; col++) {
			if (*ptr > 0) {
				map[row * SWTImage->width + col] = num_vertices;
				Point2d p;
				p.x = col;
				p.y = row;
				revmap[num_vertices] = p;
				num_vertices++;
			}
			ptr++;
		}
	}

	Graph g(num_vertices);

	for (int row = 0; row < SWTImage->height; row++) {
		float * ptr = (float*) (SWTImage->imageData + row * SWTImage->widthStep);
		for (int col = 0; col < SWTImage->width; col++) {
			if (*ptr > 0) {
				// check pixel to the right, right-down, down, left-down
				int this_pixel = map[row * SWTImage->width + col];
				if (col + 1 < SWTImage->width) {
					float right = CV_IMAGE_ELEM(SWTImage, float, row, col + 1);
					if (right > 0
							&& ((*ptr) / right <= 3.0 || right / (*ptr) <= 3.0))
						boost::add_edge(this_pixel,
								map.at(row * SWTImage->width + col + 1), g);
				}
				if (row + 1 < SWTImage->height) {
					if (col + 1 < SWTImage->width) {
						float right_down = CV_IMAGE_ELEM(SWTImage, float,
								row + 1, col + 1);
						if (right_down > 0
								&& ((*ptr) / right_down <= 3.0
										|| right_down / (*ptr) <= 3.0))
							boost::add_edge(this_pixel,
									map.at(
											(row + 1) * SWTImage->width + col
													+ 1), g);
					}
					float down = CV_IMAGE_ELEM(SWTImage, float, row + 1, col);
					if (down > 0
							&& ((*ptr) / down <= 3.0 || down / (*ptr) <= 3.0))
						boost::add_edge(this_pixel,
								map.at((row + 1) * SWTImage->width + col), g);
					if (col - 1 >= 0) {
						float left_down = CV_IMAGE_ELEM(SWTImage, float,
								row + 1, col - 1);
						if (left_down > 0
								&& ((*ptr) / left_down <= 3.0
										|| left_down / (*ptr) <= 3.0))
							boost::add_edge(this_pixel,
									map.at(
											(row + 1) * SWTImage->width + col
													- 1), g);
					}
				}
			}
			ptr++;
		}
	}

	std::vector<int> c(num_vertices);

	int num_comp = connected_components(g, &c[0]);

	std::vector<std::vector<Point2d> > components;
	components.reserve(num_comp);
	LOGL(LOG_COMPONENTS,
			"Before filtering, " << num_comp << " components and " << num_vertices << " vertices");
	for (int j = 0; j < num_comp; j++) {
		std::vector<Point2d> tmp;
		components.push_back(tmp);
	}
	for (int j = 0; j < num_vertices; j++) {
		Point2d p = revmap[j];
		(components[c[j]]).push_back(p);
	}

	return components;
}

void componentStats(IplImage * SWTImage, const std::vector<Point2d> & component,
		float & mean, float & variance, float & median, int & minx, int & miny,
		int & maxx, int & maxy) {
	std::vector<float> temp;
	temp.reserve(component.size());
	mean = 0;
	variance = 0;
	minx = 1000000;
	miny = 1000000;
	maxx = 0;
	maxy = 0;
	for (std::vector<Point2d>::const_iterator it = component.begin();
			it != component.end(); it++) {
		float t = CV_IMAGE_ELEM(SWTImage, float, it->y, it->x);
		mean += t;
		temp.push_back(t);
		miny = std::min(miny, it->y);
		minx = std::min(minx, it->x);
		maxy = std::max(maxy, it->y);
		maxx = std::max(maxx, it->x);
	}
	mean = mean / ((float) component.size());
	for (std::vector<float>::const_iterator it = temp.begin(); it != temp.end();
			it++) {
		variance += (*it - mean) * (*it - mean);
	}
	variance = variance / ((float) component.size());
	std::sort(temp.begin(), temp.end());
	median = temp[temp.size() / 2];
}
#define NO_FILTER
void filterComponents(IplImage * SWTImage,
		std::vector<std::vector<Point2d> > & components,
		std::vector<std::vector<Point2d> > & validComponents,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians, std::vector<Point2d> & compDimensions,
		std::vector<std::pair<Point2d, Point2d> > & compBB,
		const struct TextDetectionParams &params) {
	validComponents.reserve(components.size());
	compCenters.reserve(components.size());
	compMedians.reserve(components.size());
	compDimensions.reserve(components.size());
	// bounding boxes
	compBB.reserve(components.size());
	for (std::vector<std::vector<Point2d> >::iterator it = components.begin();
			it != components.end(); it++) {
		// compute the stroke width mean, variance, median
		float mean, variance, median;
		int minx, miny, maxx, maxy;
		componentStats(SWTImage, (*it), mean, variance, median, minx, miny,
				maxx, maxy);
#ifndef NO_FILTER
		// check if variance is less than half the mean
		if (variance > 0.5 * mean) {
			continue;
		}
#endif

		float length = (float) (maxx - minx + 1);
		float width = (float) (maxy - miny + 1);

		// check font height
		if (width > 300) {
			continue;
		}

		// check borders
		if ((miny < params.topBorder)
				|| (maxy > SWTImage->height - params.bottomBorder)) {
			continue;
		}

		float area = length * width;
		// compute the rotated bounding box
		float increment = 1. / 36.;
		for (float theta = increment * PI; theta < PI / 2.0;
				theta += increment * PI) {
			float xmin, xmax, ymin, ymax, xtemp, ytemp, ltemp, wtemp;
			xmin = 1000000;
			ymin = 1000000;
			xmax = 0;
			ymax = 0;
			for (unsigned int i = 0; i < (*it).size(); i++) {
				xtemp = (*it)[i].x * cos(theta) + (*it)[i].y * -sin(theta);
				ytemp = (*it)[i].x * sin(theta) + (*it)[i].y * cos(theta);
				xmin = std::min(xtemp, xmin);
				xmax = std::max(xtemp, xmax);
				ymin = std::min(ytemp, ymin);
				ymax = std::max(ytemp, ymax);
			}
			ltemp = xmax - xmin + 1;
			wtemp = ymax - ymin + 1;
			if (ltemp * wtemp < area) {
				area = ltemp * wtemp;
				length = ltemp;
				width = wtemp;
			}
		}

		// check if the aspect ratio is between the allowed range
		if (!ratio_within(length / width, COM_MAX_ASPECT_RATIO)) {
			continue;
		}

		// compute the diameter TODO finish
		// compute dense representation of component
		std::vector<std::vector<float> > denseRepr;
		denseRepr.reserve(maxx - minx + 1);
		for (int i = 0; i < maxx - minx + 1; i++) {
			std::vector<float> tmp;
			tmp.reserve(maxy - miny + 1);
			denseRepr.push_back(tmp);
			for (int j = 0; j < maxy - miny + 1; j++) {
				denseRepr[i].push_back(0);
			}
		}
		for (std::vector<Point2d>::iterator pit = it->begin(); pit != it->end();
				pit++) {
			(denseRepr[pit->x - minx])[pit->y - miny] = 1;
		}
		// create graph representing components
		/*const int num_nodes = it->size();

		 E edges[] = { E(0,2),
		 E(1,1), E(1,3), E(1,4),
		 E(2,1), E(2,3),
		 E(3,4),
		 E(4,0), E(4,1) };

		 Graph G(edges + sizeof(edges) / sizeof(E), weights, num_nodes);
		 */
		Point2dFloat center;
		center.x = ((float) (maxx + minx)) / 2.0;
		center.y = ((float) (maxy + miny)) / 2.0;

		Point2d dimensions;
		dimensions.x = maxx - minx + 1;
		dimensions.y = maxy - miny + 1;

		Point2d bb1;
		bb1.x = minx;
		bb1.y = miny;

		Point2d bb2;
		bb2.x = maxx;
		bb2.y = maxy;
		std::pair<Point2d, Point2d> pair(bb1, bb2);

		compBB.push_back(pair);
		compDimensions.push_back(dimensions);
		compMedians.push_back(median);
		compCenters.push_back(center);
		validComponents.push_back(*it);
	}
	std::vector<std::vector<Point2d> > tempComp;
	std::vector<Point2d> tempDim;
	std::vector<float> tempMed;
	std::vector<Point2dFloat> tempCenters;
	std::vector<std::pair<Point2d, Point2d> > tempBB;
	tempComp.reserve(validComponents.size());
	tempCenters.reserve(validComponents.size());
	tempDim.reserve(validComponents.size());
	tempMed.reserve(validComponents.size());
	tempBB.reserve(validComponents.size());
	for (unsigned int i = 0; i < validComponents.size(); i++) {
		int count = 0;
		for (unsigned int j = 0; j < validComponents.size(); j++) {
			if (i != j) {
				if (compBB[i].first.x <= compCenters[j].x
						&& compBB[i].second.x >= compCenters[j].x
						&& compBB[i].first.y <= compCenters[j].y
						&& compBB[i].second.y >= compCenters[j].y) {
					count++;
				}
			}
		}
		if (count < 2) {
			tempComp.push_back(validComponents[i]);
			tempCenters.push_back(compCenters[i]);
			tempMed.push_back(compMedians[i]);
			tempDim.push_back(compDimensions[i]);
			tempBB.push_back(compBB[i]);
		}
	}
	validComponents = tempComp;
	compDimensions = tempDim;
	compMedians = tempMed;
	compCenters = tempCenters;
	compBB = tempBB;

	compDimensions.reserve(tempComp.size());
	compMedians.reserve(tempComp.size());
	compCenters.reserve(tempComp.size());
	validComponents.reserve(tempComp.size());
	compBB.reserve(tempComp.size());

	LOGL(LOG_COMPONENTS,
			"After filtering " << validComponents.size() << " components");

	for (unsigned int i = 0; i < validComponents.size(); i++) {
		LOGL(LOG_COMPONENTS,
				"Component (" << i << "): dim=" << compDimensions[i].x << "*" << compDimensions[i].y << " median=" << compMedians[i] << " bb=(" << compBB[i].first.x << "," << compBB[i].first.y << ")->(" << compBB[i].second.x << "," << compBB[i].second.y << ")");

	}
}

bool sharesOneEnd(Chain c0, Chain c1) {
	if (c0.p == c1.p || c0.p == c1.q || c0.q == c1.q || c0.q == c1.p) {
		return true;
	} else {
		return false;
	}
}

bool chainSortDist(const Chain &lhs, const Chain &rhs) {
	return lhs.dist < rhs.dist;
}

bool chainSortLength(const Chain &lhs, const Chain &rhs) {
	return lhs.components.size() > rhs.components.size();
}

bool includes(std::vector<int> v, std::vector<int> V)
{
	if (v.size() > V.size())
		return false;

	for (int i=0,iend=v.size();i<iend;i++)
	{
		int j;
		int jend = V.size();
		for (j=0;j<jend;j++)
		{
			if (v[i] == V[j])
				break;
		}
		if (j==jend)
			return false;
	}
	return true;
}

std::vector<Chain> makeChains(IplImage * colorImage,
		std::vector<std::vector<Point2d> > & components,
		std::vector<Point2dFloat> & compCenters,
		std::vector<float> & compMedians, std::vector<Point2d> & compDimensions,
		const struct TextDetectionParams &params) {
	assert(compCenters.size() == components.size());
	// make vector of color averages
	std::vector<Point3dFloat> colorAverages;
	colorAverages.reserve(components.size());
	for (std::vector<std::vector<Point2d> >::iterator it = components.begin();
			it != components.end(); it++) {
		Point3dFloat mean;
		mean.x = 0;
		mean.y = 0;
		mean.z = 0;
		int num_points = 0;
		for (std::vector<Point2d>::iterator pit = it->begin(); pit != it->end();
				pit++) {
			mean.x += (float) CV_IMAGE_ELEM(colorImage, unsigned char, pit->y,
					(pit->x) * 3);
			mean.y += (float) CV_IMAGE_ELEM(colorImage, unsigned char, pit->y,
					(pit->x) * 3 + 1);
			mean.z += (float) CV_IMAGE_ELEM(colorImage, unsigned char, pit->y,
					(pit->x) * 3 + 2);
			num_points++;
		}
		mean.x = mean.x / ((float) num_points);
		mean.y = mean.y / ((float) num_points);
		mean.z = mean.z / ((float) num_points);
		colorAverages.push_back(mean);
	}

	// form all eligible pairs and calculate the direction of each
	std::vector<Chain> chains;
	for (unsigned int i = 0; i < components.size(); i++) {
		for (unsigned int j = i + 1; j < components.size(); j++) {
			// TODO add color metric
			float compMediansRatio = compMedians[i] / compMedians[j];
			float compDimRatioY = ((float) compDimensions[i].y)
					/ compDimensions[j].y;
			float compDimRatioX = ((float) compDimensions[i].x)
					/ compDimensions[j].x;
			float dist = square(compCenters[i].x - compCenters[j].x)
					+ square(compCenters[i].y - compCenters[j].y);
			float colorDist = square(colorAverages[i].x - colorAverages[j].x)
					+ square(colorAverages[i].y - colorAverages[j].y)
					+ square(colorAverages[i].z - colorAverages[j].z);
#if 0
			float maxDim = (float) square(
					std::max(std::min(compDimensions[i].x, compDimensions[i].y),
							std::min(compDimensions[j].x,
									compDimensions[j].y)));
#else
			float maxDim = (float) square(
								std::min(compDimensions[i].y,compDimensions[j].y));

#endif
			LOGL(LOG_COMP_PAIRS,
					"Pair (" << i << ":" << j << "): dist=" << dist << " colorDist=" << colorDist << " maxDim=" << maxDim << " compMediansRatio=" << compMediansRatio << " compDimRatioX=" << compDimRatioX << " compDimRatioY=" << compDimRatioY);

			if (ratio_within(compMediansRatio, COM_MAX_MEDIAN_RATIO)
					&& (ratio_within(compDimRatioY, COM_MAX_DIM_RATIO))
					&& (ratio_within(compDimRatioX, COM_MAX_DIM_RATIO))) {

				if (dist / maxDim < COM_MAX_DIST_RATIO /*&& colorDist < 6000*/) {
					Chain c;
					c.p = i;
					c.q = j;
					std::vector<int> comps;
					comps.push_back(c.p);
					comps.push_back(c.q);
					c.components = comps;
					c.dist = dist;
					float d_x = (compCenters[i].x - compCenters[j].x);
					float d_y = (compCenters[i].y - compCenters[j].y);
					/*
					 float d_x = (compBB[i].first.x - compBB[j].second.x);
					 float d_y = (compBB[i].second.y - compBB[j].second.y);
					 */
					float mag = sqrt(d_x * d_x + d_y * d_y);
					d_x = d_x / mag;
					d_y = d_y / mag;
					Point2dFloat dir;
					dir.x = d_x;
					dir.y = d_y;
					c.direction = dir;
					chains.push_back(c);

				}
			}
		}
	}

	/* print pairs */
	for (unsigned int j = 0; j < chains.size(); j++) {
		LOG(LOG_COMP_PAIRS, "Pair" << j <<":");
		for (unsigned int i = 0; i < chains[j].components.size(); i++) {
			LOG(LOG_COMP_PAIRS, chains[j].components[i] << ",");
		}
		LOGL(LOG_COMP_PAIRS, "");
	}

	LOGL(LOG_COMP_PAIRS, chains.size() << " eligible pairs");

	std::sort(chains.begin(), chains.end(), &chainSortDist);

	const float strictness = PI / 10.0;
	//merge chains
	int merges = 1;
	while (merges > 0) {
		for (unsigned int i = 0; i < chains.size(); i++) {
			chains[i].merged = false;
		}
		merges = 0;
		std::vector<Chain> newchains;
		for (unsigned int i = 0; i < chains.size(); i++) {
			for (unsigned int j = 0; j < chains.size(); j++) {
				if (i != j) {
					if (!chains[i].merged && !chains[j].merged
							&& sharesOneEnd(chains[i], chains[j])) {
						if (chains[i].p == chains[j].p) {
							if (acos(
									chains[i].direction.x
											* -chains[j].direction.x
											+ chains[i].direction.y
													* -chains[j].direction.y)
									< strictness) {
								/*      if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 }
								 std::cerr << 1 <<std::endl;

								 std::cerr << chains[i].p << " " << chains[i].q << std::endl;
								 std::cerr << chains[j].p << " " << chains[j].q << std::endl;
								 std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
								 std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
								 std::cerr << compCenters[chains[j].q].x << " " << compCenters[chains[j].q].y << std::endl;
								 std::cerr << std::endl; */

								chains[i].p = chains[j].q;
								for (std::vector<int>::iterator it =
										chains[j].components.begin();
										it != chains[j].components.end();
										it++) {
									chains[i].components.push_back(*it);
								}
								float d_x = (compCenters[chains[i].p].x
										- compCenters[chains[i].q].x);
								float d_y = (compCenters[chains[i].p].y
										- compCenters[chains[i].q].y);
								chains[i].dist = d_x * d_x + d_y * d_y;

								float mag = sqrt(d_x * d_x + d_y * d_y);
								d_x = d_x / mag;
								d_y = d_y / mag;
								Point2dFloat dir;
								dir.x = d_x;
								dir.y = d_y;
								chains[i].direction = dir;
								chains[j].merged = true;
								merges++;
								/*j=-1;
								 i=0;
								 if (i == chains.size() - 1) i=-1;
								 std::stable_sort(chains.begin(), chains.end(), &chainSortLength);*/
							}
						} else if (chains[i].p == chains[j].q) {
							if (acos(
									chains[i].direction.x
											* chains[j].direction.x
											+ chains[i].direction.y
													* chains[j].direction.y)
									< strictness) {
								/*
								 if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 }
								 std::cerr << 2 <<std::endl;

								 std::cerr << chains[i].p << " " << chains[i].q << std::endl;
								 std::cerr << chains[j].p << " " << chains[j].q << std::endl;
								 std::cerr << chains[i].direction.x << " " << chains[i].direction.y << std::endl;
								 std::cerr << chains[j].direction.x << " " << chains[j].direction.y << std::endl;
								 std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
								 std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
								 std::cerr << compCenters[chains[j].p].x << " " << compCenters[chains[j].p].y << std::endl;
								 std::cerr << std::endl; */

								chains[i].p = chains[j].p;
								for (std::vector<int>::iterator it =
										chains[j].components.begin();
										it != chains[j].components.end();
										it++) {
									chains[i].components.push_back(*it);
								}
								float d_x = (compCenters[chains[i].p].x
										- compCenters[chains[i].q].x);
								float d_y = (compCenters[chains[i].p].y
										- compCenters[chains[i].q].y);
								float mag = sqrt(d_x * d_x + d_y * d_y);
								chains[i].dist = d_x * d_x + d_y * d_y;

								d_x = d_x / mag;
								d_y = d_y / mag;

								Point2dFloat dir;
								dir.x = d_x;
								dir.y = d_y;
								chains[i].direction = dir;
								chains[j].merged = true;
								merges++;
								/*j=-1;
								 i=0;
								 if (i == chains.size() - 1) i=-1;
								 std::stable_sort(chains.begin(), chains.end(), &chainSortLength); */
							}
						} else if (chains[i].q == chains[j].p) {
							if (acos(
									chains[i].direction.x
											* chains[j].direction.x
											+ chains[i].direction.y
													* chains[j].direction.y)
									< strictness) {
								/*                           if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 }
								 std::cerr << 3 <<std::endl;

								 std::cerr << chains[i].p << " " << chains[i].q << std::endl;
								 std::cerr << chains[j].p << " " << chains[j].q << std::endl;

								 std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
								 std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
								 std::cerr << compCenters[chains[j].q].x << " " << compCenters[chains[j].q].y << std::endl;
								 std::cerr << std::endl; */
								chains[i].q = chains[j].q;
								for (std::vector<int>::iterator it =
										chains[j].components.begin();
										it != chains[j].components.end();
										it++) {
									chains[i].components.push_back(*it);
								}
								float d_x = (compCenters[chains[i].p].x
										- compCenters[chains[i].q].x);
								float d_y = (compCenters[chains[i].p].y
										- compCenters[chains[i].q].y);
								float mag = sqrt(d_x * d_x + d_y * d_y);
								chains[i].dist = d_x * d_x + d_y * d_y;

								d_x = d_x / mag;
								d_y = d_y / mag;
								Point2dFloat dir;
								dir.x = d_x;
								dir.y = d_y;

								chains[i].direction = dir;
								chains[j].merged = true;
								merges++;
								/*j=-1;
								 i=0;
								 if (i == chains.size() - 1) i=-1;
								 std::stable_sort(chains.begin(), chains.end(), &chainSortLength); */
							}
						} else if (chains[i].q == chains[j].q) {
							if (acos(
									chains[i].direction.x
											* -chains[j].direction.x
											+ chains[i].direction.y
													* -chains[j].direction.y)
									< strictness) {
								/*           if (chains[i].p == chains[i].q || chains[j].p == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].p && chains[i].q == chains[j].q) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 } else if (chains[i].p == chains[j].q && chains[i].q == chains[j].p) {
								 std::cout << "CRAZY ERROR" << std::endl;
								 }
								 std::cerr << 4 <<std::endl;
								 std::cerr << chains[i].p << " " << chains[i].q << std::endl;
								 std::cerr << chains[j].p << " " << chains[j].q << std::endl;
								 std::cerr << compCenters[chains[i].p].x << " " << compCenters[chains[i].p].y << std::endl;
								 std::cerr << compCenters[chains[i].q].x << " " << compCenters[chains[i].q].y << std::endl;
								 std::cerr << compCenters[chains[j].p].x << " " << compCenters[chains[j].p].y << std::endl;
								 std::cerr << std::endl; */
								chains[i].q = chains[j].p;
								for (std::vector<int>::iterator it =
										chains[j].components.begin();
										it != chains[j].components.end();
										it++) {
									chains[i].components.push_back(*it);
								}
								float d_x = (compCenters[chains[i].p].x
										- compCenters[chains[i].q].x);
								float d_y = (compCenters[chains[i].p].y
										- compCenters[chains[i].q].y);
								chains[i].dist = d_x * d_x + d_y * d_y;

								float mag = sqrt(d_x * d_x + d_y * d_y);
								d_x = d_x / mag;
								d_y = d_y / mag;
								Point2dFloat dir;
								dir.x = d_x;
								dir.y = d_y;
								chains[i].direction = dir;
								chains[j].merged = true;
								merges++;
								/*j=-1;
								 i=0;
								 if (i == chains.size() - 1) i=-1;
								 std::stable_sort(chains.begin(), chains.end(), &chainSortLength);*/
							}
						}
					}
				}
			}
		}
		for (unsigned int i = 0; i < chains.size(); i++) {
			if (!chains[i].merged) {
				newchains.push_back(chains[i]);
			}
		}
		chains = newchains;
		std::stable_sort(chains.begin(), chains.end(), &chainSortLength);
	}

	std::vector<Chain> newchains;
	newchains.reserve(chains.size());

	/* sort chains and remove duplicate components within chains */
	for (std::vector<Chain>::iterator cit = chains.begin(); cit != chains.end();
			cit++) {

		/* remove duplicate components */
		std::sort(cit->components.begin(), cit->components.end());
		cit->components.erase(
		std::unique(cit->components.begin(), cit->components.end()),
		cit->components.end());
	}

	/* now add all chains */
	for (int i=0,iend=chains.size(); i<iend; i++)
	{
		/* only add chains longer than minimum size */
		if (chains[i].components.size() < params.minChainLen) {
			LOGL(LOG_CHAINS, "Reject chain " << i << " on size ");
			break;
		}

		/* now make sure that chain is not already included
		 * in another chain */
		int j;
		for (j=0; j<iend; j++)
		{
			if ( (i!=j) && (includes(chains[i].components, chains[j].components)))
				break;
		}
		if (j<iend)
		{
			LOGL(LOG_CHAINS, "Reject chain " << i << " already included in chain " << j);
			break;
		}

		/* all good, add that chain */
		newchains.push_back(chains[i]);
	}

	chains = newchains;

	/* print chains */
	for (unsigned int j = 0; j < chains.size(); j++) {
		LOG(LOG_CHAINS, "Chain" << j <<":");
		for (unsigned int i = 0; i < chains[j].components.size(); i++) {
			LOG(LOG_CHAINS, chains[j].components[i] << ",");
		}
		LOGL(LOG_CHAINS, "");
	}

	LOGL(LOG_CHAINS, chains.size() << " chains after merging");

	return chains;
}
