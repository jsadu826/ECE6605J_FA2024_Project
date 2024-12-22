/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef ELLIPSE_NORMALIZATION_H_
#define ELLIPSE_NORMALIZATION_H_

#include <vector>
#include <algorithm>
#include <cmath>
#include <memory>
#include "lut_math.h"
#include "grid_info.h"
#include "image.h"
#include "mask.h"
#include "shape.h"
#include "point.h"
#include "array_deleter.h"
#include "matrix.h"

namespace msas
{

/**
 * Encapsulates logic related to normalization of an elliptical region to a disc
 * and subsequent interpolation of it to a regular grid. Computation of dominant
 * orientations, involved in the normalization workflow, is done as in SIFT.
 */
class EllipseNormalization {
public:
	/// @note All of the constructor parameters are related to the dominant orientations calculation.
	/// @param num_bins Number of bins in the orientation histogram.
	/// @param num_orientations Maximum number of orientations to be computed.
	/// @param histogram_cut_off Portion of the highest peak in the histogram below which we cut-off smaller peaks
	/// @param sigma Gaussian sigma for weighting distance from a point to the patch center.
	EllipseNormalization(int num_bins, int num_orientations, float histogram_cut_off, float sigma = DEFAULT_SIGMA);
	EllipseNormalization(int num_orientations, float histogram_cut_off = DEFAULT_HISTOGRAM_CUT_OFF);
	EllipseNormalization();

	/// Create regular grid of a given size.
	/// @note Since transformations are usually normalized by the radius, the default radius is set to 1.0f here
	std::shared_ptr<GridInfo> create_regular_grid(int grid_size, float radius = 1.0f);

	/// Calculate dominant orientations of gradient vectors within an elliptical region (patch).
	/// @param gradient_x X component of an image gradient.
	/// @param gradient_y Y component of an image gradient.
	/// @param region Set of points of the elliptical region.
	/// @param transform Normalizing transformation that maps the elliptical region to a disk.
	/// @param center Central point of the elliptical region.
	/// @return Set of dominant orientations (angles in radians).
	std::vector<float> calculate_dominant_orientations(const Image<float> &gradient_x,
													   const Image<float> &gradient_y,
													   const std::vector<Point> &region,
													   Matrix2f transform,
													   Point center);

	/// Normalize and interpolate an elliptical region (patch) to a regular grid.
	/// @param grid Regular grid to be used in the interpolation.
	/// @param image Original image.
	/// @param mask Mask defining allowed points.
	/// @param transform Normalizing transformation that maps the elliptical region to a disk.
	/// @param center Central point of the elliptical region.
	/// @return Set of interpolated color values that map ont-to-one to the grid nodes.
	std::shared_ptr<float*> interpolate_to_grid(const GridInfo &grid,
												const ImageFx<float> &image,
												const MaskFx &mask,
												Matrix2f transform,
												Point center);

	/// Rotate given normalized patch by 180 degrees.
	std::shared_ptr<float*> flip(const std::shared_ptr<float*> normalized_patch,
								 int grid_length,
								 uint number_of_channels);

	/// Compute rotation matrix from a dominant orientation.
	/// @param orientation Dominant orientation in radians.
	Matrix2f rotation(const float &orientation);

private:
	constexpr static float NO_VALUE = -9999.0f;

	// Parameters for SIFT-like dominant orientations calculation
	constexpr static int 	DEFAULT_NUM_BINS = 72;
	constexpr static int 	DEFAULT_NUM_ORIENTATIONS = 3;
	constexpr static float 	DEFAULT_HISTOGRAM_CUT_OFF = 0.45;	// Note: in the SIFT paper was suggested to be 0.8
	constexpr static float 	DEFAULT_SIGMA = 0.2f;

	int _num_bins;
	int _num_orientations;		// maximum number of orientations to be returned
	float _histogram_cut_off;	// portion of the highest peak in the histogram below which we cut-off smaller peaks
	float _sigma;				// Gaussian sigma for weighting distance from a point to the patch center

	struct dsc_sort_by_first {
		bool operator()(const std::pair<float, int> &left, const std::pair<float, int> &right) {
			return left.first > right.first;
		}
	};
};

}	// namespace msas

#endif /* ELLIPSE_NORMALIZATION_H_ */
