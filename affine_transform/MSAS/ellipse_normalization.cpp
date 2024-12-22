/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "ellipse_normalization.h"

using std::pair;
using std::vector;

namespace msas
{

EllipseNormalization::EllipseNormalization(int num_bins, int num_orientations, float histogram_cut_off, float sigma)
		: _num_bins(num_bins),
		  _num_orientations(num_orientations),
		  _histogram_cut_off(histogram_cut_off),
		  _sigma(sigma)
{

}


EllipseNormalization::EllipseNormalization(int num_orientations, float histogram_cut_off)
: _num_bins(DEFAULT_NUM_BINS),
  _num_orientations(num_orientations),
  _histogram_cut_off(histogram_cut_off),
  _sigma(DEFAULT_SIGMA)
{

}


EllipseNormalization::EllipseNormalization()
		: _num_bins(DEFAULT_NUM_BINS),
		  _num_orientations(DEFAULT_NUM_ORIENTATIONS),
		  _histogram_cut_off(DEFAULT_HISTOGRAM_CUT_OFF),
		  _sigma(DEFAULT_SIGMA)
{

}


std::shared_ptr<GridInfo> EllipseNormalization::create_regular_grid(int grid_size, float radius)
{
	// Allocate memory and prepare data structure
	std::shared_ptr<GridInfo> grid_info = std::make_shared<GridInfo>();
	grid_info->nodes.reset(new GridCoord[grid_size * grid_size]);
	grid_info->index.reset(new int[grid_size * grid_size]);
	grid_info->index_length = grid_size * grid_size;
	grid_info->size = grid_size;

	// Compute parameters
	float grid_radius = (float)grid_size / 2;
	float grid_step = radius / grid_radius;
	float radius_squared = radius * radius;

	// Go through the nodes of the grid and compute their coordinates
	int i = 0;
	for (int y = 0; y < grid_size; y++) {
		for (int x = 0; x < grid_size; x++) {
			float coord_x = grid_step / 2 + x * grid_step - radius;
			float coord_y = grid_step / 2 + y * grid_step - radius;

			// Skip the node, if it is outside of the inscribed circle
			if (coord_x * coord_x + coord_y * coord_y > radius_squared) {
				grid_info->index[y * grid_size + x] = -1;
				continue;
			}

			grid_info->nodes[i].x = coord_x;
			grid_info->nodes[i].y = coord_y;
			grid_info->nodes[i].index_x = x;
			grid_info->nodes[i].index_y = y;
			grid_info->index[y * grid_size + x] = i;
			i++;
		}
	}

	grid_info->nodes_length = i;
	grid_info->step = grid_step;

	return grid_info;
}


vector<float> EllipseNormalization::calculate_dominant_orientations(const Image<float> &gradient_x,
																	const Image<float> &gradient_y,
																	const vector<Point> &region,
																	Matrix2f transform,
																	Point center)
{
	// Compute parameters
	float bin_width = 2.0f * (float)M_PI / (float) _num_bins;
	float two_sigma_squared = 2.0f * _sigma * _sigma;
	float gauss_normalization = (sqrt(2.0 * M_PI) * _sigma);

	float transform_00 = transform[0];
	float transform_01 = transform[1];
	float transform_10 = transform[2];
	float transform_11 = transform[3];

	float det = transform_00 * transform_11 - transform_01 * transform_10;

	// Ensure that transform is valid
	if (std::isnan(det) || det == 0.0f) {
		vector<float> default_orientations = {0.0f};
		return default_orientations;
	}

	// Calculate transposed inverse of 'transform' to transform gradients
	float grad_tr_00 = transform_11 / det;
	float grad_tr_01 = -transform_10 / det;
	float grad_tr_10 = -transform_01 / det;
	float grad_tr_11 = transform_00 / det;

	// Fill-in histogram
	int histogram_length = _num_bins + 2;	// '+ 2' because we reserve first and last elements for circular convolution
	float *histogram = new float[histogram_length]();
	for (auto it = region.cbegin(); it != region.cend(); ++it) {
		// Transform gradient
		float grad_x = gradient_x(it->x, it->y) * grad_tr_00 +
					   gradient_y(it->x, it->y) * grad_tr_01;
		float grad_y = gradient_x(it->x, it->y) * grad_tr_10 +
					   gradient_y(it->x, it->y) * grad_tr_11;

		// Calculate gradient direction and norm
		float grad_norm = std::sqrt(grad_x * grad_x + grad_y * grad_y);
		float angle = std::atan2(grad_y, grad_x);    // from X axis, in interval [-pi,+pi] radians

		// Re-arrange angle: [0,+pi] -> [0,+pi] and [-pi,0] -> [+pi,+2pi]
		if (angle < 0.0f) {
			angle += 2.0f * M_PI;
		}

		// Locate a proper bin and calculate distance to it's center
		float bin_pos = angle / bin_width;
		int bin_id = (int)floor(bin_pos - 0.5f);
		float bin_distance = bin_pos - bin_id - 0.5f;

		// Calculate anisotropic intra-patch Gaussian weights
		float delta_x = it->x - center.x;
		float delta_y = it->y - center.y;
		float dist_x = transform_00 * delta_x + transform_01 * delta_y;
		float dist_y = transform_10 * delta_x + transform_11 * delta_y;
		float distance = dist_x * dist_x + dist_y * dist_y;
		float weight = LUT::exp(-distance / two_sigma_squared) / gauss_normalization;

		// Distribute gradient norm value between two closest bins
		histogram[bin_id + 1] += (1.0f - bin_distance) * grad_norm * weight;		// '+ 1' because we reserve first element for circular convolution
		histogram[bin_id + 2] += (bin_distance) * grad_norm * weight;
	}

	// Merge values at the borders to make the histogram circular
	histogram[0] += histogram[_num_bins];
	histogram[_num_bins + 1] += histogram[1];
	histogram[_num_bins] = histogram[0];
	histogram[1] = histogram[_num_bins + 1];

	// Smooth histogram (convolve the histogram with [1/3, 1/3, 1/3] kernel several times)
	float *buffer_src = histogram;
	float *buffer_dst = new float[histogram_length]();
	for(int i = 0 ; i < 6; i++)	{
		for (int j = 1; j <= _num_bins; j++) {
			buffer_dst[j] = (buffer_src[j - 1] +
							 buffer_src[j] +
							 buffer_src[j + 1]) / 3.0f;
		}

		// Update boundaries
		buffer_dst[0] = buffer_dst[_num_bins];
		buffer_dst[_num_bins + 1] = buffer_dst[1];

		// Swap buffers
		std::swap(buffer_dst, buffer_src);
	}
	delete[] buffer_dst;

	histogram = buffer_src;

	// Find maximum value in histogram
	float max_value = -1.0f;
	for(int i = 1; i <= _num_bins; ++i) {
		max_value = std::max(max_value, histogram[i]);
	}

	// Locate candidate orientations
	vector<float> dominant_orientations;
	vector<pair<float, int> > candidate_orientations;
	candidate_orientations.reserve(10);
	float cut_off = _histogram_cut_off * max_value;
	for(int i = 1; i <= _num_bins; ++i) {
		if( (histogram[i] > cut_off) && (histogram[i] > histogram[i-1]) && (histogram[i] > histogram[i+1]) ) {
			candidate_orientations.push_back(pair<float, int>(histogram[i], i));
		}
	}

	// Sort candidate orientations by the histogram value
	std::sort(candidate_orientations.begin(), candidate_orientations.end(), dsc_sort_by_first());

	// Select at most '_num_orientations' best orientations
	auto it = candidate_orientations.begin();
	for (int i = 0; i < _num_orientations && it != candidate_orientations.end(); ++i, ++it) {
		int id = it->second;
		float angle = bin_width * ((float)id + 0.5f * (histogram[id-1] - histogram[id+1]) / (histogram[id-1] - 2.0f * histogram[id] + histogram[id+1]) + 0.5f);
		dominant_orientations.push_back(angle);
	}

	delete[] histogram;

	// If no dominant orientation is detected, return the original one
	if (dominant_orientations.size() == 0) {
		dominant_orientations.push_back(0.0f);
	}

	return dominant_orientations;
}


std::shared_ptr<float*> EllipseNormalization::interpolate_to_grid(const GridInfo &grid,
																  const ImageFx<float> &image,
																  const MaskFx &mask,
																  Matrix2f transform,
																  Point center)
{
	uint number_of_channels = image.number_of_channels();
	Shape size = image.size();

	// Allocate memory
	float** interpolated_values = new float*[number_of_channels];
	for (int ch = 0; ch < number_of_channels; ch++) {
		interpolated_values[ch] = new float[grid.nodes_length]();
	}

	// Get raw pointers
	const bool* mask_data = (!mask.is_empty()) ? mask.raw() : 0;
	const float* image_data = image.raw();

	float transform_00 = transform[0];
	float transform_01 = transform[1];
	float transform_10 = transform[2];
	float transform_11 = transform[3];
	float det_transform = transform_00 * transform_11 - transform_01 * transform_10;

	if (!mask_data) {
		for (int i = 0; i < grid.nodes_length; i++) {
			// Map grid points to the elliptical patch
			float x = (transform_11 * grid.nodes[i].x - transform_01 * grid.nodes[i].y) / det_transform + center.x;
			float y = (transform_00 * grid.nodes[i].y - transform_10 * grid.nodes[i].x) / det_transform + center.y;

			// Check that point is inside the image domain
			if (x < 0.0f || x > (size.size_x - 1) || y < 0.0f || y > (size.size_y - 1)) {
				for (int ch = 0; ch < number_of_channels; ch++) {
					interpolated_values[ch][i] = NO_VALUE;
				}
				continue;
			}

			int ix = (int) x;
			int iy = (int) y;
			int index = iy * size.size_x + ix;
			float dx = x - (float) ix;
			float dy = y - (float) iy;

			// Do interpolation
			if (ix + 1 < size.size_x && iy + 1 < size.size_y) {
				// Bilinear interpolation
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dx) * (1.0f - dy)
									  + image_data[number_of_channels * (index + 1) + ch] * dx * (1.0f - dy)
									  + image_data[number_of_channels * (index + size.size_x) + ch] * (1.0f - dx) * dy
									  + image_data[number_of_channels * (index + size.size_x + 1) + ch] * dx * dy;
					interpolated_values[ch][i] = intensity;
				}
			} else if (ix + 1 < size.size_x) {
				// Linear interpolation in X direction for the bottom edge
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dx)
									  + image_data[number_of_channels * (index + 1) + ch] * dx;
					interpolated_values[ch][i] = intensity;
				}
			} else if (iy + 1 < size.size_y) {
				// Linear interpolation in Y direction for the right edge
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dy)
									  + image_data[number_of_channels * (index + size.size_x) + ch] * dy;
					interpolated_values[ch][i] = intensity;
				}
			} else {
				// No interpolation for the bottom-right corner
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch];
					interpolated_values[ch][i] = intensity;
				}
			}
		}
	} else {
		for (int i = 0; i < grid.nodes_length; i++) {
			// Map grid points to the elliptical patch
			float x = (transform_11 * grid.nodes[i].x - transform_01 * grid.nodes[i].y) / det_transform + center.x;
			float y = (transform_00 * grid.nodes[i].y - transform_10 * grid.nodes[i].x) / det_transform + center.y;

			// Check that point is inside the image domain
			if (x < 0.0f || x > (size.size_x - 1) || y < 0.0f || y > (size.size_y - 1)) {
				for (int ch = 0; ch < number_of_channels; ch++) {
					interpolated_values[ch][i] = NO_VALUE;
				}
				continue;
			}

			int ix = (int) x;
			int iy = (int) y;
			int index = iy * size.size_x + ix;

			// Check that point is included in the mask
			if (!mask_data[index] || !mask_data[index + 1]
				|| !mask_data[index + size.size_x] || !mask_data[index + size.size_x + 1]) {
				for (int ch = 0; ch < number_of_channels; ch++) {
					interpolated_values[ch][i] = NO_VALUE;
				}
				continue;
			}

			float dx = x - (float) ix;
			float dy = y - (float) iy;

			// Do interpolation
			if (ix + 1 < size.size_x && iy + 1 < size.size_y) {
				// Bilinear interpolation
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dx) * (1.0f - dy)
									  + image_data[number_of_channels * (index + 1) + ch] * dx * (1.0f - dy)
									  + image_data[number_of_channels * (index + size.size_x) + ch] * (1.0f - dx) * dy
									  + image_data[number_of_channels * (index + size.size_x + 1) + ch] * dx * dy;
					interpolated_values[ch][i] = intensity;
				}
			} else if (ix + 1 < size.size_x) {
				// Linear interpolation in X direction for the bottom edge
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dx)
									  + image_data[number_of_channels * (index + 1) + ch] * dx;
					interpolated_values[ch][i] = intensity;
				}
			} else if (iy + 1 < size.size_y) {
				// Linear interpolation in Y direction for the right edge
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch] * (1.0f - dy)
									  + image_data[number_of_channels * (index + size.size_x) + ch] * dy;
					interpolated_values[ch][i] = intensity;
				}
			} else {
				// No interpolation for the bottom-right corner
				for (int ch = 0; ch < number_of_channels; ch++) {
					float intensity = image_data[number_of_channels * index + ch];
					interpolated_values[ch][i] = intensity;
				}
			}
		}
	}

	std::shared_ptr<float*> ptr(interpolated_values, ArrayDeleter2d<float>(number_of_channels));

	return ptr;

}


std::shared_ptr<float*> EllipseNormalization::flip(const std::shared_ptr<float*> normalized_patch,
												   int grid_length,
												   uint number_of_channels)
{
	// Flip normalized patch by reversing it
	float** original_patch = normalized_patch.get();
	float** flipped_patch = new float*[number_of_channels];
	for (int ch = 0; ch < number_of_channels; ch++) {
		flipped_patch[ch] = new float[grid_length];
		for (int i = 0; i < grid_length; i++) {
			flipped_patch[ch][i] = original_patch[ch][grid_length - i - 1];
		}
	}

	std::shared_ptr<float*> ptr(flipped_patch, ArrayDeleter2d<float>(number_of_channels));
	return ptr;
}


Matrix2f EllipseNormalization::rotation(const float &orientation)
{
	// Rotation that aligns given orientation with X axis
	Matrix2f rotation;
	rotation[0] = std::cos(orientation);
	rotation[1] = std::sin(orientation);
	rotation[2] = -std::sin(orientation);
	rotation[3] = std::cos(orientation);

	return rotation;
}

}	// namespace msas
