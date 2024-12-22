/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "affine_patch_distance.h"

using std::vector;

namespace msas
{

AffinePatchDistance::AffinePatchDistance(int grid_size)
: _scale(1.0f),
  _bilateral_k_color(0.0f),
  _bilateral_k_spatial(1.0f),
  _use_bilateral(false),
  _grid_size(grid_size),
  _reference_channel(-1),
  _use_cache(true)
{
	_grid = _normalization.create_regular_grid(_grid_size);
	_weights.reset(calculate_weights(_grid.get(), _scale));
}


AffinePatchDistance::~AffinePatchDistance()
{

}


msas::DistanceInfo AffinePatchDistance::calculate(const msas::StructureTensorBundle &source_bundle,
												  Point source_point,
												  const msas::StructureTensorBundle &target_bundle,
												  Point target_point)
{
	// Get normalized patches

	vector<NormalizedPatch> *normalized_source = (_use_cache) ?
												 source_bundle.normalized_patch(source_point.x, source_point.y) :
												 new vector<NormalizedPatch>();

	if (!normalized_source->size()) {
        normalize_patch_internal(source_bundle, source_point, *normalized_source);
	}

	vector<NormalizedPatch> *normalized_target = (_use_cache) ?
												 target_bundle.normalized_patch(target_point.x, target_point.y) :
												 new vector<NormalizedPatch>();

	if (!normalized_target->size()) {
		normalize_patch_internal(target_bundle, target_point, *normalized_target);
	}

	// Find minimum distance among all possible combinations of orientations
	int number_of_channels = target_bundle.image().number_of_channels();
	float target_radius = target_bundle.radius();
	int target_id = -2;
	int source_id = -2;
	msas::DistanceInfo min_distance;
	if (_use_bilateral) {
		min_distance.distance = calculate_geodesic(*normalized_source, *normalized_target, target_radius, number_of_channels, source_id, target_id);
	} else {
		min_distance.distance = calculate_gaussian(*normalized_source, *normalized_target, target_radius, number_of_channels, source_id, target_id);
	}

	// Fill min_distance
	min_distance.first_point = source_point;
	min_distance.second_point = target_point;
	min_distance.first_transform = Matrix::multiply((*normalized_source)[source_id].extra_transform,
													(*normalized_source)[source_id].base_transform);
	min_distance.second_transform = Matrix::multiply((*normalized_target)[target_id].extra_transform,
													 (*normalized_target)[target_id].base_transform);

	if (!_use_cache) {
		delete normalized_source;
		delete normalized_target;
	}

	return min_distance;
}


float AffinePatchDistance::scale()
{
	return _scale;
}


void AffinePatchDistance::set_scale(float value)
{
	_scale = value;
	update_weights();
}


float AffinePatchDistance::bilateral_k_color()
{
	return _bilateral_k_color;
}


void AffinePatchDistance::set_bilateral_k_color(float value)
{
	_bilateral_k_color = value;
	_use_bilateral = std::abs(_bilateral_k_color) > EPS;
}


float AffinePatchDistance::bilateral_k_spatial()
{
	return _bilateral_k_spatial;
}


void AffinePatchDistance::set_bilateral_k_spatial(float value)
{
	_bilateral_k_spatial = value;
	update_weights();
}


std::shared_ptr<GridInfo> AffinePatchDistance::grid()
{
	return _grid;
}


size_t AffinePatchDistance::normalized_patch_length()
{
	return _grid->nodes_length;
}


int AffinePatchDistance::grid_size()
{
	return _grid_size;
}


void AffinePatchDistance::set_grid_size(int value)
{
	if (_grid_size != value) {
		_grid_size = value;

		// Recompute grid
		_grid = _normalization.create_regular_grid(_grid_size);

		// Recompute weights
		_weights.reset(calculate_weights(_grid.get(), _scale));
	}
}


int AffinePatchDistance::reference_channel() const
{
	return _reference_channel;
}


void AffinePatchDistance::set_reference_channel(int value)
{
	_reference_channel = value;
}


void AffinePatchDistance::set_use_cache(bool value)
{
	_use_cache = value;
}


void AffinePatchDistance::precompute_normalized_patches(const StructureTensorBundle &bundle)
{
	if (!_use_cache) {
		return;
	}

	#pragma omp parallel for schedule(dynamic,1) collapse(2) shared(bundle)
	for (uint y = 0; y < bundle.size_y(); y++) {
		for (uint x = 0; x < bundle.size_x(); x++) {
			Point point(x, y);
			vector<NormalizedPatch> *normalized_patch = bundle.normalized_patch(point.x, point.y);
			normalize_patch_internal(bundle, point, *normalized_patch);
		}
	}
}

/* Private */

/**
 * Calculate patch distance using Gaussian weights.
 * @param normalized_source Set of candidate normalizations of the source patch.
 * @param normalized_target Set of candidate normalizations of the target patch.
 * @param source_id [out] Id of the candidate source normalization that gives the smallest distance.
 * @param target_id [out] Id of the candidate target normalization that gives the smallest distance.
 */
float AffinePatchDistance::calculate_gaussian(const vector<NormalizedPatch> &normalized_source,
											 const vector<NormalizedPatch> &normalized_target,
											 float radius,
											 int number_of_channels,
											 int &source_id,
											 int &target_id)
{
	int number_of_channels_used = (_reference_channel < 0 || _reference_channel >= number_of_channels)
								  ? number_of_channels : 1;
	double min_distance = std::numeric_limits<float>::max();
	target_id = -2;
	source_id = -2;

	// Calculate distance values for every combination of source and target normalizations
	for (uint i = 0; i < normalized_target.size(); i++) {
		for (uint j = 0; j < normalized_source.size(); j++) {
			float** target_patch = normalized_target[i].patch.get();
			float** source_patch = normalized_source[j].patch.get();

			double distance = 0.0;
			double total_weight = 0.0;

			// Calculate distances using either all channels or only the reference one, if it is specified
			if (_reference_channel < 0 || _reference_channel >= number_of_channels) {
				for (int k = 0; k < _grid->nodes_length; k++) {
					if (source_patch[0][k] < -256.0f ||
						target_patch[0][k] < -256.0f) {    // we cannot compare points, if at least one of them is unknown
						continue;
					}

					// Calculate color difference at k-th node
					double color_distance = 0.0;
					for (int ch = 0; ch < number_of_channels; ch++) {
						color_distance += (source_patch[ch][k] - target_patch[ch][k]) *
										  (source_patch[ch][k] - target_patch[ch][k]);
					}

					distance += _weights[k] * color_distance;
					total_weight += _weights[k];
				}
			} else {
				for (int k = 0; k < _grid->nodes_length; k++) {
					if (source_patch[0][k] < -256.0f ||
						target_patch[0][k] < -256.0f) {    // we cannot compare points, if at least one of them is unknown
						continue;
					}

					// Calculate color difference at k-th node
					double color_distance = (source_patch[_reference_channel][k] - target_patch[_reference_channel][k]) *
											(source_patch[_reference_channel][k] - target_patch[_reference_channel][k]);

					distance += _weights[k] * color_distance;
					total_weight += _weights[k];
				}
			}

			// Normalize
			if (total_weight > 0) {
				distance /= ((double)number_of_channels_used * total_weight);
			} else {
				distance = std::numeric_limits<float>::max();
			}

			// Select the smallest distance and keep its corresponding configuration of patches
			if (distance < min_distance) {
				min_distance = (float)distance;
				target_id = i;
				source_id = j;
			}
		}
	}

	return (float)min_distance;
}


/**
 * Calculate patch distance using approximated geodesic weights.
 * @param normalized_source Set of candidate normalizations of the source patch.
 * @param normalized_target Set of candidate normalizations of the target patch.
 * @param source_id [out] Id of the candidate source normalization that gives the smallest distance.
 * @param target_id [out] Id of the candidate target normalization that gives the smallest distance.
 */
float AffinePatchDistance::calculate_geodesic(const vector<NormalizedPatch> &normalized_source,
											 const vector<NormalizedPatch> &normalized_target,
											 float radius,
											 int number_of_channels,
											 int &source_id,
											 int &target_id)
{
	float **target_patch = normalized_target[0].patch.get();
	float **source_patch = normalized_source[0].patch.get();

	// Compute color component for bilateral weights (geodesic weights approximation)
	std::unique_ptr<float[]> central_color;
	if (_reference_channel < 0 || _reference_channel >= number_of_channels) {
		central_color.reset(new float[number_of_channels]);
		for (int ch = 0; ch < number_of_channels; ch++) {
			central_color[ch] = target_patch[ch][_grid->nodes_length / 2];
		}
	} else {
		central_color.reset(new float[1]);
		central_color[0] = target_patch[_reference_channel][_grid->nodes_length / 2];
	}
	float color_k = _bilateral_k_color / (2.0f * (radius / _scale) * (radius / _scale));

	int number_of_channels_used = (_reference_channel < 0 || _reference_channel >= number_of_channels)
								  ? number_of_channels : 1;
	double min_distance = std::numeric_limits<float>::max();
	target_id = -2;
	source_id = -2;

	for (uint i = 0; i < normalized_target.size(); i++) {
		for (uint j = 0; j < normalized_source.size(); j++) {
			target_patch = normalized_target[i].patch.get();
			source_patch = normalized_source[j].patch.get();

			double distance = 0.0;
			double total_weight = 0.0;

			// Calculate distances using either all channels or only the reference one, if it is specified
			if (_reference_channel < 0 || _reference_channel >= number_of_channels) {
				for (int k = 0; k < _grid->nodes_length; k++) {
					if (source_patch[0][k] < -256.0f ||
						target_patch[0][k] <
						-256.0f) {    // we cannot compare points, if at least one of them is unknown
						continue;
					}

					// Calculate color difference at k-th node
					double color_distance = 0.0;
					for (int ch = 0; ch < number_of_channels; ch++) {
						color_distance += (source_patch[ch][k] - target_patch[ch][k]) *
										  (source_patch[ch][k] - target_patch[ch][k]);
					}

					// Calculate color weight
					double central_distance = 0.0;
					for (int ch = 0; ch < number_of_channels; ch++) {
						central_distance += (central_color[ch] - target_patch[ch][k]) *
											(central_color[ch] - target_patch[ch][k]);
					}
					double color_weight = LUT::exp_rcn(-color_k * central_distance);

					distance += color_weight * _weights[k] * color_distance;
					total_weight += color_weight * _weights[k];
				}
			} else {
				for (int k = 0; k < _grid->nodes_length; k++) {
					if (source_patch[0][k] < -256.0f ||
						target_patch[0][k] <
						-256.0f) {    // we cannot compare points, if at least one of them is unknown
						continue;
					}

					// Calculate color difference at k-th node
					double color_distance = (source_patch[_reference_channel][k] - target_patch[_reference_channel][k]) *
											(source_patch[_reference_channel][k] - target_patch[_reference_channel][k]);

					// Calculate color weight
					double central_distance = (central_color[0] - target_patch[_reference_channel][k]) *
											  (central_color[0] - target_patch[_reference_channel][k]);
					double color_weight = LUT::exp_rcn(-color_k * central_distance);

					distance += color_weight * _weights[k] * color_distance;
					total_weight += color_weight * _weights[k];
				}
			}

			// Normalize
			if (total_weight > 0) {
				distance /= ((double) number_of_channels_used * total_weight);
			} else {
				distance = std::numeric_limits<float>::max();
			}

			// Select the smallest distance and keep its corresponding configuration of patches
			if (distance < min_distance) {
				min_distance = (float) distance;
				target_id = i;
				source_id = j;
			}
		}
	}

	return (float)min_distance;
}


void AffinePatchDistance::update_weights()
{
	// Recompute weights
	_weights.reset(calculate_weights(_grid.get(), _scale));
}


/**
 * @param sigma_factor - number of sigmas we want to be fitted within the radius
 */
float* AffinePatchDistance::calculate_weights(const GridInfo *grid, float sigma_factor)
{
	// NOTE: since transformations are normalized by the radius, we transform ellipses to unit circles.
	const float radius = 1.0f;

	float *weights = new float[grid->nodes_length];
	float sigma_squared = 2.0f * (radius / sigma_factor) * (radius / sigma_factor) / _bilateral_k_spatial;

	for (int i = 0; i < grid->nodes_length; i++) {
		float weight = exp(-(grid->nodes[i].x * grid->nodes[i].x + grid->nodes[i].y * grid->nodes[i].y) / sigma_squared);
		weights[i] = weight;
	}

	return weights;
}


inline void AffinePatchDistance::normalize_patch_internal(const StructureTensorBundle &bundle,
														  Point point,
														  std::vector<NormalizedPatch> &normalized_patch)
{
	// Compute dominant orientations
	Matrix2f transformation = bundle.transform(point);
	vector<Point> region = bundle.region(point);
	vector<float> dominant_orientations = _normalization.calculate_dominant_orientations(bundle.gradient_x(),
																						 bundle.gradient_y(),
																						 region,
																						 transformation,
																						 point);

	// For every dominant orientation compute its corresponding patch normalization
	for (auto it = dominant_orientations.begin(); it != dominant_orientations.end(); ++it) {
		Matrix2f rotation = _normalization.rotation(*it);
		std::shared_ptr<float*> normalization = _normalization.interpolate_to_grid(*_grid,
																				   bundle.image(),
																				   bundle.mask(),
																				   Matrix::multiply(rotation,
																									transformation),
																				   point);
		normalized_patch.push_back(NormalizedPatch(normalization, transformation, rotation));
	}
}

}	// namespace msas
