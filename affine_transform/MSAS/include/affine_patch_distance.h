/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef AFFINE_PATCH_DISTANCE_H_
#define AFFINE_PATCH_DISTANCE_H_

#include <memory>
#include "ellipse_normalization.h"
#include "distance_info.h"
#include "structure_tensor_bundle.h"
#include "point.h"
#include "matrix.h"

namespace msas
{

/**
 * Specific implementation of affine invariant patch distance calculator that
 * uses intermediate regular grid to compare two patches. Dominant orientations
 * (as in SIFT) are computed to determine the additional rotation between patches.
 */
class AffinePatchDistance
{
public:
	AffinePatchDistance(int grid_size);
	~AffinePatchDistance();

	/// Compute patch distance between two given points.
	/// @param source_bundle Bundle embedding the first image and its corresponding structure tensor field.
	/// @param source_point Point of interest in the first image.
	/// @param target_bundle Bundle embedding the second image and its corresponding structure tensor field.
	/// @param target_point Point of interest in the second image.
	/// @note @param source_bundle and @param target_bundle may coincide.
	DistanceInfo calculate(const StructureTensorBundle &source_bundle,
						   Point source_point,
						   const StructureTensorBundle &target_bundle,
						   Point target_point);

	/// Get scale parameter (relative scale w.r.t. the radius)
	float scale();

	/// Set scale parameter (relative scale w.r.t. the radius)
	void set_scale(float value);

	/// Get kappa-color for bilateral weights
	float bilateral_k_color();

	/// Set kappa-color for bilateral weights
	void set_bilateral_k_color(float value);

	/// Get kappa-spatial for bilateral weights
	float bilateral_k_spatial();

	/// Set kappa-spatial for bilateral weights
	void set_bilateral_k_spatial(float value);

	/// Get the regular grid that is used in distance computation.
	std::shared_ptr<GridInfo> grid();

	/// Get number of points in a normalized patch that is used in distance computation.
	size_t normalized_patch_length();

	/// Get size (resolution) of the regular grid.
	int grid_size();

	/// Set size (resolution) of the regular grid.
	/// @note Causes recreation of the grid.
	void set_grid_size(int value);

	/// Get id of the reference channel.
	int reference_channel() const;

	/// Set id of a channel to be used to calculate distance between two patches. By default set to -1.
	/// If the value is out of range (e.g. -1), then all channel are used.
	void set_reference_channel(int value);

	/// Specify whether normalized patches should be cached or not (true by default).
	void set_use_cache(bool value);

    void precompute_normalized_patches(const StructureTensorBundle &bundle);

private:
	static constexpr float EPS = 0.0001f;

	EllipseNormalization _normalization;
	std::shared_ptr<GridInfo> _grid;	// Note: we normalize patches to unit circles, so no need for two grids
	std::unique_ptr<float[]> _weights;

	float _scale;
	float _bilateral_k_color;
	float _bilateral_k_spatial;
	bool _use_bilateral;
	int _grid_size;
	int _reference_channel;
	bool _use_cache;

	float calculate_gaussian(const std::vector<NormalizedPatch> &normalized_source,
							 const std::vector<NormalizedPatch> &normalized_target,
							 float radius,
							 int number_of_channels,
							 int &source_id,
							 int &target_id);

	float calculate_geodesic(const std::vector<NormalizedPatch> &normalized_source,
							 const std::vector<NormalizedPatch> &normalized_target,
							 float radius,
							 int number_of_channels,
							 int &source_id,
							 int &target_id);

	void update_weights();

	float* calculate_weights(const GridInfo *grid, float sigma_factor);

	inline void normalize_patch_internal(const StructureTensorBundle &bundle,
										 Point point,
										 std::vector<NormalizedPatch> &normalized_patch);
};

}	// namespace msas

#endif /* AFFINE_PATCH_DISTANCE_H_ */
