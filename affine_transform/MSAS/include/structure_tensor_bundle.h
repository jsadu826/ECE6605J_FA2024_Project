/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef STRUCTURE_TENSOR_BUNDLE_H_
#define STRUCTURE_TENSOR_BUNDLE_H_


#include <vector>
#include "structure_tensor.h"
#include "image.h"
#include "mask.h"
#include "field_operations.h"
#include "normalized_patch.h"

namespace msas
{

struct DataEntry;

/**
 * Represents a field of structure tensors computed on the corresponding image.
 * Structure tensors are computed on-demand and then stored in an internal cache for later access.
 * Computation is delegated to an instance of the StructureTensor calculator.
 * Optional mask, specifying available portion of the image, can also be provided.
 * For convenience the underlying data (image, mask, gradients, etc.) is also accessible.
 */
class StructureTensorBundle
{
public:
	StructureTensorBundle();
	StructureTensorBundle(const ImageFx<float> &image,
						  const StructureTensor &structure_tensor,
						  const MaskFx &mask = MaskFx());
	StructureTensorBundle(const StructureTensorBundle &other);
	~StructureTensorBundle();

	/// Get embedded image.
	ImageFx<float> image() const;

	/// Get embedded mask.
	MaskFx mask() const;

	/// Get computed gradient.
	Image<float> gradient_x() const;
	Image<float> gradient_y() const;

	/// Get size of the field.
	int size_x() const;
	int size_y() const;
	Shape size() const;

	/// Get or calculate structure tensor at the given point.
	/// @return Structure tensor when the point is within the domain, zero matrix otherwise.
	Matrix2f tensor(int x, int y) const;
	Matrix2f tensor(Point p) const;

	/// Calculate elliptical regions at the given point using internal radius.
	/// @return Elliptical region when the point is within the domain, empty set otherwise.
	std::vector<Point> region(int x, int y) const;
	std::vector<Point> region(Point p) const;

	/// Calculate elliptical regions at the given point using given radius.
	/// @return Elliptical region when the point is within the domain, empty set otherwise.
	std::vector<Point> region(int x, int y, float radius) const;
	std::vector<Point> region(Point p, float radius) const;

	/// Calculate intra-patch anisotropic Gaussian weights for an elliptical region.
	/// @param region Set of points (normally shape-adaptive patch).
	/// @param center Central point of the elliptical region.
	/// @param radius [optional] Value of R parameter that overwrites the internal value in this particular call.
	/// @param sigma_factor [optional] Coefficient for the Gaussian sigma: sigma = radius / sigma_factor.
	std::unique_ptr<float[]> weights(const std::vector<Point> &region,
									 Point center,
									 float radius = -1.0f,
									 float sigma_factor = 3.0f) const;

	/// Get or calculate transformation matrix at the given point.
	/// @return Transformation matrix when the point is within the domain, zero matrix otherwise.
	Matrix2f transform(int x, int y) const;
	Matrix2f transform(Point p) const;

	/// Calculate square root of a structure tensor at the given point.
	/// @return Square root of the structure tensor when the point is within the domain, zero matrix otherwise.
	Matrix2f sqrt(int x, int y) const;
	Matrix2f sqrt(Point p) const;

	/// Get or calculate rotation angle associated with a tensor at the given point.
	/// @return Angle when the point is within the domain, zero otherwise.
	float angle(int x, int y) const;
	float angle(Point p) const;

	/// Check if the given point belongs to the mask.
	bool mask_contains(int x, int y) const;
	bool mask_contains(Point p) const;

	/// Get/set internal value of R (radius) parameter.
	float radius() const;
	void set_radius(float value);

	void populate_gradient_cache(const Image<float> &gradient_x, const Image<float> &gradient_y) const;

	/// Clear internal cache, containing already computed structure tensors.
	void drop_cache();

	/// Get a pointer to a set of patch normalizations at a given location.
	/// @note Memory allocated for these sets is managed internally, do not attempt to release it.
	std::vector<NormalizedPatch>* normalized_patch(int x, int y) const;

private:
	StructureTensor _structure_tensor;
	ImageFx<float> _image;
	MaskFx _mask;
	int _size_x, _size_y;
	mutable Image<float> _gradient_x, _gradient_y;
	mutable Image<float> _dyadics;
	mutable std::vector<DataEntry* > _data;
	mutable Image<std::vector<NormalizedPatch > > _normalized_patches_cache;

	DataEntry* get_or_calculate_data(int x, int y) const;
	void calculate_gradient() const;
	void calculate_dyadics() const;
	DataEntry* calculate_data(int x, int y) const;

	inline int index(int x, int y) const;
	inline bool is_in_range(uint x, uint y) const;
	inline bool is_in_mask(int x, int y) const;
};

// NOTE: this structure could not be within the StructureTensorBundle due to the EIGEN_MAKE_ALIGNED_OPERATOR_NEW
struct DataEntry
{
	Matrix2f tensor;
	Matrix2f transform;
	float angle;
};

}	// namespace msas

#endif /* STRUCTURE_TENSOR_BUNDLE_H_ */
