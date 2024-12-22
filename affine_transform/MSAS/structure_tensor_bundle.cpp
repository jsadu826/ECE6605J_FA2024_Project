/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include <io_utility.h>
#include "structure_tensor_bundle.h"

using std::vector;

namespace msas
{

StructureTensorBundle::StructureTensorBundle()
: _structure_tensor(0)
{
	_data = vector<DataEntry* >();
}

StructureTensorBundle::StructureTensorBundle(const ImageFx<float> &image,
											 const StructureTensor &structure_tensor,
											 const MaskFx &mask)
: _structure_tensor(structure_tensor),
  _image(image)
{
	_size_x = _image.size_x();
	_size_y = _image.size_y();

	if (!mask.is_empty() && mask.size() == _image.size()) {
		_mask = mask;
	} else {
		_mask = MaskFx();
	}

	_data = vector<DataEntry* >(_size_x * _size_y, (DataEntry*)0);

	_normalized_patches_cache = Image<vector<NormalizedPatch> >(_image.size());
}


StructureTensorBundle::StructureTensorBundle(const StructureTensorBundle &other)
: _structure_tensor(other._structure_tensor),
  _image(other._image),
  _mask(other._mask),
  _size_x(other._size_x),
  _size_y(other._size_y)
{
	_data = vector<DataEntry* >(_size_x * _size_y, (DataEntry*)0);

	_normalized_patches_cache = Image<vector<NormalizedPatch> >(_size_x, _size_y);
}


StructureTensorBundle::~StructureTensorBundle()
{
	for(auto it = _data.begin(); it != _data.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}
}


ImageFx<float> StructureTensorBundle::image() const
{
	return _image;
}


MaskFx StructureTensorBundle::mask() const
{
	return _mask;
}


Image<float> StructureTensorBundle::gradient_x() const
{
	if (_gradient_x.is_empty()) {
		calculate_gradient();
	}

	return _gradient_x;
}


Image<float> StructureTensorBundle::gradient_y() const
{
	if (_gradient_y.is_empty()) {
		calculate_gradient();
	}

	return _gradient_y;
}


int StructureTensorBundle::size_x() const
{
	return _size_x;
}


int StructureTensorBundle::size_y() const
{
	return _size_y;
}


Shape StructureTensorBundle::size() const
{
	return Shape(_size_x, _size_y);
}


Matrix2f StructureTensorBundle::tensor(int x, int y) const
{
	if (!is_in_range(x, y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return data->tensor;
}


Matrix2f StructureTensorBundle::tensor(Point p) const
{
	if (!is_in_range(p.x, p.y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(p.x, p.y);
	return data->tensor;
}


vector<Point> StructureTensorBundle::region(int x, int y) const
{
	if (!is_in_range(x, y)) {
		return vector<Point>();
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return _structure_tensor.calculate_region(data->tensor, Point(x, y), _image.size());
}


vector<Point> StructureTensorBundle::region(Point p) const
{
	if (!is_in_range(p.x, p.y)) {
		return vector<Point>();
	}

	DataEntry *data = get_or_calculate_data(p.x, p.y);
	return _structure_tensor.calculate_region(data->tensor, p, _image.size());
}


vector<Point> StructureTensorBundle::region(int x, int y, float radius) const
{
	if (!is_in_range(x, y)) {
		return vector<Point>();
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return _structure_tensor.calculate_region(data->tensor, Point(x, y), _image.size(), radius);
}


vector<Point> StructureTensorBundle::region(Point p, float radius) const
{
	if (!is_in_range(p.x, p.y)) {
		return vector<Point>();
	}

	DataEntry *data = get_or_calculate_data(p.x, p.y);
	return _structure_tensor.calculate_region(data->tensor, p, _image.size(), radius);
}


std::unique_ptr<float[]> StructureTensorBundle::weights(const vector<Point> &region,
														Point center,
														float radius,
														float sigma_factor) const
{
	if (!is_in_range(center.x, center.y)) {
		return std::unique_ptr<float[]>();
	}

	DataEntry *data = get_or_calculate_data(center.x, center.y);
	return _structure_tensor.calculate_weights(region, data->tensor, center, radius, sigma_factor);
}


Matrix2f StructureTensorBundle::transform(int x, int y) const
{
	if (!is_in_range(x, y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return data->transform;
}


Matrix2f StructureTensorBundle::transform(Point p) const
{
	if (!is_in_range(p.x, p.y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(p.x, p.y);
	return data->transform;
}


Matrix2f StructureTensorBundle::sqrt(int x, int y) const
{
	if (!is_in_range(x, y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return _structure_tensor.sqrt(data->tensor);
}


Matrix2f StructureTensorBundle::sqrt(Point p) const
{
	if (!is_in_range(p.x, p.y)) {
		return Matrix::zero();
	}

	DataEntry *data = get_or_calculate_data(p.x, p.y);
	return _structure_tensor.sqrt(data->tensor);
}


float StructureTensorBundle::angle(int x, int y) const
{
	if (!is_in_range(x, y)) {
		return 0;
	}

	DataEntry *data = get_or_calculate_data(x, y);
	return data->angle;
}


float StructureTensorBundle::angle(Point p) const
{
	return angle(p.x, p.y);
}


bool StructureTensorBundle::mask_contains(int x, int y) const
{
	return is_in_mask(x, y);
}


bool StructureTensorBundle::mask_contains(Point p) const
{
	return is_in_mask(p.x, p.y);
}


float StructureTensorBundle::radius() const
{
	return _structure_tensor.radius();
}


void StructureTensorBundle::set_radius(float value)
{
	if (value != _structure_tensor.radius()) {
		_structure_tensor.set_radius(value);

		drop_cache();
	}
}


void StructureTensorBundle::populate_gradient_cache(const Image<float> &gradient_x, const Image<float> &gradient_y) const
{
	if ( _image.size() == gradient_x.size() &&
		 _image.size() == gradient_y.size() ) {
		_gradient_x = gradient_x;
		_gradient_y = gradient_y;

		calculate_dyadics();
	}
}


void StructureTensorBundle::drop_cache()
{
	for (auto it = _data.begin(); it != _data.end(); ++it) {
		if (*it) {
			delete *it;
		}
	}

	_data.clear();
	_data = vector<DataEntry* >(_size_x * _size_y, (DataEntry*)0);
}


vector<NormalizedPatch>* StructureTensorBundle::normalized_patch(int x, int y) const
{
	return _normalized_patches_cache.raw() + (y * _size_x + x);
}


/* Private */

DataEntry* StructureTensorBundle::get_or_calculate_data(int x, int y) const
{
	DataEntry *data = _data[index(x, y)];
	if (!data) {
		data = calculate_data(x, y);
		_data[index(x, y)] = data;
	}

	return data;
}

/**
 * [Re]Calculates gradient for the whole sequence.
 */
void StructureTensorBundle::calculate_gradient() const
{
	if (!_image) {
		return;
	}

	if (_gradient_x.is_empty()) {
		_gradient_x = Image<float>(_image.size(), 0.0f);
	}

	if (_gradient_y.is_empty()) {
		_gradient_y = Image<float>(_image.size(), 0.0f);
	}

	// Convert image to gray, if it is multichannel
	Image<float> image = (_image.number_of_channels() != 1) ? IOUtility::to_mono(_image) : _image;

	FieldOperations::centered_gradient(image.raw(),
									   _gradient_x.raw(),
									   _gradient_y.raw(),
									   _image.size_x(),
									   _image.size_y());
}


/**
 * Precompute and store dyadic products of gradient vectors
 */
void StructureTensorBundle::calculate_dyadics() const
{
	_dyadics = Image<float>(_gradient_x.size(), (uint)3);
	int size_x = _gradient_x.size_x();
	int size_y = _gradient_x.size_y();
	float *dyadics_data = _dyadics.raw();
	float *grad_x_data = _gradient_x.raw();
	float *grad_y_data = _gradient_y.raw();
	for (uint y = 0; y < size_y; ++y) {
		for (uint x = 0; x < size_x; ++x) {
			int index = size_x * y + x;
			dyadics_data[index * 3] = 		grad_x_data[index] * grad_x_data[index];
			dyadics_data[index * 3 + 1] = 	grad_x_data[index] * grad_y_data[index];
			dyadics_data[index * 3 + 2] = 	grad_y_data[index] * grad_y_data[index];
		}
	}
}


/**
 * Calculate structure tensor and transform at the given point.
 */
DataEntry* StructureTensorBundle::calculate_data(int x, int y) const
{
    #pragma omp critical (GRADIENT)
	if (_gradient_x.is_empty()) {
        calculate_gradient();
		calculate_dyadics();
	}

	DataEntry *data = new DataEntry();

//	data->tensor = _structure_tensor.calculate_stabilized(_gradient_x, _gradient_y, Point(x, y), _mask);
	data->tensor = _structure_tensor.calculate(_dyadics, Point(x, y), _mask);
	data->transform = _structure_tensor.calculate_transformation(data->tensor, data->angle);

	return data;
}


/**
 * Convert x and y coordinates into the linear index
 */
inline int StructureTensorBundle::index(int x, int y) const
{
	return _size_x * y + x;
}


/**
 * Check that a point is inside the image domain.
 */
inline bool StructureTensorBundle::is_in_range(uint x, uint y) const
{
	return x < _size_x && y < _size_y;
}


/**
 * Check that a point belongs to the mask.
 */
inline bool StructureTensorBundle::is_in_mask(int x, int y) const
{
	return (_mask && _mask.test(x, y)) ||
		   (!_mask && x >= 0 && x < _size_x && y >= 0 && y < _size_y);
}

}	// namespace msas
