/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "structure_tensor.h"

using std::vector;

namespace msas
{

StructureTensor::StructureTensor(float radius, int iterations_amount, float gamma)
		: _radius(radius),
		  _iterations_amount(iterations_amount),
		  _gamma(gamma),
		  _max_size_limit(DEFAULT_SIZE_LIMIT),
		  _variation_threshold(DEFAULT_VARIATION_THRESHOLD)
{
	configure();
}


StructureTensor::StructureTensor(float radius, int iterations_amount)
		: _radius(radius),
		  _iterations_amount(iterations_amount),
		  _gamma(DEFAULT_GAMMA),
		  _max_size_limit(DEFAULT_SIZE_LIMIT),
		  _variation_threshold(DEFAULT_VARIATION_THRESHOLD)
{
	configure();
}


StructureTensor::StructureTensor(float radius)
		: _radius(radius),
		  _iterations_amount(DEFAULT_ITERATIONS_AMOUNT),
		  _gamma(DEFAULT_GAMMA),
		  _max_size_limit(DEFAULT_SIZE_LIMIT),
		  _variation_threshold(DEFAULT_VARIATION_THRESHOLD)
{
	configure();
}


StructureTensor::StructureTensor()
	: _radius(DEFAULT_RADIUS),
	  _iterations_amount(DEFAULT_ITERATIONS_AMOUNT),
	  _gamma(DEFAULT_GAMMA),
	  _max_size_limit(DEFAULT_SIZE_LIMIT),
	  _variation_threshold(DEFAULT_VARIATION_THRESHOLD)
{
	configure();
}


StructureTensor::StructureTensor(const StructureTensor &other)
	: _radius(other._radius),
	  _iterations_amount(other._iterations_amount),
	  _gamma(other._gamma),
	  _max_size_limit(other._max_size_limit),
	  _variation_threshold(other._variation_threshold)
{
	configure();
}


StructureTensor& StructureTensor::operator= (const StructureTensor &other)
{
	_radius = other._radius;
	_iterations_amount = other._iterations_amount;
	_gamma = other._gamma;
	_max_size_limit = other._max_size_limit;
	_variation_threshold = other._variation_threshold;

	configure();
}


Matrix2f StructureTensor::calculate(const ImageFx<float> &grad_x,
									const ImageFx<float> &grad_y,
									const Point &point,
									const MaskFx &mask) const
{
	Shape size = grad_x.size();
	const bool* mask_data = (mask) ? mask.raw() : 0;

	// Define functor for computing an initial structure tensor
	CalcFirstFunc calc_first = [&] (Point p) {
		return calculate_initial_tensor(grad_x.raw(), grad_y.raw(), mask_data, size.size_x, size.size_y, _radius, p);
	};

	// Define functor for iterative computation of a structure tensor
	CalcNextFunc  calc_next = [&] (Point p, Matrix2f tensor) {
		return calculate_next_tensor(grad_x.raw(), grad_y.raw(), mask_data, size.size_x, size.size_y, _radius, p,
									 tensor);
	};

	return _run_scheme_func(calc_first, calc_next, point);
}


Matrix2f StructureTensor::calculate(const ImageFx<float> &dyadics,
									const Point &point,
									const MaskFx &mask) const
{
	Shape size = dyadics.size();
	const bool* mask_data = (mask) ? mask.raw() : 0;

	// Define functor for computing an initial structure tensor
	CalcFirstFunc calc_first = [&] (Point p) {
		return calculate_initial_tensor(dyadics.raw(), mask_data, size.size_x, size.size_y, _radius, p);
	};

	// Define functor for iterative computation of a structure tensor
	CalcNextFunc  calc_next = [&] (Point p, Matrix2f tensor) {
		return calculate_next_tensor(dyadics.raw(), mask_data, size.size_x, size.size_y, _radius, p, tensor);
	};

	return _run_scheme_func(calc_first, calc_next, point);
}


Image<Matrix2f> StructureTensor::calculate(const ImageFx<float> &grad_x,
										   const ImageFx<float> &grad_y,
										   const MaskFx &mask) const
{
	Shape size = grad_x.size();
	const bool* mask_data = (mask) ? mask.raw() : 0;

	// Define functor for computing an initial structure tensor
	CalcFirstFunc calc_first = [&] (Point p) {
		return calculate_initial_tensor(grad_x.raw(), grad_y.raw(), mask_data, size.size_x, size.size_y, _radius, p);
	};

	// Define functor for iterative computation of a structure tensor
	CalcNextFunc  calc_next = [&] (Point p, Matrix2f tensor) {
		return calculate_next_tensor(grad_x.raw(), grad_y.raw(), mask_data, size.size_x, size.size_y, _radius, p,
									 tensor);
	};

	// Compute structure tensors for all points in the image
	Image<Matrix2f> tensors(size.size_x, size.size_y);
	for (uint y = 0; y < size.size_y; y++) {
		for (uint x = 0; x < size.size_x; x++) {
			Point p(x, y);
			tensors(p) = _run_scheme_func(calc_first, calc_next, p);
		}
	}

	return tensors;
}


Image<Matrix2f> StructureTensor::calculate(const ImageFx<float> &dyadics,
										   const MaskFx &mask) const
{
	Shape size = dyadics.size();
	const bool* mask_data = (mask) ? mask.raw() : 0;

	// Define functor for computing an initial structure tensor
	CalcFirstFunc calc_first = [&] (Point p) {
		return calculate_initial_tensor(dyadics.raw(), mask_data, size.size_x, size.size_y, _radius, p);
	};

	// Define functor for iterative computation of a structure tensor
	CalcNextFunc  calc_next = [&] (Point p, Matrix2f tensor) {
		return calculate_next_tensor(dyadics.raw(), mask_data, size.size_x, size.size_y, _radius, p, tensor);
	};

	// Compute structure tensors for all points in the image
	Image<Matrix2f> tensors(size.size_x, size.size_y);
	for (uint y = 0; y < size.size_y; y++) {
		for (uint x = 0; x < size.size_x; x++) {
			Point p(x, y);
			tensors(p) = _run_scheme_func(calc_first, calc_next, p);
		}
	}

	return tensors;
}


Matrix2f StructureTensor::calculate(const ImageFx<float> &grad_x,
									const ImageFx<float> &grad_y,
									const vector<Point> &region,
									const MaskFx &mask) const
{
	// Get raw pointers
	const float *grad_x_data = grad_x.raw();
	const float *grad_y_data = grad_y.raw();
	const bool *mask_data = (mask) ? mask.raw() : 0;

	// Compute sum of dyadic products
	double a = 0.0, bc = 0.0, d = 0.0;
	long normalizer = 0;
	uint size_x = grad_x.size_x();
	for (auto it = region.begin(); it != region.end(); ++it) {
		int index = it->y * size_x + it->x;
		if (!mask_data || mask_data[index]) {
			a += grad_x_data[index] * grad_x_data[index];
			bc += grad_x_data[index] * grad_y_data[index];
			d += grad_y_data[index] * grad_y_data[index];
			normalizer += 1;
		}
	}

	// Normalize
	a /= (double)normalizer;
	bc /= (double)normalizer;
	d /= (double)normalizer;

	Matrix2f tensor;
	tensor[0] = (float)a;
	tensor[1] = tensor[2] = (float)bc;
	tensor[3] = (float)d;

	return tensor;
}


Matrix2f StructureTensor::calculate(const ImageFx<float> &dyadics,
									const vector<Point> &region,
									const MaskFx &mask) const
{
	// Get raw pointers
	const float *diadics_data = dyadics.raw();
	const bool *mask_data = (mask) ? mask.raw() : 0;

	// Compute sum of dyadic products
	double a = 0.0, bc = 0.0, d = 0.0;
	long normalizer = 0;
	uint size_x = dyadics.size_x();
	for (auto it = region.begin(); it != region.end(); ++it) {
		int index = it->y * size_x + it->x;
		if (!mask_data || mask_data[index]) {
			a += diadics_data[index * 3];
			bc += diadics_data[index * 3 + 1];
			d += diadics_data[index * 3 + 2];
			normalizer += 1;
		}
	}

	// Normalize
	a /= (double)normalizer;
	bc /= (double)normalizer;
	d /= (double)normalizer;

	Matrix2f tensor;
	tensor[0] = (float)a;
	tensor[1] = tensor[2] = (float)bc;
	tensor[3] = (float)d;

	return tensor;
}


vector<Point> StructureTensor::calculate_initial_region(const ImageFx<float> &grad_x,
														const ImageFx<float> &grad_y,
														const Point &point,
														const MaskFx &mask,
														float radius) const
{
	const int margin = 1;

	// If radius parameter is not set, use internal value
	if (radius < 0.0f) {
		radius = _radius;
	}

	vector<Point> region;

	// Get raw pointers and size
	Shape size = grad_x.size();
	const float *grad_x_data = grad_x.raw();
	const float *grad_y_data = grad_y.raw();
	const bool *mask_data = (mask) ? mask.raw() : 0;

	// Get gradient vector at the central point
	int index_at_center = point.y * size.size_x + point.x;
	float grad_x_at_center = grad_x_data[index_at_center];
	float grad_y_at_center = grad_y_data[index_at_center];

	// Calculate possible limits in Y dimension
	int y_lower, y_upper;
	if (std::abs(grad_y_at_center) > EPS) {
		float y_1 = (-radius - grad_x_at_center * (float) point.x) / -grad_y_at_center + point.y;
		float y_2 = (+radius - grad_x_at_center * (float) point.x) / -grad_y_at_center + point.y;
		float y_3 = (-radius - grad_x_at_center * (float) (size.size_x - point.x - 1)) / grad_y_at_center + point.y;
		float y_4 = (+radius - grad_x_at_center * (float) (size.size_x - point.x - 1)) / grad_y_at_center + point.y;
		y_lower = std::max(0, (int) std::min(std::min(y_1, y_2), std::min(y_3, y_4)));
		y_upper = std::min((int)size.size_y - 1, (int) (std::max(std::max(y_1, y_2), std::max(y_3, y_4)) + 0.5f));
	} else {
		y_lower = 0;
		y_upper = size.size_y - 1;
	}

	if (std::abs(grad_x_at_center) > EPS) {
		// Scan rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			// For every row compute possible limits in X dimension
			float x_1 = (-radius - grad_y_at_center * (float) (y - point.y)) / grad_x_at_center + point.x;
			float x_2 = (+radius - grad_y_at_center * (float) (y - point.y)) / grad_x_at_center + point.x;
			long x_lower = std::max(0, (int) std::min(x_1, x_2) - margin);
			long x_upper = std::min((int)size.size_x - 1, (int) std::max(x_1, x_2) + margin);

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float d = grad_x_at_center * (float) (x_lower - point.x) + grad_y_at_center * (float) (y - point.y);
				if (abs(d) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float d = grad_x_at_center * (float) (x_upper - point.x) + grad_y_at_center * (float) (y - point.y);
				if (abs(d) < radius) {
					break;
				}
			}

			// Add points of y row between x_lower and x_upper to the region
			for (long x = x_lower; x <= x_upper; x++) {
				region.push_back(Point(x, y));
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	} else {
		// Scan complete rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			long x_lower = 0;
			long x_upper = size.size_x - 1;

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float d = grad_x_at_center * (float) (x_lower - point.x) + grad_y_at_center * (float) (y - point.y);
				if (abs(d) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float d = grad_x_at_center * (float) (x_upper - point.x) + grad_y_at_center * (float) (y - point.y);
				if (abs(d) < radius) {
					break;
				}
			}

			// Add points of y row between x_lower and x_upper to the region
			for (long x = x_lower; x <= x_upper; x++) {
				region.push_back(Point(x, y));
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	}

	return region;
}


vector<Point> StructureTensor::calculate_region(const Matrix2f &tensor,
												const Point &point,
												const Shape &size,
												float radius) const
{
	// If radius parameter is not set, use internal value
	if (radius < 0.0f) {
		radius = _radius;
	}

	vector<Point> region;

	// NOTE: we use tensor to locate two extreme points of the elliptical region in Y direction, we then
	//		 use tensor again to locate boundaries at every row
	double t_00 = tensor[0];
	double t_01 = tensor[1];
	double t_11 = tensor[3];

	// Ensure that tensor is positive definite and not too elongated
	double trace = t_00 + t_11;
	double det = t_00 * t_11 - t_01 * t_01;
	if (det <= 0.0 || trace * trace / det > EIGEN_RATIO_THRESHOLD) {
		region.push_back(point);
		return region;
	}

	// Locate extrema in Y direction
	double aux = std::sqrt(t_11 - t_01 * t_01 / t_00);
	double dy = radius / aux;	// dy > 0

	// Define aux terms
	double inv_t_00 = 1.0 / t_00;
	double a = t_01 * inv_t_00;
	double b = a * a - t_11 * inv_t_00;
	double c = radius * radius * inv_t_00;

	// Calculate limits in Y dimension
	int y_0 = std::max(point.y - (int)std::floor(dy), (int)0);
	int y_1 = std::min(point.y + (int)std::floor(dy), (int)size.size_y - 1);

	// Scan rows between y_0 and y_1
	for (int y = y_0; y <= y_1; ++y) {
		// For every row compute limits in X dimension
		double offset_y = y - point.y;
		double dis = std::sqrt(b * offset_y * offset_y + c);
		int x_0 = (int)std::ceil((double)point.x - a * offset_y - dis);
		int x_1 = (int)std::floor((double)point.x - a * offset_y + dis);
		x_0 = std::max(x_0, 0);
		x_1 = std::min(x_1, (int)size.size_x - 1);

		// Add points of y row between x_0 and x_1 to the region
		for (int x = x_0; x <= x_1; ++x) {
			region.push_back(Point(x, y));
		}
	}

	return region;
}


std::unique_ptr<float[]> StructureTensor::calculate_weights(const vector<Point> &region,
															const Matrix2f &tensor,
															const Point &center,
															float radius,
															float sigma_factor) const
{
	// If radius parameter is not set, use internal value
	if (radius < 0.0f) {
		radius = _radius;
	}

	long size = region.size();
	float *weights = new float[size];

	// Shortcut for degenerate ellipses
	if (size == 1) {
		weights[0] = 1.0f;
		return std::unique_ptr<float[]>(weights);
	}

	// For rather small sigma_factor, simply return equal weights
	if (sigma_factor < 0.0001f) {
		for (int i = 0; i < size; i++) {
			weights[i] = 1.0f / size;
		}

		return std::unique_ptr<float[]>(weights);
	}

	float sigma = radius / sigma_factor;
	float denominator = 2 * sigma * sigma;

	// For every point within the region compute its corresponding weight
	float total_value = 0.0f;
	for (int i = 0; i < size; ++i) {
		Point p = region[i];
		float d_x = center.x - p.x;
		float d_y = center.y - p.y;
		float enumerator = d_x * d_x * tensor[0] + 2 * d_x * d_y * tensor[1] + d_y * d_y * tensor[3];
		weights[i] = exp(-enumerator / denominator);
		total_value += weights[i];
	}

	// Normalize
	for (int i = 0; i < size; ++i) {
		weights[i] /= total_value;
	}

	return std::unique_ptr<float[]>(weights);
}


Matrix2f StructureTensor::calculate_transformation(const Matrix2f &tensor,
												   float &angle,
												   float radius) const
{
	// If radius parameter is not set, use internal value
	if (radius < 0.0f) {
		radius = _radius;
	}

	float a = tensor[0];
	float bc = tensor[2];
	float d = tensor[3];

	float trace = a + d;
	float aux = std::sqrt( (a - d) * (a - d) + 4 * bc * bc);

	if (!std::isfinite(aux) || aux == 0 || d == 0) {
		angle = 0;
		return Matrix::identity();
	}

	// Find eigenvalues
	float eigenvalue_1 = (trace + aux) / 2.0f;
	float eigenvalue_2 = std::abs(trace - aux) / 2.0f;

	// Find eigenvectors
	float eigenvector_1_x = bc;
	float eigenvector_1_y = eigenvalue_1 - a;
	float eigenvector_2_x = eigenvalue_2 - d;
	float eigenvector_2_y = bc;

	// Normalize eigenvectors
	float norm_1 = std::sqrt(eigenvector_1_x * eigenvector_1_x + eigenvector_1_y * eigenvector_1_y);
	float norm_2 = std::sqrt(eigenvector_2_x * eigenvector_2_x + eigenvector_2_y * eigenvector_2_y);
	eigenvector_1_x /= norm_1;
	eigenvector_1_y /= norm_1;
	eigenvector_2_x /= norm_2;
	eigenvector_2_y /= norm_2;

	radius = std::max(radius, 1.0f);

	// Calculate transform matrix T = sqrt(D) * U (and normalize it by the radius)
	Matrix2f transform;
	transform[0] = std::sqrt(eigenvalue_1) * eigenvector_1_x / radius;
	transform[1] = std::sqrt(eigenvalue_1) * eigenvector_1_y / radius;
	transform[2] = std::sqrt(eigenvalue_2) * eigenvector_2_x / radius;
	transform[3] = std::sqrt(eigenvalue_2) * eigenvector_2_y / radius;

	// Calculate the angle measured from the 0Y axis in the range [-p; p].
	// EXAMPLE: for a line from left-top to right-bottom, the angle will be either -pi/4, or 3pi/4 (depending on the rotation direction)
	angle = std::atan2(eigenvector_1_y, eigenvector_1_x);

	return transform;
}


Matrix2f StructureTensor::sqrt(const Matrix2f &tensor) const
{
	float a = tensor[0];
	float bc = tensor[2];
	float d = tensor[3];

	float trace = a + d;
	float aux = std::sqrt( (a - d) * (a - d) + 4 * bc * bc );

	if (!std::isfinite(aux) || aux == 0 || d == 0) {
		return Matrix::identity();
	}

	// Find eigenvalues
	float eigenvalue_1 = (trace + aux) / 2;
	float eigenvalue_2 = abs(trace - aux) / 2;

	// Find eigenvectors
	float eigenvector_1_x = bc;
	float eigenvector_1_y = eigenvalue_1 - a;
	float eigenvector_2_x = eigenvalue_2 - d;
	float eigenvector_2_y = bc;

	// Normalize eigenvectors
	float norm_1 = std::sqrt(eigenvector_1_x * eigenvector_1_x + eigenvector_1_y * eigenvector_1_y);
	float norm_2 = std::sqrt(eigenvector_2_x * eigenvector_2_x + eigenvector_2_y * eigenvector_2_y);
	eigenvector_1_x /= norm_1;
	eigenvector_1_y /= norm_1;
	eigenvector_2_x /= norm_2;
	eigenvector_2_y /= norm_2;

	// Calculate matrix T = U' * sqrt(D) * U
	Matrix2f result;
	result[0] = std::sqrt(eigenvalue_1) * eigenvector_1_x * eigenvector_1_x + std::sqrt(eigenvalue_2) * eigenvector_2_x * eigenvector_2_x;
	result[1] = std::sqrt(eigenvalue_1) * eigenvector_1_x * eigenvector_1_y + std::sqrt(eigenvalue_2) * eigenvector_2_x * eigenvector_2_y;
	result[2] = std::sqrt(eigenvalue_1) * eigenvector_1_x * eigenvector_1_y + std::sqrt(eigenvalue_2) * eigenvector_2_x * eigenvector_2_y;
	result[3] = std::sqrt(eigenvalue_1) * eigenvector_1_y * eigenvector_1_y + std::sqrt(eigenvalue_2) * eigenvector_2_y * eigenvector_2_y;

	return result;
}


float StructureTensor::radius() const
{
	return _radius;
}


void StructureTensor::set_radius(float value)
{
	_radius = value;
}


float StructureTensor::gamma() const
{
	return _gamma;
}


void StructureTensor::set_gamma(float value)
{
	_gamma = value;

	configure();
}


int StructureTensor::iterations_amount() const
{
	return _iterations_amount;
}


void StructureTensor::set_iterations_amount(int value)
{
	_iterations_amount = value;
}


float StructureTensor::max_size_limit() const
{
    return _max_size_limit;
}


void StructureTensor::set_max_size_limit(float value)
{
    _max_size_limit = value;
}

/* Private */

void StructureTensor::configure()
{
	// Choose one of two schemes depending on the value of _gamma
	if (_gamma > 0.0f && _gamma < 1.0f) {
		_run_scheme_func = [this] (CalcFirstFunc &calc_first, CalcNextFunc &calc_next, const Point &point) {
			return run_stabilized_scheme(calc_first, calc_next, point);
		};
	} else {
		_run_scheme_func = [this] (CalcFirstFunc &calc_first, CalcNextFunc &calc_next, const Point &point) {
			return run_original_scheme(calc_first, calc_next, point);
		};
	}
}


inline Matrix2f StructureTensor::run_original_scheme(CalcFirstFunc &calc_first,
													 CalcNextFunc &calc_next,
													 const Point &point) const
{
	// Calculate structure tensor at the first iteration
	Matrix2f tensor = calc_first(point);

	for (int i = 1; i < _iterations_amount; i++) {
		Matrix2f next_tensor = calc_next(point, tensor);

		// Compute the difference
		float aux_a = next_tensor[0] - tensor[0];
		float aux_bc = next_tensor[1] - tensor[1];
		float aux_d = next_tensor[3] - tensor[3];

		// Stop if tensor has converged
		float variation = aux_a * aux_a + 2 * aux_bc * aux_bc + aux_d * aux_d;
		if (variation < _variation_threshold) {
			tensor = next_tensor;
			break;
		}

		tensor = next_tensor;
	}

	return tensor;
}


inline Matrix2f StructureTensor::run_stabilized_scheme(CalcFirstFunc &calc_first,
													   CalcNextFunc &calc_next,
													   const Point &point) const
{
	// NOTE: the following three constants (5, 2.0 and 0.0001) were picked experimentally
	float gamma = _gamma;
	int gamma_decrease_step = std::max(_iterations_amount / 5, 1);
	float gamma_divider = 2.0f;

	// Calculate tensor of the first iteration
	Matrix2f tensor = calc_first(point);

	Matrix2f proposed_tensor;
	for (int i = 1; i < _iterations_amount; i++) {
		proposed_tensor = calc_next(point, tensor);

		// Update tensor using its previous value and the proposed tensor
		float aux_a = proposed_tensor[0] - tensor[0];
		float aux_bc = proposed_tensor[1] - tensor[1];
		float aux_d = proposed_tensor[3] - tensor[3];
		tensor[0] = tensor[0] + gamma * aux_a;
		tensor[1] = tensor[1] + gamma * aux_bc;
		tensor[2] = tensor[2] + gamma * aux_bc;
		tensor[3] = tensor[3] + gamma * aux_d;

		// Stop if tensor has converged
		float variation = aux_a * aux_a + 2 * aux_bc * aux_bc + aux_d * aux_d;
		if (variation < _variation_threshold) {
			break;
		}

		// Reduce 'gamma' by the 'gamma_divider' factor every 'gamma_decrease_step' iterations
		if (i % gamma_decrease_step == 0) {
			gamma /= gamma_divider;
		}
	}

	return tensor;
}


inline Matrix2f StructureTensor::calculate_initial_tensor(const float *grad_x,
														  const float *grad_y,
														  const bool *mask,
														  int size_x,
														  int size_y,
														  float radius,
														  const Point &center) const
{
	int index_at_center = center.y * size_x + center.x;
	float grad_x_at_center = grad_x[index_at_center];
	float grad_y_at_center = grad_y[index_at_center];
	const int margin = 1;
	double a = 0.0, bc = 0.0, d = 0.0;
	long normalizer = 0;

	// Calculate possible limits in Y dimension
	long y_lower, y_upper;
	if (std::abs(grad_y_at_center) > EPS) {
		float y_1 = (-radius - grad_x_at_center * (float) center.x) / -grad_y_at_center + center.y;
		float y_2 = (+radius - grad_x_at_center * (float) center.x) / -grad_y_at_center + center.y;
		float y_3 = (-radius - grad_x_at_center * (float) (size_x - center.x - 1)) / grad_y_at_center + center.y;
		float y_4 = (+radius - grad_x_at_center * (float) (size_x - center.x - 1)) / grad_y_at_center + center.y;
		y_lower = std::max(0, (int) std::min(std::min(y_1, y_2), std::min(y_3, y_4)));
		y_upper = std::min(size_y - 1, (int) (std::max(std::max(y_1, y_2), std::max(y_3, y_4)) + 0.5f));
	} else {
		y_lower = 0;
		y_upper = size_y - 1;
	}

	if (std::abs(grad_x_at_center) > EPS) {
		// Scan rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			// For every row compute possible limits in X dimension
			float x_1 = (-radius - grad_y_at_center * (float) (y - center.y)) / grad_x_at_center + center.x;
			float x_2 = (+radius - grad_y_at_center * (float) (y - center.y)) / grad_x_at_center + center.x;
			long x_lower = std::max(0, (int) std::min(x_1, x_2) - margin);
			long x_upper = std::min(size_x - 1, (int) std::max(x_1, x_2) + margin);

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float dist = grad_x_at_center * (float) (x_lower - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float dist = grad_x_at_center * (float) (x_upper - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Compute and aggregate dyadic products at the points of y row between x_lower and x_upper
			for (long x = x_lower; x <= x_upper; x++) {
				long index = y * size_x + x;
				if (!mask || mask[index]) {
					a += grad_x[index] * grad_x[index];
					bc += grad_x[index] * grad_y[index];
					d += grad_y[index] * grad_y[index];
					normalizer += 1;
				}
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	} else {    // grad_x_at_center <= EPS
		// Scan complete rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			long x_lower = 0;
			long x_upper = size_x - 1;

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float dist = grad_x_at_center * (float) (x_lower - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float dist = grad_x_at_center * (float) (x_upper - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Compute and aggregate dyadic products at the points of y row between x_lower and x_upper
			for (long x = x_lower; x <= x_upper; x++) {
				long index = y * size_x + x;
				if (!mask || mask[index]) {
					a += grad_x[index] * grad_x[index];
					bc += grad_x[index] * grad_y[index];
					d += grad_y[index] * grad_y[index];
					normalizer += 1;
				}
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	}

	// Normalize
	a /= (double)normalizer;
	bc /= (double)normalizer;
	d /= (double)normalizer;

	Matrix2f tensor;
	tensor[0] = a;
	tensor[2] = tensor [1] = bc;
	tensor[3] = d;

	return tensor;
}


inline Matrix2f StructureTensor::calculate_initial_tensor(const float *dyadics,
														  const bool *mask,
														  int size_x,
														  int size_y,
														  float radius,
														  const Point &center) const
{
	int index_at_center = 3 * (center.y * size_x + center.x);
	float grad_x_at_center = std::sqrt(dyadics[index_at_center]);
	float grad_y_at_center = (dyadics[index_at_center + 1] >= 0) ? std::sqrt(dyadics[index_at_center + 2])
																 : -std::sqrt(dyadics[index_at_center + 2]);
	const int margin = 1;
	double a = 0.0, bc = 0.0, d = 0.0;
	long normalizer = 0;

	// Calculate possible limits in Y axis
	long y_lower, y_upper;
	if (std::abs(grad_y_at_center) > EPS) {
		float y_1 = (-radius - grad_x_at_center * (float) center.x) / -grad_y_at_center + center.y;
		float y_2 = (+radius - grad_x_at_center * (float) center.x) / -grad_y_at_center + center.y;
		float y_3 = (-radius - grad_x_at_center * (float) (size_x - center.x - 1)) / grad_y_at_center + center.y;
		float y_4 = (+radius - grad_x_at_center * (float) (size_x - center.x - 1)) / grad_y_at_center + center.y;
		y_lower = std::max(0, (int) std::min(std::min(y_1, y_2), std::min(y_3, y_4)));
		y_upper = std::min(size_y - 1, (int) (std::max(std::max(y_1, y_2), std::max(y_3, y_4)) + 0.5f));
	} else {
		y_lower = 0;
		y_upper = size_y - 1;
	}

	if (std::abs(grad_x_at_center) > EPS) {
		// Scan rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			// For every row compute possible limits in X dimension
			float x_1 = (-radius - grad_y_at_center * (float) (y - center.y)) / grad_x_at_center + center.x;
			float x_2 = (+radius - grad_y_at_center * (float) (y - center.y)) / grad_x_at_center + center.x;
			long x_lower = std::max(0, (int) std::min(x_1, x_2) - margin);
			long x_upper = std::min(size_x - 1, (int) std::max(x_1, x_2) + margin);

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float dist = grad_x_at_center * (float) (x_lower - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float dist = grad_x_at_center * (float) (x_upper - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Aggregate dyadic products at the points of y row between x_lower and x_upper
			for (long x = x_lower; x <= x_upper; x++) {
				long index = y * size_x + x;
				if (!mask || mask[index]) {
					a += dyadics[index * 3];
					bc += dyadics[index * 3 + 1];
					d += dyadics[index * 3 + 2];
					normalizer += 1;
				}
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	} else {    // grad_x_at_center <= EPS
		// Scan complete rows between y_lower and y_upper
		for (long y = y_lower; y <= y_upper; y++) {
			long x_lower = 0;
			long x_upper = size_x - 1;

			// Find exact lower limit in X dimension
			for (; x_lower <= x_upper; x_lower++) {
				float dist = grad_x_at_center * (float) (x_lower - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Find exact upper limit in X dimension
			for (; x_upper >= x_lower; x_upper--) {
				float dist = grad_x_at_center * (float) (x_upper - center.x) + grad_y_at_center * (float) (y - center.y);
				if (abs(dist) < radius) {
					break;
				}
			}

			// Aggregate dyadic products at the points of y row between x_lower and x_upper
			for (long x = x_lower; x <= x_upper; x++) {
				long index = y * size_x + x;
				if (!mask || mask[index]) {
					a += dyadics[index * 3];
					bc += dyadics[index * 3 + 1];
					d += dyadics[index * 3 + 2];
					normalizer += 1;
				}
			}
		}    // for(int y = y_lower; y <= y_upper; y++)
	}

	// Normalize
	a /= (double)normalizer;
	bc /= (double)normalizer;
	d /= (double)normalizer;

	Matrix2f tensor;
	tensor[0] = a;
	tensor[2] = tensor [1] = bc;
	tensor[3] = d;

	return tensor;
}


inline Matrix2f StructureTensor::calculate_next_tensor(const float *grad_x,
													   const float *grad_y,
													   const bool *mask,
													   int size_x,
													   int size_y,
													   float radius,
													   const Point &center,
													   const Matrix2f &tensor) const
{
	// NOTE: we use tensor to locate two extreme points of the ellipse in Y direction, we then
	//		 use tensor again to locate boundaries at every row

	double t_00 = tensor[0];
	double t_01 = tensor[1];
	double t_11 = tensor[3];

	// Ensure that tensor is positive definite and not too elongated
	double trace = t_00 + t_11;
	double det = t_00 * t_11 - t_01 * t_01;
	if (det <= 0.0 || trace * trace / det > EIGEN_RATIO_THRESHOLD) {
		int index = center.y * size_x + center.x;
		Matrix2f new_tensor;
		new_tensor[0] = grad_x[index] * grad_x[index];
		new_tensor[2] = new_tensor[1] = grad_x[index] * grad_y[index];
		new_tensor[3] = grad_y[index] * grad_y[index];
		return new_tensor;
	}

	double nt_00 = 0.0, nt_01 = 0.0, nt_11 = 0.0;
	long normalizer = 0;
	double beta = (_max_size_limit >= 1.0f) ? radius * radius / (std::pow(_max_size_limit, 2.0f)) : 0.0;

	// Locate extrema in Y direction
	double aux = std::sqrt(t_11 - t_01 * t_01 / t_00);
	double dy = radius / aux;	// dy > 0

	// Define aux terms
	double inv_t_00 = 1.0 / t_00;
	double a = t_01 * inv_t_00;
	double b = a * a - t_11 * inv_t_00;
	double c = radius * radius * inv_t_00;

	// Calculate limits in Y dimension
	int y_0 = std::max(center.y - (int)std::floor(dy), 0);
	int y_1 = std::min(center.y + (int)std::floor(dy), size_y - 1);

	// Scan rows between y_0 and y_1
	for (int y = y_0; y <= y_1; ++y) {
		// For every row compute limits in X dimension
		double offset_y = y - center.y;
		double dis = std::sqrt(b * offset_y * offset_y + c);
		int x_0 = (int)std::ceil((double)center.x - a * offset_y - dis);
		int x_1 = (int)std::floor((double)center.x - a * offset_y + dis);
		x_0 = std::max(x_0, 0);
		x_1 = std::min(x_1, size_x - 1);

		// Compute and aggregate dyadic products at the points of y row between x_0 and x_1
		for (int x = x_0; x <= x_1; ++x) {
			int index = y * size_x + x;
			if (!mask || mask[index]) {
				nt_00 += grad_x[index] * grad_x[index];
				nt_01 += grad_x[index] * grad_y[index];
				nt_11 += grad_y[index] * grad_y[index];
				normalizer += 1;
			}
		}
	}

	// Normalize
	nt_00 /= (double)normalizer;
	nt_01 /= (double)normalizer;
	nt_11 /= (double)normalizer;

	// Add some quantity to maintain max_size_limit
	nt_00 += beta;
	nt_11 += beta;

	Matrix2f new_tensor;
	new_tensor[0] = nt_00;
	new_tensor[2] = new_tensor [1] = nt_01;
	new_tensor[3] = nt_11;

	return new_tensor;
}


inline Matrix2f StructureTensor::calculate_next_tensor(const float *dyadics,
													   const bool *mask,
													   int size_x,
													   int size_y,
													   float radius,
													   const Point &center,
													   const Matrix2f &tensor) const
{
	// NOTE: we use tensor to locate two extreme points of the ellipse in Y direction, we then
	//		 use tensor again to locate boundaries at every row

	double t_00 = tensor[0];
	double t_01 = tensor[1];
	double t_11 = tensor[3];

	// Ensure that tensor is positive definite and not too elongated
	double trace = t_00 + t_11;
	double det = t_00 * t_11 - t_01 * t_01;
	if (det <= 0.0 || trace * trace / det > EIGEN_RATIO_THRESHOLD) {
		int index = 3 * (center.y * size_x + center.x);
		Matrix2f new_tensor;
		new_tensor[0] = dyadics[index];
		new_tensor[2] = new_tensor[1] = dyadics[index + 1];
		new_tensor[3] = dyadics[index + 2];
		return new_tensor;
	}

	double nt_00 = 0.0, nt_01 = 0.0, nt_11 = 0.0;
	long normalizer = 0;
	double beta = (_max_size_limit >= 1.0f) ? radius * radius / (std::pow(_max_size_limit, 2.0f)) : 0.0;

	// Locate extrema in Y direction
	double aux = std::sqrt(t_11 - t_01 * t_01 / t_00);
	double dy = radius / aux;	// dy > 0

	// Define aux terms
	double inv_t_00 = 1.0 / t_00;
	double a = t_01 * inv_t_00;
	double b = a * a - t_11 * inv_t_00;
	double c = radius * radius * inv_t_00;

	// Calculate limits in Y dimension
	int y_0 = std::max(center.y - (int)std::floor(dy), 0);
	int y_1 = std::min(center.y + (int)std::floor(dy), size_y - 1);

	// Scan rows between y_0 and y_1
	for (int y = y_0; y <= y_1; ++y) {
		// For every row compute limits in X dimension
		double offset_y = y - center.y;
		double dis = std::sqrt(b * offset_y * offset_y + c);
		int x_0 = (int)std::ceil((double)center.x - a * offset_y - dis);
		int x_1 = (int)std::floor((double)center.x - a * offset_y + dis);
		x_0 = std::max(x_0, 0);
		x_1 = std::min(x_1, size_x - 1);

		// Aggregate dyadic products at the points of y row between x_0 and x_1
		for (int x = x_0; x <= x_1; ++x) {
			int index = y * size_x + x;
			if (!mask || mask[index]) {
				nt_00 += dyadics[index * 3];
				nt_01 += dyadics[index * 3 + 1];
				nt_11 += dyadics[index * 3 + 2];
				normalizer += 1;
			}
		}
	}

	// Normalize
	nt_00 /= (double)normalizer;
	nt_01 /= (double)normalizer;
	nt_11 /= (double)normalizer;

	// Add some quantity to maintain max_size_limit
	nt_00 += beta;
	nt_11 += beta;

	Matrix2f new_tensor;
	new_tensor[0] = nt_00;
	new_tensor[2] = new_tensor[1] = nt_01;
	new_tensor[3] = nt_11;

	return new_tensor;
}

}	// namespace msas
