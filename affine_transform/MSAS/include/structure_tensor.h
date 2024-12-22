/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef STRUCTURE_TENSOR_H_
#define STRUCTURE_TENSOR_H_

#include <vector>
#include <cmath>
#include <memory>
#include <functional>
#include "image.h"
#include "mask.h"
#include "point.h"
#include "shape.h"
#include "matrix.h"

namespace msas {

/**
 * Encapsulates iterative scheme for computing affine covariant structure tensors
 * and affine covariant regions (shape-adaptive patches) in 2D case. Implements also
 * an experimental scheme with weighted update of structure tensors which is controlled
 * by parameter 'gamma'. Values of gamma should be in range (0.0, 1.0]. When gamma is set
 * to 1.0 (default value) the original scheme is applied.
 */
class StructureTensor {
public:
	/// @param radius Value of R parameter to be used in computations (can be overwritten in some methods).
	/// @param iterations_amount Number of iterations of the computational scheme.
	/// @param gamma Mixing coefficient for Structure Tensor update between iterations.
	/// @note When @param gamma is set to 1.0 (default), the original iterative scheme is applied.
	StructureTensor(float radius, int iterations_amount, float gamma);

	StructureTensor(float radius, int iterations_amount);

	StructureTensor(float radius);

	StructureTensor();

	StructureTensor(const StructureTensor &other);

	~StructureTensor() {}

	StructureTensor &operator=(const StructureTensor &other);

	/// Compute structure tensor at the given point.
	/// @param grad_x X component of an image gradient.
	/// @param grad_y Y component of an image gradient.
	/// @param point Point of interest.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Matrix2f calculate(const ImageFx<float> &grad_x,
					   const ImageFx<float> &grad_y,
					   const Point &point,
					   const MaskFx &mask) const;

	/// Compute structure tensor at the given point.
	/// @param dyadics Precomputed dyadic products of gradient vectors, stored in 3 channels: dx*dx, dx*dy, dy*dy.
	/// @param point Point of interest.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Matrix2f calculate(const ImageFx<float> &dyadics,
					   const Point &point,
					   const MaskFx &mask) const;

	/// Compute structure tensors at every point.
	/// @param grad_x X component of an image gradient.
	/// @param grad_y Y component of an image gradient.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Image<Matrix2f> calculate(const ImageFx<float> &grad_x,
							  const ImageFx<float> &grad_y,
							  const MaskFx &mask) const;

	/// Compute structure tensors at every point.
	/// @param dyadics Precomputed dyadic products of gradient vectors, stored in 3 channels: dx*dx, dx*dy, dy*dy.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Image<Matrix2f> calculate(const ImageFx<float> &dyadics,
							  const MaskFx &mask) const;

	/// Compute structure tensor for a given region (set of points).
	/// @param grad_x X component of an image gradient.
	/// @param grad_y Y component of an image gradient.
	/// @param region Set of points (normally shape-adaptive patch) to be considered in the computation.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Matrix2f calculate(const ImageFx<float> &grad_x,
					   const ImageFx<float> &grad_y,
					   const std::vector<Point> &region,
					   const MaskFx &mask) const;

	/// Compute structure tensor for a given region (set of points).
	/// @param dyadics Precomputed dyadic products of gradient vectors, stored in 3 channels: dx*dx, dx*dy, dy*dy.
	/// @param region Set of points (normally shape-adaptive patch) to be considered in the computation.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	Matrix2f calculate(const ImageFx<float> &dyadics,
					   const std::vector<Point> &region,
					   const MaskFx &mask) const;

	/// Compute initial region which is a band depending on the gradient at a given point.
	/// @param grad_x X component of an image gradient.
	/// @param grad_y Y component of an image gradient.
	/// @param point Point of interest.
	/// @param mask Binary mask defining points that are allowed to contribute. When empty, all points are allowed.
	/// @param radius [optional] Value of R parameter that overwrites the internal value in this particular call.
	std::vector<Point> calculate_initial_region(const ImageFx<float> &grad_x,
												const ImageFx<float> &grad_y,
												const Point &point,
												const MaskFx &mask,
												float radius = -1.0f) const;

	/// Compute elliptical region (shape-adaptive patch) for a given structure tensor at a given point.
	/// @param tensor Structure tensor.
	/// @param point Point where the structure tensor was computed.
	/// @param size Size of the image domain.
	/// @param radius [optional] Value of R parameter that overwrites the internal value in this particular call.
	std::vector<Point> calculate_region(const Matrix2f &tensor,
										const Point &point,
										const Shape &size,
										float radius = -1.0f) const;

	/// Compute intra-patch anisotropic Gaussian weights.
	/// @param region Set of points (normally shape-adaptive patch), for which weights should be computed.
	/// @param tensor Corresponding structure tensor.
	/// @param center Point where the structure tensor was computed.
	/// @param radius [optional] Value of R parameter that overwrites the internal value in this particular call.
	/// @param sigma_factor [optional] Coefficient for the Gaussian sigma: sigma = radius / sigma_factor.
	/// @return Array of weights for points of @param region.
	/// @note Size of the returned array is equal to the size of @param region.
	std::unique_ptr<float[]> calculate_weights(const std::vector<Point> &region,
											   const Matrix2f &tensor,
											   const Point &center,
											   float radius = -1.0f,
											   float sigma_factor = 3.0f) const;

	/// Compute transformation matrix T = sqrt(D) * U / radius.
	/// @param tensor Structure tensor.
	/// @param angle [out] Angle between the major axis and 0Y axis in the range [-p; p].
	/// @param radius [optional] Value of R parameter that overwrites the internal value in this particular call.
	/// @return A transform that maps an elliptical patch to a unit circle.
	Matrix2f calculate_transformation(const Matrix2f &tensor,
									  float &angle,
									  float radius = -1.0f) const;

	/// Compute square root of a structure tensor matrix T = U' * sqrt(D) * U.
	/// @param tensor Structure tensor.
	Matrix2f sqrt(const Matrix2f &tensor) const;

	// Getters/setters for parameters
	float radius() const;

	void set_radius(float value);

	float gamma() const;

	/// @note Should be in range (0.0, 1.0] where 1.0 corresponds to the original scheme.
	void set_gamma(float value);

	int iterations_amount() const;

	void set_iterations_amount(int value);

	float max_size_limit() const;

	/// Set the maximum allowed radius of an elliptical region (circle) shall it appear in a uniform region.
	void set_max_size_limit(float value);

private:
	// Structure tensors can be computed using gradients or precomputed dyadic products,
	// also using the original or modified scheme, one by one or all together.
	// Following functors allow to abstract from these details
	using CalcFirstFunc = std::function<Matrix2f(Point)>;
	using CalcNextFunc = std::function<Matrix2f(Point, Matrix2f)>;
	using RunSchemeFunc = std::function<Matrix2f(CalcFirstFunc &, CalcNextFunc &, const Point &)>;

	// Default values for parameters
	constexpr static float DEFAULT_RADIUS = 300.0f;
	constexpr static int DEFAULT_ITERATIONS_AMOUNT = 60;
	constexpr static float DEFAULT_GAMMA = 1.0f;		// original iterative scheme
	constexpr static float DEFAULT_SIZE_LIMIT = 0.0f;	// no limit by default
	constexpr static float DEFAULT_VARIATION_THRESHOLD = 0.0001f;

	constexpr static float EPS = 0.0001f;
	constexpr static float MAX_EIGEN_RATIO = 100.0f;
	constexpr static float EIGEN_RATIO_THRESHOLD = (MAX_EIGEN_RATIO + 1) * (MAX_EIGEN_RATIO + 1) / MAX_EIGEN_RATIO;

	float _radius;
	int _iterations_amount;
	float _gamma;
	float _max_size_limit;        // max allowed radius of an ellipse (circle) in a uniform region
	float _variation_threshold;

	RunSchemeFunc _run_scheme_func;    // NOTE: it depends on the value of _gamma and is defined in configure()

	void configure();

	inline Matrix2f run_original_scheme(CalcFirstFunc &calc_first,
										CalcNextFunc &calc_next,
										const Point &point) const;

	inline Matrix2f run_stabilized_scheme(CalcFirstFunc &calc_first,
										  CalcNextFunc &calc_next,
										  const Point &point) const;

	inline Matrix2f calculate_initial_tensor(const float *grad_x,
											 const float *grad_y,
											 const bool *mask,
											 int size_x,
											 int size_y,
											 float radius,
											 const Point &center) const;

	inline Matrix2f calculate_initial_tensor(const float *dyadics,
											 const bool *mask,
											 int size_x,
											 int size_y,
											 float radius,
											 const Point &center) const;

	inline Matrix2f calculate_next_tensor(const float *grad_x,
										  const float *grad_y,
										  const bool *mask,
										  int size_x,
										  int size_y,
										  float radius,
										  const Point &center,
										  const Matrix2f &tensor) const;

	inline Matrix2f calculate_next_tensor(const float *dyadics,
										  const bool *mask,
										  int size_x,
										  int size_y,
										  float radius,
										  const Point &center,
										  const Matrix2f &tensor) const;
};

}    // namespace msas

#endif /* STRUCTURE_TENSOR_H_ */
