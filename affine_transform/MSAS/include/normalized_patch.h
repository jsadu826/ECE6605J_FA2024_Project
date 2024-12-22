/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef NORMALIZED_PATCH_H
#define NORMALIZED_PATCH_H

#include <memory>
#include "matrix.h"

namespace msas {

/**
 * Contains the normalized patch (interpolated to a regular grid) together with
 * the normalizing transformation and the additional orthogonal transformation (e.g. rotation).
 * Notice that the color values of the patch are stored as 1D array which elements
 * map one-to-one to the grid nodes.
 * @see GridInfo
 */
struct NormalizedPatch {
	std::shared_ptr<float *> patch;        // normalized and interpolated patch
	Matrix2f base_transform;     // normalizing transformation
	Matrix2f extra_transform;    // additional orthogonal transformation

	NormalizedPatch(std::shared_ptr<float *> patch,
					Matrix2f base_transform,
					Matrix2f extra_transform = Matrix::identity())
			: patch(patch), base_transform(base_transform), extra_transform(extra_transform) {}
};

}
#endif //NORMALIZED_PATCH_H
