/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef MATRIX_H_
#define MATRIX_H_

#include <array>

using Matrix2f = std::array<float, 4>;

namespace Matrix {

constexpr Matrix2f zero() {
	return Matrix2f{0.0f, 0.0f, 0.0f, 0.0f};
}

constexpr Matrix2f identity() {
	return Matrix2f{1.0f, 0.0f, 0.0f, 1.0f};
}

static Matrix2f multiply(const Matrix2f &first, const Matrix2f &second) {
	return Matrix2f{first[0] * second[0] + first[1] * second[2],
					first[0] * second[1] + first[1] * second[3],
					first[2] * second[0] + first[3] * second[2],
					first[2] * second[1] + first[3] * second[3]};
}

}

#endif //MATRIX_H_
