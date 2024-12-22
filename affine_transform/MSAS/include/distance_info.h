/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef DISTANCE_INFO_H
#define DISTANCE_INFO_H

#include "matrix.h"
#include "point.h"

namespace msas {

/**
 * Contains the value of (affine invariant) patch distance together with
 * the transformations that gave this value and points at which it was computed.
 */
struct DistanceInfo {
	float distance;
	Matrix2f first_transform;
	Matrix2f second_transform;
	Point first_point;
	Point second_point;

	DistanceInfo() : distance(-1) { }

	friend inline bool operator<(const DistanceInfo &lhs, const DistanceInfo &rhs) {
		return lhs.distance < rhs.distance;
	}

	friend inline bool operator>(const DistanceInfo &lhs, const DistanceInfo &rhs) { return operator<(rhs, lhs); }

	friend inline bool operator<=(const DistanceInfo &lhs, const DistanceInfo &rhs) { return !operator>(lhs, rhs); }

	friend inline bool operator>=(const DistanceInfo &lhs, const DistanceInfo &rhs) { return !operator<(lhs, rhs); }
};

}	// namespace msas

#endif //DISTANCE_INFO_H
