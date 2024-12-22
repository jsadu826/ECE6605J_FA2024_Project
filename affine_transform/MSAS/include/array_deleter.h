/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef ARRAY_DELETER_2D_H
#define ARRAY_DELETER_2D_H

namespace msas
{

/**
 * Generic deleter for a 2D dynamic array.
 * @tparam T Type of array's elements.
 */
template<typename T>
class ArrayDeleter2d
{
public:
	/// @param first_size Size in the first dimension.
	ArrayDeleter2d(size_t first_size) : _first_size(first_size) {}

	void operator()(T** p)
	{
		for (unsigned i = 0; i < _first_size; ++i) {
			delete[] p[i];
		}
		delete[] p;
	}
private:
	size_t _first_size;
};

}	// namespace msas

#endif //ARRAY_DELETER_2D_H
