/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef I_ITERABLE_MASK_H_
#define I_ITERABLE_MASK_H_

#include "point.h"

/**
 * Interface for a mask that allows to iterate through the masked points.
 */
class IIterableMask
{
public:
	virtual ~IIterableMask() {}

	/// Get the first masked point
	virtual Point first() const = 0;

	/// Get the last masked point
	virtual Point last() const = 0;

	/// Get the next masked point
	virtual Point next(const Point &current) const = 0;

	/// Get the previous masked point
	virtual Point prev(const Point &current) const = 0;
};


#endif /* I_ITERABLE_MASK_H_ */
