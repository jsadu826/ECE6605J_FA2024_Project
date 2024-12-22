/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef LUT_MATH_H_
#define LUT_MATH_H_

#include <limits>

/**
 * @brief Some math functions approximated using Look-Up Table technique.
 */
class LUT
{
public:
	/// @brief Look-up table implementation of exponent
	/// @note: make sure to use this method only for the range (-709; 709).
	///        For automatic range check use LUT::exp_rc version.
	static double exp(double x);



	/// @brief Look-up table implementation of exponent with range check.
	static double exp_rc(double x);

	/// @brief Look-up table implementation of exponent with negative boundary of the range check.
	static double exp_rcn(double x);

	/// @brief Look-up table implementation of exponent with positive boundary of the range check.
	static double exp_rcp(double x);

private:
	union float_long {
		double f;
		long long l;
	};
	static double ExpAdjustment[256];
};



#endif /* LUT_MATH_H_ */
