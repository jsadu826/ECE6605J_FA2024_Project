/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef FIELD_OPERATIONS_H_
#define FIELD_OPERATIONS_H_

class FieldOperations
{
public:
	static void centered_gradient(const float *in,float *dx, float *dy, const int nx, const int ny);
	static void separate_convolution(const float *in, float *out,
									 int size_x, int size_y,
									 const float *filter_x, const float *filter_y,
									 int filter_x_size, int filter_y_size);
};

#endif /* FIELD_OPERATIONS_H_ */
