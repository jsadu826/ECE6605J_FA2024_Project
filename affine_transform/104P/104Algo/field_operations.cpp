/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "field_operations.h"

void FieldOperations::centered_gradient(const float *in, float *dx, float *dy, const int nx, const int ny)
{
	float filter_der[5] = {-1.0/12.0, 8.0/12.0, 0.0, -8.0/12.0, 1.0/12.0};
	float filter_id[1]  = {1.0};

	separate_convolution(in, dx, nx, ny, filter_der, filter_id, 5, 1);
	separate_convolution(in, dy, nx, ny, filter_id, filter_der, 1, 5);
}


/**
 * @note Remember that the filters are ﬂipped in convolution
 */
void FieldOperations::separate_convolution(const float *in, float *out,
										   int size_x, int size_y,
										   const float *filter_x, const float *filter_y,
										   int filter_x_size, int filter_y_size)
{
	// Initialize temporal buffer
	float *buffer;
	buffer = new float[size_x * size_y];

	float sum;
	int id;

	// Do convolution along x axis
	int radius = (filter_x_size - 1) / 2;

	for (int y = 0; y < size_y; y++) {
		for (int x = 0;x < size_x; x++) {
			sum = 0.0;

			for (int i = filter_x_size - 1; i >= 0; i--) {
				id = x + radius - i;
				//id = max(0, min(size_x - 1, id));	// neumann boundary conditions

				// Apply symmetric boundary conditions ( | 3 2 1 0 | 0 1 2 3 | 3 2 1 0 | )
				while ((id < 0) || (id >= size_x)) {
					if (id < 0) {
						id = -id - 1;
					}
					if (id >= size_x) {
						id = 2 * size_x - id -1;
					}
				}

				sum += filter_x[i] * in[y * size_x + id];
			}

			buffer[y * size_x + x] = sum;
		}
	}

	// Do convolution along y axis
	radius = (filter_y_size - 1) / 2;

	for (int y = 0;y < size_y; y++) {
		for (int x = 0; x < size_x; x++) {
			sum = 0.0;

			for (int i = filter_y_size - 1; i >= 0; i--) {
				id = y + radius - i;
				//id = max(0, min(size_y - 1, id));	// neumann boundary conditions

				// Apply symmetric boundary conditions ( | 3 2 1 0 | 0 1 2 3 | 3 2 1 0 | )
				while ((id < 0) || (id >= size_y)) {
					if (id < 0) {
						id = -id - 1;
					}
					if (id >= size_y) {
						id = 2 * size_y - id -1;
					}
				}

				sum += filter_y[i] * buffer[id * size_x + x];
			}

			out[y * size_x + x] = sum;
		}
	}

	// Free memory
	delete [] buffer;
}