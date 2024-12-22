/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef IO_HELPERS_H_H
#define IO_HELPERS_H_H

#include <string>
#include <fstream>
#include "point.h"

namespace iohelpers {

/**
 * Creates a Point from 'x:y' string.
 */
static Point parse_point(std::string str)
{
	size_t delimiter_pos = str.find(':');

	if (delimiter_pos == std::string::npos) {
		return Point(-1, -1);
	}

	try {
		int x = std::stoi(str.substr(0, delimiter_pos));
		int y = std::stoi(str.substr(delimiter_pos + 1, -1));
		return Point(x, y);
	} catch (...) {
		return Point(-1, -1);
	}
}


/**
 * Writes a single-channeled image into a text file.
 * @note Each row of an image is in a separate line, values are separated by the ' ' character.
 */
template <class T = float>
static void write_as_text(std::string filename, const ImageFx<T> &data)
{
	if (data.number_of_channels() > 1) {
		return;
	}

	std::ofstream file(filename.c_str(), std::ios_base::out);

	if (!file.is_open()) {
		return;
	}

	for (uint y = 0; y < data.size_y(); y++) {
		for (uint x = 0; x < data.size_x(); x++) {
			file << data(x, y) << " ";
		}
		file << '\n';
	}

	file.close();
}

}	// namespace iohelpers

#endif // IO_HELPERS_H_H
