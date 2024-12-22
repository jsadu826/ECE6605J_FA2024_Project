/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#ifndef GRID_INFO_H
#define GRID_INFO_H

#include <memory>

namespace msas {

/**
 * Represents coordinates of a node of a regular grid.
 */
struct GridCoord {
	float x;        // real x coordinate within a grid
	float y;        // real y coordinate within a grid
	int index_x;    // column id of the node in a grid
	int index_y;    // row id of the node in a grid
};


/**
 * Represents a regular grid.
 * @see EllipseNormalization::create_regular_grid() for construction details.
 */
struct GridInfo {
	float step;								// distance between the nodes
	int size;								// number of nodes along each dimension
	std::unique_ptr<GridCoord[]> nodes;		// array of nodes' coordinates
	size_t nodes_length;					// total number of nodes
	std::unique_ptr<int[]> index;			// mapping between position of a node and its id in the nodes array
	size_t index_length;					// length of the index array

	GridInfo() : step(0.0f), size(0), nodes_length(0), index_length(0) { }
};

}

#endif //GRID_INFO_H
