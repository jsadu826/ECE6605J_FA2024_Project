/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "mask.h"

MaskFx::iterator MaskFx::empty_iterator()
{
	return iterator(0, Point(-1, -1));
}

MaskFx::MaskFx()
 : ImageFx<bool>(), _internal()
{

}


MaskFx::MaskFx(uint size_x, uint size_y)
 : ImageFx<bool>(size_x, size_y, false)
{
	init_internal(Point(size_x, size_y), Point(-1, -1), true);
}


MaskFx::MaskFx(uint size_x, uint size_y, bool default_value)
 : ImageFx<bool>(size_x, size_y, default_value)
{
	if (default_value) {
		init_internal(Point(0, 0), Point(size_x - 1, size_y - 1), true);
	} else {
		init_internal(Point(size_x, size_y), Point(-1, -1), true);
	}
}


MaskFx::MaskFx(Shape size)
 : ImageFx<bool>(size, false)
{
	init_internal(Point(size.size_x, size.size_y), Point(-1, -1), true);
}


MaskFx::MaskFx(Shape size, bool default_value)
 : ImageFx<bool>(size.size_x, size.size_y, default_value)
{
	if (default_value) {
		init_internal(Point(0, 0), Point(size.size_x - 1, size.size_y - 1), true);
	} else {
		init_internal(Point(size.size_x, size.size_y), Point(-1, -1), true);
	}
}


/**
 * Deep copy
 */
MaskFx::MaskFx(const Image<bool> &source)
 : ImageFx<bool>(source.clone())	// we have to invoke deep copying explicitly, because Image to ImageFx cast leads to the data sharing
{
	init_internal(Point(0, 0), Point(0, 0), false);
}


/**
 * Deep copy
 */
MaskFx::MaskFx(const ImageFx<bool> &source)
 : ImageFx<bool>(source.clone())	// we have to invoke deep copying explicitly, because ImageFx to ImageFx cast leads to the data sharing
{
	init_internal(Point(0, 0), Point(0, 0), false);
}


/**
 * Ref++
 */
MaskFx::MaskFx(const Mask &source)
 : ImageFx<bool>(source)	// Mask is casted to ImageFx implicitly, then ImageFx to ImageFx cast leads to the data sharing
{
	_internal = source._internal;
}


/**
 * Ref++
 */
MaskFx::MaskFx(const MaskFx &source)
 : ImageFx<bool>(source)	// MaskFx is casted to ImageFx implicitly, then ImageFx to ImageFx cast leads to the data sharing
{
	_internal = source._internal;
}


MaskFx::~MaskFx()
{

}


MaskFx& MaskFx::operator= (const Mask &other)
{
	// check for self-assignment
	if(this == &other) {
		return *this;
	}

	// assign new data
	this->_size_x = other._size_x;
	this->_size_y = other._size_y;
	this->_number_of_channels = other._number_of_channels;
	this->_color_space = other._color_space;
	this->_data = other._data;
	this->_internal = other._internal;

	return *this;
}


MaskFx& MaskFx::operator= (const MaskFx &other)
{
	// check for self-assignment
	if(this == &other) {
		return *this;
	}

	// assign new data
	this->_size_x = other._size_x;
	this->_size_y = other._size_y;
	this->_number_of_channels = other._number_of_channels;
	this->_color_space = other._color_space;
	this->_data = other._data;
	this->_internal = other._internal;

	return *this;
}


MaskFx::iterator MaskFx::begin() const
{
	return iterator(this, first());
}


MaskFx::iterator MaskFx::end() const
{
	return iterator(this, Point(_size_x, _size_y));
}


MaskFx::iterator MaskFx::rbegin() const
{
	return iterator(this, last(), true);
}


MaskFx::iterator MaskFx::rend() const
{
	return iterator(this, Point(-1, -1), true);
}


bool MaskFx::get(uint x, uint y) const
{
	return _data.get()[index(x, y, 0)];
}


bool MaskFx::get(uint x, uint y, uint channel) const
{

	return _data.get()[index(x, y, channel)];
}


bool MaskFx::get(Point p) const
{
	return _data.get()[index(p.x, p.y, 0)];
}


bool MaskFx::get(Point p, uint channel) const
{
	return _data.get()[index(p.x, p.y, channel)];
}


bool MaskFx::test(uint x, uint y) const
{
	if (x >= _size_x || y >= _size_y) {
		return false;
	}

	return _data.get()[index(x, y, 0)];
}


bool MaskFx::test(uint x, uint y, uint channel) const
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		return false;
	}

	return _data.get()[index(x, y, channel)];
}


bool MaskFx::test(Point p) const
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.y < 0 || (uint)p.y >= _size_y) {
		return false;
	}

	return _data.get()[index(p.x, p.y, 0)];
}


bool MaskFx::test(Point p, uint channel) const
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.x < 0 || (uint)p.y >= _size_y || channel >= _number_of_channels) {
		return false;
	}

	return _data.get()[index(p.x, p.y, channel)];
}


MaskFx MaskFx::clone() const
{
	MaskFx clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->color_space();

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		memcpy(clone._data.get(), this->_data.get(),  this->_number_of_channels * this->_size_y * this->_size_x * sizeof(bool));
		clone.init_internal(_internal->first, _internal->last, _internal->is_first_last_valid);
	}

	return clone;
}


MaskFx MaskFx::clone_invert() const
{
	MaskFx clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->color_space();

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		bool *clone_data = clone._data.get();
		bool *data = this->_data.get();
		for (uint i = 0; i < _size_x * _size_y * _number_of_channels; i++) {
			clone_data[i] = !data[i];
		}

		clone.init_internal(_internal->first, _internal->last, false);
	}

	return clone;
}


std::vector<Point> MaskFx::masked_points() const
{
	if (!_internal->is_points_cache_valid) {
		_internal->points_cache.clear();

		// fill points cache
		for(iterator it = begin(); it != end(); ++it) {
			_internal->points_cache.push_back(*it);
		}

		_internal->is_points_cache_valid = true;
	}

	return _internal->points_cache;
}


Point MaskFx::first() const
{
	if (!_internal->is_first_last_valid) {
		actualize_first_last();
	}

	return _internal->first;
}


Point MaskFx::last() const
{
	if (!_internal->is_first_last_valid) {
		actualize_first_last();
	}

	return _internal->last;
}


Point MaskFx::next(const Point &current) const
{
	int from_x = current.x + 1;
	bool *data = _data.get();
	for (uint y = current.y; y < _size_y; y++) {
		for (uint x = from_x; x < _size_x; x++) {
			if (data[index(x, y, 0)]) {
				return Point(x, y);
			}
		}
		from_x = 0;
	}

	return Point(_size_x, _size_y);
}


Point MaskFx::prev(const Point &current) const
{
	int from_x = current.x - 1;
	bool *data = _data.get();
	for (int y = current.y; y >= 0; y--) {
		for (int x = from_x; x >= 0; x--) {
			if (data[index(x, y, 0)]) {
				return Point(x, y);
			}
		}
		from_x = _size_x - 1;
	}

	return Point(-1, -1);
}


/* Protected */

inline void MaskFx::init_internal(Point first, Point last, bool is_first_last_valid)
{
	_internal = std::make_shared<__Internal>(first, last, is_first_last_valid);
}


inline void MaskFx::actualize_first_last() const
{
	_internal->first = Point(_size_x, _size_y);
	_internal->last = Point(-1, -1);

	bool *data = _data.get();

	bool is_last_found = false;
	for (int y = _size_y - 1; (!is_last_found) && (y >= 0); y--) {
		for (int x = _size_x - 1; (!is_last_found) && (x >= 0); x--) {
			if (data[index(x, y, 0)]) {
				_internal->last = Point(x, y);
				is_last_found = true;
			}
		}
	}

	if (is_last_found) {
		bool _is_first_found = false;
		for (uint y = 0; (!_is_first_found) && (y < _size_y); y++) {
			for (uint x = 0; (!_is_first_found) && (x < _size_x); x++) {
				if (data[index(x, y, 0)]) {
					_internal->first = Point(x, y);
					_is_first_found = true;
				}
			}
		}
	}

	_internal->is_first_last_valid = true;
}


/* ==================== Mask ==================== */

Mask::Mask()
	: MaskFx()
{

}


Mask::Mask(uint size_x, uint size_y)
	: MaskFx(size_x, size_y, false)
{

}


Mask::Mask(uint size_x, uint size_y, bool default_value)
	: MaskFx(size_x, size_y, default_value)
{

}


Mask::Mask(Shape size)
	: MaskFx(size, false)
{

}


Mask::Mask(Shape size, bool default_value)
	: MaskFx(size.size_x, size.size_y, default_value)
{

}


/**
 * Deep copy
 */
Mask::Mask(const Image<bool> &source)
	: MaskFx(source)	// Image to MaskFx cast leads to deep copying
{

}


/**
 * Deep copy
 */
Mask::Mask(const ImageFx<bool> &source)
	: MaskFx(source)	// ImageFx to MaskFx cast leads to deep copying
{

}


/**
 * Ref++
 */
Mask::Mask(const Mask &source)
	: MaskFx(source)	// Mask to MaskFx cast leads to the data sharing
{

}


/**
 * Deep copy
 */
Mask::Mask(const MaskFx &source)
	: MaskFx(source.clone())	// we have to invoke deep copying explicitly, because ImageFx to ImageFx cast leads to the data sharing
{

}


Mask::~Mask()
{

}


Mask& Mask::operator= (const Mask &other)
{
	// check for self-assignment
	if(this == &other) {
		return *this;
	}

	// assign new data
	this->_size_x = other._size_x;
	this->_size_y = other._size_y;
	this->_number_of_channels = other._number_of_channels;
	this->_color_space = other._color_space;
	this->_data = other._data;
	this->_internal = other._internal;

	return *this;
}


Mask& Mask::operator= (const MaskFx &other)
{
	// check for self-assignment
	if(this == &other) {
		return *this;
	}

	if (other._data) {
		this->_size_x = other._size_x;
		this->_size_y = other._size_y;
		this->_number_of_channels = other._number_of_channels;
		this->_color_space = other._color_space;
		init(other._size_x, other._size_y, other._number_of_channels);
		memcpy(this->_data.get(), other._data.get(),  other._number_of_channels * other._size_y * other._size_x * sizeof(bool));
		init_internal(other._internal->first, other._internal->last, other._internal->is_first_last_valid);
	} else {
		this->_size_x = 0;
		this->_size_y = 0;
		this->_number_of_channels = 0;
		this->_color_space = ColorSpaces::unknown;
		this->_data.reset();
		this->_internal.reset();
	}

	return *this;
}


bool& Mask::operator() (uint x, uint y)
{
	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(x, y, 0)];
}


bool& Mask::operator() (uint x, uint y, uint channel)
{
	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(x, y, channel)];
}


bool& Mask::operator() (Point p)
{
	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(p.x, p.y, 0)];
}


bool& Mask::operator() (Point p, uint channel)
{
	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(p.x, p.y, channel)];
}


bool& Mask::at(uint x, uint y)
{
	if (x >= _size_x || y >= _size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(x, y, 0)];
}


bool& Mask::at(uint x, uint y, uint channel)
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(x, y, channel)];
}


bool& Mask::at(Point p)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.y < 0 || (uint)p.y >= _size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(p.x, p.y, 0)];
}


bool& Mask::at(Point p, uint channel)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.x < 0 || (uint)p.y >= _size_y || channel >= _number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	return _data.get()[index(p.x, p.y, channel)];
}


void Mask::mask(uint x, uint y)
{
	if (x >= _size_x || y >= _size_y) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(x, y, 0)] = true;
}


void Mask::mask(uint x, uint y, uint channel)
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(x, y, channel)] = true;
}


void Mask::mask(Point p)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.y < 0 || (uint)p.y >= _size_y) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(p.x, p.y, 0)] = true;
}


void Mask::mask(Point p, uint channel)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.x < 0 || (uint)p.y >= _size_y || channel >= _number_of_channels) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(p.x, p.y, channel)] = true;
}


void Mask::unmask(uint x, uint y)
{
	if (x >= _size_x || y >= _size_y) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(x, y, 0)] = false;
}


void Mask::unmask(uint x, uint y, uint channel)
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(x, y, channel)] = true;
}


void Mask::unmask(Point p)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.y < 0 || (uint)p.y >= _size_y) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(p.x, p.y, 0)] = true;
}


void Mask::unmask(Point p, uint channel)
{
	if (p.x < 0 || (uint)p.x >= _size_x || p.x < 0 || (uint)p.y >= _size_y || channel >= _number_of_channels) {
		return;
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;

	_data.get()[index(p.x, p.y, channel)] = true;
}


Mask Mask::clone() const
{
	Mask clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->_color_space;

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		memcpy(clone._data.get(), this->_data.get(),  this->_number_of_channels * this->_size_y * this->_size_x * sizeof(bool));
		clone.init_internal(_internal->first, _internal->last, _internal->is_first_last_valid);
	}

	return clone;
}


Mask Mask::clone_invert() const
{
	Mask clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->_color_space;

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		bool *clone_data = clone._data.get();
		bool *data = this->_data.get();
		for (uint i = 0; i < _size_x * _size_y * _number_of_channels; i++) {
			clone_data[i] = !data[i];
		}

		clone.init_internal(_internal->first, _internal->last, false);
	}

	return clone;
}


void Mask::invert()
{
	bool *data = _data.get();
	for (uint i = 0; i < _size_x * _size_y * _number_of_channels; i++) {
		data[i] = !data[i];
	}

	_internal->is_first_last_valid = false;
	_internal->is_points_cache_valid = false;
}
