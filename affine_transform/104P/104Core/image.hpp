/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include "image.h"

/* ==================== FixedImage ==================== */


template <class T>
ImageFx<T>::ImageFx()
 : _size_x(0), _size_y(0), _number_of_channels(0), _color_space(ColorSpaces::unknown), _data()
{

}


template <class T>
ImageFx<T>::ImageFx(uint size_x, uint size_y)
 : _size_x(size_x), _size_y(size_y), _number_of_channels(1), _color_space(ColorSpaces::unknown)
{
	init(size_x, size_y, 1);
}


template <class T>
ImageFx<T>::ImageFx(uint size_x, uint size_y, uint number_of_channels)
 : _size_x(size_x), _size_y(size_y), _number_of_channels(number_of_channels), _color_space(ColorSpaces::unknown)
{
	init(size_x, size_y, number_of_channels);
}


template <class T>
ImageFx<T>::ImageFx(uint size_x, uint size_y, T default_value)
 : _size_x(size_x), _size_y(size_y), _number_of_channels(1), _color_space(ColorSpaces::unknown)
{
	init(size_x, size_y, 1);
	fill_internal(default_value);
}


template <class T>
ImageFx<T>::ImageFx(uint size_x, uint size_y, uint number_of_channels, T default_value)
 : _size_x(size_x), _size_y(size_y), _number_of_channels(number_of_channels), _color_space(ColorSpaces::unknown)
{
	init(size_x, size_y, number_of_channels);
	fill_internal(default_value);
}


template <class T>
ImageFx<T>::ImageFx(Shape size)
 : _size_x(size.size_x), _size_y(size.size_y), _number_of_channels(1), _color_space(ColorSpaces::unknown)
{
	init(size.size_x, size.size_y, 1);
}


template <class T>
ImageFx<T>::ImageFx(Shape size, uint number_of_channels)
 : _size_x(size.size_x), _size_y(size.size_y), _number_of_channels(number_of_channels), _color_space(ColorSpaces::unknown)
{
	init(size.size_x, size.size_y, number_of_channels);
}


template <class T>
ImageFx<T>::ImageFx(Shape size, T default_value)
 : _size_x(size.size_x), _size_y(size.size_y), _number_of_channels(1), _color_space(ColorSpaces::unknown)
{
	init(size.size_x, size.size_y, 1);
	fill_internal(default_value);
}


template <class T>
ImageFx<T>::ImageFx(Shape size, uint number_of_channels, T default_value)
 : _size_x(size.size_x), _size_y(size.size_y), _number_of_channels(number_of_channels), _color_space(ColorSpaces::unknown)
{
	init(size.size_x, size.size_y, number_of_channels);
	fill_internal(default_value);
}


template <class T>
ImageFx<T>::ImageFx(const ImageFx<T> &source)
 : _size_x(source._size_x), _size_y(source._size_y), _number_of_channels(source._number_of_channels),
   _color_space(source._color_space), _data(source._data)
{

}


template <class T>
ImageFx<T>::ImageFx(const Image<T> &source)
 : _size_x(source._size_x), _size_y(source._size_y), _number_of_channels(source._number_of_channels),
   _color_space(source._color_space), _data(source._data)
{

}


template <class T>
ImageFx<T>& ImageFx<T>::operator= (const ImageFx<T> &other)
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

	return *this;
}


template <class T>
ImageFx<T>& ImageFx<T>::operator= (const Image<T> &other)
{
	// assign new data
	this->_size_x = other._size_x;
	this->_size_y = other._size_y;
	this->_number_of_channels = other._number_of_channels;
	this->_color_space = other._color_space;
	this->_data = other._data;

	return *this;
}


template <class T>
ImageFx<T>::operator bool() const
{
	return (bool)_data;
}


template <class T>
bool ImageFx<T>::is_empty() const
{
	return !_data;
}


template <class T>
uint ImageFx<T>::size_x() const
{
	return _size_x;
}


template <class T>
uint ImageFx<T>::size_y() const
{
	return _size_y;
}


template <class T>
Shape ImageFx<T>::size() const
{
	return Shape(_size_x, _size_y);
}


template <class T>
uint ImageFx<T>::number_of_channels() const
{
	return _number_of_channels;
}


template <class T>
ColorSpaces::ColorSpace ImageFx<T>::color_space() const
{
    return _color_space;
}


template <class T>
const T& ImageFx<T>::operator() (uint x, uint y) const
{
	return _data.get()[index(x, y, 0)];
}


template <class T>
const T& ImageFx<T>::operator() (uint x, uint y, uint channel) const
{
	return _data.get()[index(x, y, channel)];
}


template <class T>
const T& ImageFx<T>::operator() (const Point &p) const
{
	return _data.get()[index(p.x, p.y, 0)];
}


template <class T>
const T& ImageFx<T>::operator() (const Point &p, uint channel) const
{
	return _data.get()[index(p.x, p.y, channel)];
}


template <class T>
const T& ImageFx<T>::at(uint x, uint y) const
{
	if (x >= _size_x || y >= _size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	return _data.get()[index(x, y, 0)];
}


template <class T>
const T& ImageFx<T>::at(uint x, uint y, uint channel) const
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	return _data.get()[index(x, y, channel)];
}


template <class T>
const T& ImageFx<T>::at(const Point &p) const
{
	if (p.x >= _size_x || p.y >= _size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	return _data.get()[index(p.x, p.y, 0)];
}


template <class T>
const T& ImageFx<T>::at(const Point &p, uint channel) const
{
	if (p.x >= _size_x || p.y >= _size_y || channel >= _number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	return _data.get()[index(p.x, p.y, channel)];
}


template <class T>
bool ImageFx<T>::try_get_value(uint x, uint y, T& value) const
{
	if (x >= this->_size_x || y >= this->_size_y) {
		return false;
	}

	value = this->_data.get()[index(x, y, 0)];

	return true;
}


template <class T>
bool ImageFx<T>::try_get_value(uint x, uint y, uint channel, T& value) const
{
	if (x >= _size_x || y >= _size_y || channel >= _number_of_channels) {
		return false;
	}

	value = _data.get()[index(x, y, channel)];

	return true;
}


template <class T>
bool ImageFx<T>::try_get_value(const Point &p, T& value) const
{
	if (p.x >= _size_x || p.y >= _size_y) {
		return false;
	}

	value = _data.get()[index(p.x, p.y, 0)];

	return true;
}


template <class T>
bool ImageFx<T>::try_get_value(const Point &p, uint channel, T& value) const
{
	if (p.x >= _size_x || p.y >= _size_y || channel >= _number_of_channels) {
		return false;
	}

	value = _data.get()[index(p.x, p.y, channel)];

	return true;
}


template <class T>
const T* ImageFx<T>::raw() const
{
	return _data.get();
}


template <class T>
const uint ImageFx<T>::raw_length() const
{
	return _size_x * _size_y * _number_of_channels;
}


template <class T>
const uint ImageFx<T>::number_of_pixels() const
{
	return _size_x * _size_y;
}


/**
 * Invokes deep copy.
 */
template <class T>
ImageFx<T> ImageFx<T>::clone() const
{
	ImageFx<T> clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->_color_space;

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		memcpy(clone._data.get(), this->_data.get(),  this->_number_of_channels * this->_size_y * this->_size_x * sizeof(T));
	}

	return clone;
}

/* Protected */

template <class T>
inline void ImageFx<T>::init(uint size_x, uint size_y, uint number_of_channels)
{
	_data = std::shared_ptr<T>(new T[size_x * size_y * number_of_channels](), std::default_delete<T[]>());
}


template <class T>
void ImageFx<T>::fill_internal(const T &value)
{
	std::fill_n(_data.get(), _number_of_channels * _size_y * _size_x, value);
}


template <class T>
inline uint ImageFx<T>::index(uint x, uint y, uint channel) const
{
	//return _size_x * (_size_y * channel + y) + x;	// NOTE: channel by channel
	return _number_of_channels * (_size_x * y + x) + channel;
}


/* ==================== Image ==================== */


template <class T>
Image<T>::Image()
: ImageFx<T>()
{

}


template <class T>
Image<T>::Image(uint size_x, uint size_y)
 : ImageFx<T>(size_x, size_y)
{

}


template <class T>
Image<T>::Image(uint size_x, uint size_y, uint number_of_channels)
 : ImageFx<T>(size_x, size_y, number_of_channels)
{

}


template <class T>
Image<T>::Image(uint size_x, uint size_y, T default_value)
 : ImageFx<T>(size_x, size_y, default_value)
{

}


template <class T>
Image<T>::Image(uint size_x, uint size_y, uint number_of_channels, T default_value)
 : ImageFx<T>(size_x, size_y, number_of_channels, default_value)
{

}


template <class T>
Image<T>::Image(Shape size)
 : ImageFx<T>(size)
{

}


template <class T>
Image<T>::Image(Shape size, uint number_of_channels)
 : ImageFx<T>(size, number_of_channels)
{

}


template <class T>
Image<T>::Image(Shape size, T default_value)
 : ImageFx<T>(size, default_value)
{

}


template <class T>
Image<T>::Image(Shape size, uint number_of_channels, T default_value)
 : ImageFx<T>(size, number_of_channels, default_value)
{

}


template <class T>
Image<T>::Image(const Image<T> &source)
 : ImageFx<T>(source)	// NOTE: no data copying, ref++
{

}


template <class T>
Image<T>::Image(const ImageFx<T> &source)
{
	if (source._data) {
		this->_size_x = source._size_x;
		this->_size_y = source._size_y;
		this->_number_of_channels = source._number_of_channels;
		this->_color_space = source._color_space;
		ImageFx<T>::init(source._size_x, source._size_y, source._number_of_channels);
		memcpy(this->_data.get(), source._data.get(),  source._number_of_channels * source._size_y * source._size_x * sizeof(T));
	} else {
		this->_size_x = 0;
		this->_size_y = 0;
		this->_number_of_channels = 0;
		this->_color_space = ColorSpaces::unknown;
		this->_data.reset();
	}
}


template <class T>
Image<T>& Image<T>::operator= (const Image<T> &other)
{
	// check for self-assignment
	if(this == &other) {
		return *this;
	}

	// finish all deals with the previous data
	this->_data.reset();

	// assign new data
	this->_size_x = other._size_x;
	this->_size_y = other._size_y;
	this->_number_of_channels = other._number_of_channels;
	this->_color_space = other._color_space;
	this->_data = other._data;

	return *this;
}


template <class T>
Image<T>& Image<T>::operator= (const ImageFx<T> &other)
{
	if (other._data) {
		this->_size_x = other._size_x;
		this->_size_y = other._size_y;
		this->_number_of_channels = other._number_of_channels;
		this->_color_space = other._color_space;
		Image<T>::init(other._size_x, other._size_y, other._number_of_channels);
		memcpy(this->_data.get(), other._data.get(),  other._number_of_channels * other._size_y * other._size_x * sizeof(T));
	} else {
		this->_size_x = 0;
		this->_size_y = 0;
		this->_number_of_channels = 0;
		this->_color_space = ColorSpaces::unknown;
		this->_data.reset();
	}

	return *this;
}


template <class T>
void Image<T>::set_color_space(ColorSpaces::ColorSpace value)
{
    this->_color_space = value;
}


template <class T>
T& Image<T>::operator() (uint x, uint y)
{
	return this->_data.get()[Image<T>::index(x, y, 0)];
}


template <class T>
T& Image<T>::operator() (uint x, uint y, uint channel)
{
	return this->_data.get()[Image<T>::index(x, y, channel)];
}


template <class T>
T& Image<T>::operator() (const Point &p)
{
	return this->_data.get()[Image<T>::index(p.x, p.y, 0)];
}


template <class T>
T& Image<T>::operator() (const Point &p, uint channel)
{
	return this->_data.get()[Image<T>::index(p.x, p.y, channel)];
}


template <class T>
T& Image<T>::at(uint x, uint y)
{
	if (x >= this->_size_x || y >= this->_size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	return this->_data.get()[Image<T>::index(x, y, 0)];
}


template <class T>
T& Image<T>::at(uint x, uint y, uint channel)
{
	if (x >= this->_size_x || y >= this->_size_y || channel >= this->_number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	return this->_data.get()[Image<T>::index(x, y, channel)];
}


template <class T>
T& Image<T>::at(const Point &p)
{
	if (p.x < 0 || p.x >= this->_size_x || p.y < 0 || p.y >= this->_size_y) {
		throw std::out_of_range("x or y coordinate is out of range");
	}

	return this->_data.get()[Image<T>::index(p.x, p.y, 0)];
}


template <class T>
T& Image<T>::at(const Point &p, uint channel)
{
	if (p.x < 0 || p.x >= this->_size_x || p.y < 0 || p.y >= this->_size_y || channel >= this->_number_of_channels) {
		throw std::out_of_range("channel, x or y coordinate is out of range");
	}

	return this->_data.get()[Image<T>::index(p.x, p.y, channel)];
}


template <class T>
void Image<T>::fill(const T &value)
{
	Image<T>::fill_internal(value);
}


template <class T>
T* Image<T>::raw()
{
	return this->_data.get();
}


template <class T>
Image<T> Image<T>::clone() const
{
	Image<T> clone;
	clone._size_x = this->_size_x;
	clone._size_y = this->_size_y;
	clone._number_of_channels = this->_number_of_channels;
	clone._color_space = this->_color_space;

	if (this->_data) {
		clone.init(this->_size_x, this->_size_y, this->_number_of_channels);
		memcpy(clone._data.get(), this->_data.get(),  this->_number_of_channels * this->_size_y * this->_size_x * sizeof(T));
	}

	return clone;
}
