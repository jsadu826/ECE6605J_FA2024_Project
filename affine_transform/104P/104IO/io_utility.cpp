/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include <tinydir.h>
#include "io_utility.h"

const float IOUtility::EPS = 0.00001f;
const float IOUtility::TAG_FLOAT = 202021.25f;
const std::string IOUtility::TAG_STRING = "PIEH";

std::string IOUtility::_prefix = "";

/**
 * Reads image in PGM format from file with provided path\name
 * to the object of 'Image' class
 */
Image<float> IOUtility::read_pgm_image(const std::string &name)
{
	/* open file */
	// COMPATIBILITY: for win 'rb' file mode instead of just 'r'
	FILE *f = fopen(name.data(),"rb");
	if( f == NULL ) {
		return Image<float>();
	}

	/* read header */
	bool isBinary = false;
	int c, x_size,y_size,depth;

	if ( fgetc(f) != 'P' ) {
		fclose(f);
		return Image<float>();
	}

	if( (c=fgetc(f)) == '2' ) {
		isBinary = false;
	} else if ( c == '5' ) {
		isBinary = true;
	} else {
		fclose(f);
		return Image<float>();
	}

	skip_spaces_and_comments(f);
	fscanf(f,"%d",&x_size);
	skip_spaces_and_comments(f);
	fscanf(f,"%d",&y_size);
	skip_spaces_and_comments(f);
	fscanf(f,"%d",&depth);

	/* get memory */
	Image<float> image(x_size, y_size);
    image.set_color_space(ColorSpaces::mono);

	/* read data */
	skip_spaces_and_comments(f);
	int value;
	for(int y=0;y<y_size;y++) {
		for(int x=0;x<x_size;x++) {
			value = isBinary ? fgetc(f) : get_number(f);
			image(x,y) = value;
		}
	}

	/* close and return */
	fclose(f);

	return image;
}


/**
 * Writes image in PGM format to file with provided path\name
 * from the object of 'Image' class
 */
void IOUtility::write_pgm_image(const std::string &name, const ImageFx<float> &image)
{
	/* open file */
	FILE *f = fopen(name.data(),"wb");

	/* write header */
	putc('P', f);
	putc('5', f);

	/* write attributes */
	int x_size = image.size_x();
	int y_size = image.size_y();
	fprintf(f, "\n%d %d\n%d\n", x_size, y_size, 255);

	/* write data */
	char value;
	for (int y = 0;y < y_size;y++) {
		for (int x = 0;x < x_size;x++) {
			value = (char)image(x,y);
			putc(value, f);
		}
	}

	/* close file */
	fclose(f);
}


Image<float> IOUtility::read_mono_image(const std::string &name)
{
	int width, height;

	float *image_data = iio_read_image_float(name.data(), &width, &height);

	if (!image_data) {
		fprintf(stderr, "read_mono_image: cannot read %s\n", name.data());
		return Image<float>();
	}

	Image<float> image(width, height);
    image.set_color_space(ColorSpaces::mono);

	// copy from image_data to Image<float>
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x);
			float value = image_data[index];
			image(x, y) = value;
		}
	}

	free(image_data);

	return image;
}


void IOUtility::write_mono_image(const std::string &name, const ImageFx<float> &image)
{
	int width = image.size_x();
	int height = image.size_y();

	float *image_data = new float[width * height];

	// copy from Image<float> to image_data
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = (y * width + x);
			image_data[index] = image(x, y);
		}
	}

	iio_save_image_float(const_cast<char* >(name.data()), image_data, width, height);
	delete[] image_data;
}


Image<float> IOUtility::read_rgb_image(const std::string &name)
{
	int width, height;

	float *image_data = iio_read_image_float_rgb(name.data(), &width, &height);

	if (! image_data) {
		fprintf(stderr, "read_rgb_image: cannot read %s\n", name.data());
		exit(1);
	}

	Image<float> image(width, height, (uint)3);
    image.set_color_space(ColorSpaces::RGB);

	// copy from image_data to Image<float>
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = 3 * (y * width + x);
			image(x, y, 0) = image_data[index];
			image(x, y, 1) = image_data[index + 1];
			image(x, y, 2) = image_data[index + 2];
		}
	}

	free(image_data);

	return image;
}


void IOUtility::write_rgb_image(const std::string &name, const ImageFx<float> &image)
{
	int width = image.size_x();
	int height = image.size_y();

	ImageFx<float> rgb_image = to_rgb(image);

	float *image_data = new float[width * height * 3];

	// copy from Image<float> to image_data
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int index = 3 * (y * width + x);
			image_data[index + 0] = rgb_image(x, y, 0);
			image_data[index + 1] = rgb_image(x, y, 1);
			image_data[index + 2] = rgb_image(x, y, 2);
		}
	}

	iio_save_image_float_vec(const_cast<char* >(name.data()), image_data, width, height, 3);

   delete[] image_data;
}


void IOUtility::write_float_image(const std::string &name, const ImageFx<float> &image)
{
	iio_save_image_float_split(const_cast<char*>(name.data()), const_cast<float*>(image.raw()), image.size_x(), image.size_y(), 1);
}


Image<float> IOUtility::read_optical_flow(const std::string &name)
{
	if (name.empty()) {
		return Image<float>();
	}

	if (name.find_last_of(".flo") == std::string::npos) {
		// .flo extension is expected
		return Image<float>();
	}

	// Try to open the flo file
	std::fstream flo_file(name, std::ios_base::in | std::ios_base::binary);
	if (!flo_file.is_open()) {
		return Image<float>();
	}

	// Read tag, width and height
	float tag;
	int width, height;
	flo_file.read((char*)&tag, sizeof(float));
	flo_file.read((char*)&width, sizeof(int32_t));
	flo_file.read((char*)&height, sizeof(int32_t));

	if (!flo_file.good()) {
		return Image<float>();
	}

	// Simple test for correct endian-ness
	if (tag != TAG_FLOAT) {
		return Image<float>();
	}

	// Allocate memory for the flow
	Image<float> flow(width, height, 2, 0.0f);
	float *flow_ptr = flow.raw();

	// Read flow row by row
	for (int y = 0; y < height; ++y) {
		flo_file.read((char*)flow_ptr, sizeof(float) * 2 * width);
		if (flo_file.fail()) {
			return Image<float>();	// file is too short
		}

		flow_ptr += 2 * width;
	}

	return flow;
}


void IOUtility::write_optical_flow(const std::string &name, Image<float> flow)
{
	if (name.empty() || flow.number_of_channels() != 2) {
		return;
	}

	if (name.find_last_of(".flo") == std::string::npos) {
		// .flo extension is expected
		return;
	}

	std::fstream flo_file(name, std::ios_base::out | std::ios_base::binary);
	if (!flo_file.is_open()) {
		return;
	}

	int width = flow.size_x();
	int height = flow.size_y();

	// Write the header
	flo_file.write(TAG_STRING.c_str(), 4);
	flo_file.write((char*)&width, sizeof(width));
	flo_file.write((char*)&height, sizeof(height));

	if (!flo_file.good()) {
		return;
	}

	float *flow_ptr = flow.raw();

	// Write the rows
	for (int y = 0; y < height; ++y) {
		flo_file.write((char*)flow_ptr, sizeof(float) * 2 * width);
		flow_ptr += 2 * width;
		if (!flo_file.good()) {
			return;
		}
	}
}


std::vector<Image<float> > IOUtility::read_all_flows(const std::string &folder, const std::string &prefix)
{
	tinydir_dir dir;
	tinydir_open(&dir, folder.c_str());

	// Get all file names beginning with prefix
	std::vector<std::string> filenames;
	filenames.reserve(dir.n_files);
	while (dir.has_next) {
		tinydir_file file;
		tinydir_readfile(&dir, &file);

		if (file.is_dir == 0 && (prefix.empty() || strncmp(file.name, prefix.c_str(), prefix.length()) == 0)) {
			filenames.push_back(std::string(file.name));
		}

		tinydir_next(&dir);
	}

	tinydir_close(&dir);

	// Make sure file names are in alphabetic order
	std::sort(filenames.begin(), filenames.end());

	// Read flows
	std::vector<Image<float> > flows;
	flows.reserve(filenames.size());
	for (auto it = filenames.begin(); it != filenames.end(); ++it) {
		Image<float> flow = read_optical_flow(folder + '/' + *it);
		if (!flow.is_empty()) {
			flows.push_back(flow);
		}
	}

	return flows;
}


Image<float> IOUtility::rgb_to_gray(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::RGB) {
		std::cerr << "WARNING! Original image is not in RGB color space [IOUtility::rgb_to_gray]." << std::endl;
	}

	Image<float> image_gray(image.size(), 1, 0.0f);
	image_gray.set_color_space(ColorSpaces::mono);
	float* data_gray = image_gray.raw();
	const float *data_rgb = image.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	for (int i = 0; i < number_of_pixels; i++) {
		data_gray[i] = 0.2989f * data_rgb[3 * i] + 0.5870f * data_rgb[3 * i + 1] + 0.1140f * data_rgb[3 * i + 2];
	}

	return image_gray;
}


Image<float> IOUtility::rgb_to_lab(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::RGB) {
		std::cerr << "WARNING! Original image is not in RGB color space [IOUtility::rgb_to_lab]." << std::endl;
	}

	Image<float> image_lab(image.size(), 3, 0.0f);
    image_lab.set_color_space(ColorSpaces::Lab);
	const float* data_rgb = image.raw();
	float* data_lab = image_lab.raw();
	long number_of_pixels = image.size_x() * image.size_y();
	float *xyz_buffer = new float[3];

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		rgb_to_xyz(data_rgb + i * 3, xyz_buffer);
		xyz_to_lab(xyz_buffer, data_lab + i * 3);
	}

	delete[] xyz_buffer;

	return image_lab;
}


Image<float> IOUtility::lab_to_rgb(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::Lab) {
		std::cerr << "WARNING! Original image is not in Lab color space [IOUtility::lab_to_rgb]." << std::endl;
	}

	Image<float> image_rgb(image.size(), 3, 0.0f);
	image_rgb.set_color_space(ColorSpaces::RGB);
	const float* data_lab = image.raw();
	float* data_rgb = image_rgb.raw();
	long number_of_pixels = image.size_x() * image.size_y();
	float *xyz_buffer = new float[3];

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		lab_to_xyz(data_lab + i * 3, xyz_buffer);
		xyz_to_rgb(xyz_buffer, data_rgb + i * 3);
	}

	delete[] xyz_buffer;

	return image_rgb;
}


Image<float> IOUtility::rgb_to_hsv(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::RGB) {
		std::cerr << "WARNING! Original image is not in RGB color space [IOUtility::rgb_to_hsv]." << std::endl;
	}

	Image<float> image_hsv(image.size(), 3, 0.0f);
	image_hsv.set_color_space(ColorSpaces::HSV);
	const float* data_rgb = image.raw();
	float* data_hsv = image_hsv.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		rgb_to_hsv(data_rgb + i * 3, data_hsv + i * 3);
	}

	return image_hsv;
}


Image<float> IOUtility::hsv_to_rgb(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::HSV) {
		std::cerr << "WARNING! Original image is not in HSV color space [IOUtility::hsv_to_rgb]." << std::endl;
	}

	Image<float> image_rgb(image.size(), 3, 0.0f);
	image_rgb.set_color_space(ColorSpaces::RGB);
	const float* data_hsv = image.raw();
	float* data_rgb = image_rgb.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		hsv_to_rgb(data_hsv + i * 3, data_rgb + i * 3);
	}

	return image_rgb;
}


Image<float> IOUtility::rgb_to_yuv(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::RGB) {
		std::cerr << "WARNING! Original image is not in RGB color space [IOUtility::rgb_to_yuv]." << std::endl;
	}

	Image<float> image_yuv(image.size(), 3, 0.0f);
	image_yuv.set_color_space(ColorSpaces::YUV);
	const float* data_rgb = image.raw();
	float* data_yuv = image_yuv.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		rgb_to_yuv(data_rgb + i * 3, data_yuv + i * 3);
	}

	return image_yuv;
}


Image<float> IOUtility::yuv_to_rgb(ImageFx<float> image)
{
	if (image.number_of_channels() != 3) {
		return image;
	}

	if (image.color_space() != ColorSpaces::YUV) {
		std::cerr << "WARNING! Original image is not in YUV color space [IOUtility::yuv_to_rgb]." << std::endl;
	}

	Image<float> image_rgb(image.size(), 3, 0.0f);
	image_rgb.set_color_space(ColorSpaces::RGB);
	const float* data_yuv = image.raw();
	float* data_rgb = image_rgb.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	//#pragma omp parallel for
	for (int i = 0; i < number_of_pixels; i++) {
		yuv_to_rgb(data_yuv + i * 3, data_rgb + i * 3);
	}

	return image_rgb;
}


Image<float> IOUtility::to_mono(ImageFx<float> image)
{
	unsigned int number_of_channels = image.number_of_channels();
	if (number_of_channels == 1) {
		return image;
	}

	if (image.color_space() == ColorSpaces::RGB) {
		return rgb_to_gray(image);
	} else if (image.color_space() == ColorSpaces::YUV || image.color_space() == ColorSpaces::Lab) {
		return take_channel(image, 0);
	} else if (image.color_space() == ColorSpaces::HSV) {
		return take_channel(image, 2);
	} else {
		return average_channels(image);
	}
}


Image<float> IOUtility::to_rgb(ImageFx<float> image)
{
	if (image.color_space() == ColorSpaces::RGB) {
		return image;
	} else if (image.color_space() == ColorSpaces::Lab) {
		return IOUtility::lab_to_rgb(image);
	} else if (image.color_space() == ColorSpaces::HSV) {
		return IOUtility::hsv_to_rgb(image);
	} else if (image.color_space() == ColorSpaces::YUV) {
		return IOUtility::yuv_to_rgb(image);
	} else if (image.color_space() == ColorSpaces::mono || image.number_of_channels() == 1) {
		return repeat_channel(image, 3, ColorSpaces::RGB);
	}

	return Image<float>();
}


void IOUtility::set_prefix(const std::string &prefix)
{
	_prefix = prefix;
}


std::string IOUtility::compose_file_name(const std::string &name)
{
	return _prefix + name;
}


std::string IOUtility::compose_file_name(const std::string &name, int index, const std::string &extension)
{
	std::stringstream stream;
	stream << _prefix << name << "_" << std::setfill('0') << std::setw(3) << index << "." << extension;
	std::string file_name = stream.str();
	return file_name;
}


std::string IOUtility::compose_file_name(const std::string &name, int index, int index2, const std::string &extension)
{
	std::stringstream stream;
	stream << _prefix << name << "_" << index << "_" << std::setw(3) << std::setfill('0') << index2 << "." << extension;
	std::string file_name = stream.str();
	return file_name;
}

/* Private */

void IOUtility::skip_spaces_and_comments(FILE * f)
{
	int c;
	do
	{
		while(isspace(c=fgetc(f))); /* skip spaces */
		if(c=='#') while((c=fgetc(f))!='\n'); /* skip comments */
	}
	while(c == '#');
	ungetc(c,f);
}


/**
 *  Read a number digit by digit.
 */
int IOUtility::get_number(FILE * f)
	{
	int num, c;

	while(isspace(c=fgetc(f)));
	if(!isdigit(c)) exit(1);
	num = c - '0';
	while( isdigit(c=fgetc(f)) ) num = 10 * num + c - '0';

	return num;
}


void IOUtility::rgb_to_xyz(const float *rgb, float *xyz)
{
	float aux_r = rgb[0] / 255.0f;
	float aux_g = rgb[1] / 255.0f;
	float aux_b = rgb[2] / 255.0f;

	aux_r = (aux_r > 0.04045f) ? pow((aux_r + 0.055f) / 1.055f , 2.4f) : aux_r / 12.92f;
	aux_g = (aux_g > 0.04045f) ? pow((aux_g + 0.055f) / 1.055f , 2.4f) : aux_g / 12.92f;
	aux_b = (aux_b > 0.04045f) ? pow((aux_b + 0.055f) / 1.055f , 2.4f) : aux_b / 12.92f;

	aux_r *= 100.0f;
	aux_g *= 100.0f;
	aux_b *= 100.0f;

	xyz[0] = aux_r * 0.412453f + aux_g * 0.357580f + aux_b * 0.180423f;
	xyz[1] = aux_r * 0.212671f + aux_g * 0.715160f + aux_b * 0.072169f;
	xyz[2] = aux_r * 0.019334f + aux_g * 0.119193f + aux_b * 0.950227f;
}


void IOUtility::xyz_to_lab(const float *xyz, float *lab)
{
	// normalize by the reference white
	float aux_x = xyz[0] / 95.047f;
	float aux_y = xyz[1] / 100.000f;
	float aux_z = xyz[2] / 108.883f;

	aux_x = (aux_x > 0.008856f) ? pow(aux_x, 1.0f / 3.0f) : (7.787f * aux_x) + (16.0f / 116.0f);
	aux_y = (aux_y > 0.008856f) ? pow(aux_y, 1.0f / 3.0f) : (7.787f * aux_y) + (16.0f / 116.0f);
	aux_z = (aux_z > 0.008856f) ? pow(aux_z, 1.0f / 3.0f) : (7.787f * aux_z) + (16.0f / 116.0f);

	lab[0] = (116.0f * aux_y) - 16.0f;
	lab[1] = 500.0f * (aux_x - aux_y);
	lab[2] = 200.0f * (aux_y - aux_z);
}


void IOUtility::lab_to_xyz(const float *lab, float *xyz)
{
	float aux_y = (lab[0] + 16.0f) / 116.0f;
	float aux_x = lab[1] / 500.0f + aux_y;
	float aux_z = aux_y - lab[2] / 200.0f;

	aux_x = (pow(aux_x, 3.0f) > 0.008856f) ? pow(aux_x, 3.0f) : (aux_x - 16.0f / 116.0f) / 7.787f;
	aux_y = (pow(aux_y, 3.0f) > 0.008856f) ? pow(aux_y, 3.0f) : (aux_y - 16.0f / 116.0f) / 7.787f;
	aux_z = (pow(aux_z, 3.0f) > 0.008856f) ? pow(aux_z, 3.0f) : (aux_z - 16.0f / 116.0f) / 7.787f;

	xyz[0] = aux_x * 95.047f;
	xyz[1] = aux_y * 100.000f;
	xyz[2] = aux_z * 108.883f;
}


void IOUtility::xyz_to_rgb(const float *xyz, float *rgb)
{
	float aux_x = xyz[0] / 100.0f;
	float aux_y = xyz[1] / 100.0f;
	float aux_z = xyz[2] / 100.0f;

	float aux_r = aux_x *  3.240479f + aux_y * -1.537150f + aux_z * -0.498535f;
	float aux_g = aux_x * -0.969256f + aux_y *  1.875992f + aux_z *  0.041556f;
	float aux_b = aux_x *  0.055648f + aux_y * -0.204043f + aux_z *  1.057311f;

	aux_r = (aux_r > 0.0031308f) ? 1.055f * pow(aux_r, 1.0f / 2.4f) - 0.055f : 12.92f * aux_r;
	aux_g = (aux_g > 0.0031308f) ? 1.055f * pow(aux_g, 1.0f / 2.4f) - 0.055f : 12.92f * aux_g;
	aux_b = (aux_b > 0.0031308f) ? 1.055f * pow(aux_b, 1.0f / 2.4f) - 0.055f : 12.92f * aux_b;

	rgb[0] = aux_r * 255.0f;
	rgb[1] = aux_g * 255.0f;
	rgb[2] = aux_b * 255.0f;
}


/**
 * @note The hue value H runs from 0 to 360º.
 * The saturation S is the degree of strength or purity and is from 0 to 1.
 * Purity is how much white is added to the color, so S=1 makes the purest color (no white).
 * Brightness V also ranges from 0 to 1, where 0 is the black.
 */
void IOUtility::rgb_to_hsv(const float *rgb, float *hsv)
{
	float min_value = std::min(rgb[0], std::min(rgb[1], rgb[2]));
	float max_value = std::max(rgb[0], std::max(rgb[1], rgb[2]));

	hsv[2] = max_value / 255.0f;		// v

	float delta = max_value - min_value;

	if(delta > EPS) {
		hsv[1] = delta / max_value;		// s
	} else { // r = g = b = 0
		hsv[0] = 0.0f;	// h is undefined
		hsv[1] = 0.0f;
		return;
	}

	if(rgb[0] >= max_value) {
		hsv[0] = (rgb[1] - rgb[2]) / delta;			// between yellow & magenta
	} else if(rgb[1] >= max_value) {
		hsv[0] = 2.0f + (rgb[2] - rgb[0]) / delta;	// between cyan & yellow
	} else {
		hsv[0] = 4.0f + (rgb[0] - rgb[1]) / delta;	// between magenta & cyan
	}

	hsv[0] *= 60.0f;	// degrees
	if(hsv[0] < 0.0f) {
		hsv[0] += 360.0f;
	}
}


void IOUtility::hsv_to_rgb(const float *hsv, float *rgb)
{
	// If achromatic (grey)
	if(hsv[1] < EPS) {
		rgb[0] = hsv[2] * 255.0f;
		rgb[1] = hsv[2] * 255.0f;
		rgb[2] = hsv[2] * 255.0f;
		return;
	}

	float h = hsv[0] / 60.0f;
	int id = (int)h;			// sector 0 to 5
	float f = h - (float)id;	// factorial part of h

	float p = hsv[2] * (1.0f - hsv[1]);
	float q = hsv[2] * (1.0f - hsv[1] * f);
	float t = hsv[2] * (1.0f - hsv[1] * (1.0f - f));

	switch( id ) {
		case 0:
			rgb[0] = hsv[2] * 255.0f;
			rgb[1] = t * 255.0f;
			rgb[2] = p * 255.0f;
			break;
		case 1:
			rgb[0] = q * 255.0f;
			rgb[1] = hsv[2] * 255.0f;
			rgb[2] = p * 255.0f;
			break;
		case 2:
			rgb[0] = p * 255.0f;
			rgb[1] = hsv[2] * 255.0f;
			rgb[2] = t * 255.0f;
			break;
		case 3:
			rgb[0] = p * 255.0f;
			rgb[1] = q * 255.0f;
			rgb[2] = hsv[2] * 255.0f;
			break;
		case 4:
			rgb[0] = t * 255.0f;
			rgb[1] = p * 255.0f;
			rgb[2] = hsv[2] * 255.0f;
			break;
		default:		// case 5:
			rgb[0] = hsv[2] * 255.0f;
			rgb[1] = p * 255.0f;
			rgb[2] = q * 255.0f;
			break;
	}
}


void IOUtility::rgb_to_yuv(const float *rgb, float *yuv)
{
	constexpr float a = 1.0f / sqrt(3.0f);
	constexpr float b = 1.0f / sqrt(2.0f);
	constexpr float c = 2.0f * a * sqrt(2.0f);

	yuv[0] = a * (rgb[0] + rgb[1] + rgb[2]);
	yuv[1] = b * (rgb[0] - rgb[2]);
	yuv[2] = c * (0.25f * rgb[0] - 0.5f * rgb[1] + 0.25f * rgb[2]);
}


void IOUtility::yuv_to_rgb(const float *yuv, float *rgb)
{
	constexpr float a = 1.0f / sqrt(3.0f);
	constexpr float b = 1.0f / sqrt(2.0f);
	constexpr float c = a / b;

	rgb[0] = a * yuv[0] + b * yuv[1] + c * 0.5f * yuv[2];
	rgb[1] = a * yuv[0] - c * yuv[2];
	rgb[2] = a * yuv[0] - b * yuv[1] + c * 0.5f * yuv[2];
}


/**
 * @brief Repeat a single channel given number of times.
 */
Image<float> IOUtility::repeat_channel(const ImageFx<float> &image,
											  unsigned int num,
											  ColorSpaces::ColorSpace color_space)
{
	unsigned int number_of_channels = image.number_of_channels();
	if (number_of_channels != 1) {
		return Image<float>();
	}

	Image<float> image_out(image.size(), num, 0.0f);
	image_out.set_color_space(color_space);
	float *data_out = image_out.raw();
	const float *data = image.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	for (int i = 0; i < number_of_pixels; ++i) {
		for (int j = 0; j < num; ++j) {
			data_out[num * i + j] = data[i];
		}
	}

	return image_out;
}


/**
 * @brief Extract a single channel from a given image.
 */
Image<float> IOUtility::take_channel(const ImageFx<float> &image, unsigned int channel_id)
{
	unsigned int number_of_channels = image.number_of_channels();
	if (channel_id >= number_of_channels) {
		return Image<float>();
	}

	Image<float> image_out(image.size(), 1, 0.0f);
	image_out.set_color_space(ColorSpaces::mono);
	float *data_out = image_out.raw();
	const float *data = image.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	for (int i = 0; i < number_of_pixels; i++) {
		data_out[i] = data[number_of_channels * i + channel_id];
	}

	return image_out;
}


/**
 * @brief Create a new image with a single channel which values are averages of multiple channels of an input image.
 */
Image<float> IOUtility::average_channels(const ImageFx<float> &image)
{
	unsigned int number_of_channels = image.number_of_channels();
	if (number_of_channels == 1) {
		return image;
	}

	Image<float> image_avg(image.size(), 1, 0.0f);
	image_avg.set_color_space(ColorSpaces::mono);
	float *data_avg = image_avg.raw();
	const float *data = image.raw();
	long number_of_pixels = image.size_x() * image.size_y();

	if (number_of_channels == 3) {
		for (int i = 0; i < number_of_pixels; i++) {
			data_avg[i] = (data[3 * i] + data[3 * i + 1] + data[3 * i + 2]) / 3.0f;
		}
	} else if (number_of_channels == 2) {
		for (int i = 0; i < number_of_pixels; i++) {
			data_avg[i] = (data[2 * i] + data[2 * i + 1]) / 2.0f;
		}
	} else {	// general case
		for (int i = 0; i < number_of_pixels; i++) {
			float avg = 0.0f;
			for (int ch = 0; ch < number_of_channels; ch++) {
				avg += data[number_of_channels * i + ch];
			}
			data_avg[i] = avg / (float)number_of_channels;
		}
	}

	return image_avg;
}