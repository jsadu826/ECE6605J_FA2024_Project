/**
 * Copyright (C) 2016, Vadim Fedorov <coderiks@gmail.com>
 *
 * This program is free software: you can use, modify and/or
 * redistribute it under the terms of the simplified BSD
 * License. You should have received a copy of this license along
 * this program. If not, see
 * <http://www.opensource.org/licenses/bsd-license.html>.
 */

#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <tclap/CmdLine.h>
#include "io_utility.h"
#include "structure_tensor.h"
#include "field_operations.h"
#include "ellipse_normalization.h"
#include "io_helpers.h"
#include "matrix.h"

using std::vector;
using std::pair;
using std::string;

int main(int argc, char* argv[])
{
	// Declare command line arguments
	TCLAP::CmdLine cmd("In default mode compute Affine Covariant Structure Tensors for given image and parameters. "
					   "See 'modes' argument for other available options.", ' ', "1.0");
	TCLAP::UnlabeledValueArg<string> image_arg("image", "Load the given image.", true, string(), "file name", cmd);
	TCLAP::ValueArg<float> hue_arg("", "hue", "Set the Hue [0, 360] for drawing regions in the 'ellipses' mode. Default: 60.0.", false, 60.0f, "float", cmd);
	TCLAP::ValueArg<float> saturation_arg("", "saturation", "Set the Saturation [0.0, 1.0] for drawing regions in the 'ellipses' mode. Default: 1.0.", false, 1.0f, "float", cmd);
	TCLAP::ValueArg<float> size_limit_arg("", "size-limit", "Set the maximum allowed radius of an elliptical region (circle) shall it appear in a uniform region. Default: 0.0.", false, 0.0f, "float", cmd);
	TCLAP::ValueArg<float> gamma_arg("g", "gamma", "Set the mixing coefficient for the experimental scheme of Structure Tensors computation. Should be in range (0.0, 1.0], where 1.0 corresponds to the original scheme. Default: 1.0.", false, 1.0f, "float", cmd);
	TCLAP::ValueArg<int> iterations_arg("i", "iter", "Set the number of iterations for Structure Tensors. Default: 60.", false, 60, "int", cmd);
	TCLAP::ValueArg<float> radius_arg("r", "radius", "Set the R ('radius') parameter. Default: 100.0.", false, 100.0f, "float", cmd);
	TCLAP::ValueArg<int> step_arg("s", "step", "Set the step between the points of interest. Applicable in the 'avg_size' and 'ellipses' modes. Default: 50.", false, 50, "int", cmd);
	TCLAP::ValueArg<string> points_arg("", "points", "Load the given text file with a set of points of interest (one point per line: 'X Y'). Applicable in the 'ellipses' mode.", false, string(), "string", cmd);
	TCLAP::ValueArg<string> output_arg("o", "output", "Set the name for output file(s) without extension.", false, "out", "string", cmd);
	vector<string>  modes_list;
	modes_list.push_back("sizes");
	modes_list.push_back("avg_size");
	modes_list.push_back("transforms");
	modes_list.push_back("ellipses");
	modes_list.push_back("ellipses+tensors");
	TCLAP::ValuesConstraint<string> modes_constrain(modes_list);
	TCLAP::ValueArg<string> mode_arg("m", "mode",
									 "Specify what should be done instead of computing Affine Covariant Structure Tensors:\n"
									 "'sizes' - compute sizes (in pixels) of Affine Covariant Regions;\n"
									 "'avg_size' - compute the average size of Affine Covariant Regions;\n"
									 "'transforms' - compute transformations that normalize elliptical Affine Covariant Regions to a disk;\n"
									 "'ellipses' - draw Affine Covariant Regions over the image."
									 "'ellipses+tensors' - draw Affine Covariant Regions over the image and output their corresponding Structure Tensors.",
									 false, string(), &modes_constrain, cmd);

	// Parse command line arguments
	try {
		cmd.parse(argc, argv);
	} catch (TCLAP::ArgException &e) {  // catch any exceptions
		std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}

	// Get names
	string image_name = image_arg.getValue();
	string output_name = output_arg.getValue();
	string points_filename = points_arg.getValue();

	// Get parameters
	float radius = radius_arg.getValue();
	int number_of_iterations = iterations_arg.getValue();
	float gamma = gamma_arg.getValue();
	int step = std::max(step_arg.getValue(), 1);
	string mode = mode_arg.getValue();
	float max_size_limit = size_limit_arg.getValue();
	float hue = std::max(0.0f, std::min(360.0f, hue_arg.getValue()));
	float saturation = std::max(0.0f, std::min(1.0f, saturation_arg.getValue()));

	// Read the image
	Image<float> image = IOUtility::read_mono_image(image_name);

	if (!image) {
		std::cerr << "Could not open image '" << image_name << "'" << std::endl;
		return 1;
	}

	// Compute image gradient
	Image<float> gradient_x(image.size_x(), image.size_y(), 0.0f);
	Image<float> gradient_y(image.size_x(), image.size_y(), 0.0f);
	FieldOperations::centered_gradient(image.raw(),
									   gradient_x.raw(),
									   gradient_y.raw(),
									   image.size_x(),
									   image.size_y());

	// Compute tensor products
	Image<float> dyadics(gradient_x.size(), (uint)3);
	int size_x = gradient_x.size_x();
	float *dyadics_data = dyadics.raw();
	float *grad_x_data = gradient_x.raw();
	float *grad_y_data = gradient_y.raw();
	for (uint y = 0; y < image.size_y(); y++) {
		for (uint x = 0; x < image.size_x(); x++) {
			int index = size_x * y + x;
			dyadics_data[index * 3] = grad_x_data[index] * grad_x_data[index];
			dyadics_data[index * 3 + 1] = grad_x_data[index] * grad_y_data[index];
			dyadics_data[index * 3 + 2] = grad_y_data[index] * grad_y_data[index];
		}
	}

	// Create StructureTensor calculator
	msas::StructureTensor *structure_tensor = new msas::StructureTensor(radius, number_of_iterations, gamma);
	structure_tensor->set_max_size_limit(max_size_limit);

	// Do processing according to the selected mode
	if (mode == "sizes") {
		std::cout << "Computing sizes of Affine Covariant Regions..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Compute sizes of regions at every point
		Image<float> sizes(image.size_x(), image.size_y());
		#pragma omp parallel for schedule(dynamic,1) collapse(2) shared(sizes)
		for (uint y = 0; y < image.size_y(); ++y) {
			for (uint x = 0; x < image.size_x(); ++x) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, Point(x, y), MaskFx());
				vector<Point> region = structure_tensor->calculate_region(tensor, Point(x, y), image.size());
				sizes(x, y) = region.size();
			}
		}

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

		iohelpers::save_floats(output_name + "_region_sizes.txt", sizes);
	} else if (mode == "avg_size") {
		std::cout << "Computing average size of Affine Covariant Regions..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Redefine step if it was not explicitly set by a user.
		if (!step_arg.isSet()) {
			step = 1;
		}

		// Compute average size of regions
		double total = 0.0;
		double count = 0.0;
		#pragma omp parallel for reduction(+:total,count)
		for (uint y = 0; y < image.size_y(); y += step) {
			for (uint x = 0; x < image.size_x(); x += step) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, Point(x, y), MaskFx());
				vector<Point> region = structure_tensor->calculate_region(tensor, Point(x, y), image.size());
				total += region.size();
				count++;
			}
		}

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

        double avg_size = total / count;
        std::cout << "Average size of Affine Covariant Regions is " << avg_size << std::endl;
    } else if (mode == "transforms") {
		std::cout << "Computing transformations..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Create elliptical patch normalization calculator
		msas::EllipseNormalization normalization;

		// At every point compute transformations that maps elliptical patches to a disk
		vector<iohelpers::TransformInfo> transforms;
		#pragma omp parallel for schedule(dynamic,1) shared(transforms)
		for (uint y = 0; y < image.size_y(); ++y) {
			for (uint x = 0; x < image.size_x(); ++x) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, Point(x, y), MaskFx());
				float angle;
				Matrix2f transform = structure_tensor->calculate_transformation(tensor, angle, radius);
				vector<Point> region = structure_tensor->calculate_region(tensor, Point(x, y), image.size(), radius);
				vector<float> dominant_orientations = normalization.calculate_dominant_orientations(gradient_x,
																									gradient_y, region,
																									transform,
																									Point(x, y));

				for (auto it = dominant_orientations.begin(); it != dominant_orientations.end(); ++it) {
					Matrix2f rotation = normalization.rotation(*it);

					iohelpers::TransformInfo info;
					info.transform = Matrix::multiply(rotation, transform);
					info.x = x;
					info.y = y;

					#pragma omp critical
					transforms.push_back(info);
				}
			}
		}

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

		iohelpers::save_transforms(output_name + "_transforms.txt", transforms);
	} else if (mode == "ellipses") {
		std::cout << "Computing and drawing Affine Covariant Regions..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Prepare an image to draw over it
		Image<float> canvas(image.size(), 3, 0.0f);
		canvas.set_color_space(ColorSpaces::RGB);
		for (uint y = 0; y < image.size_y(); y++) {
			for (uint x = 0; x < image.size_x(); x++) {
				canvas(x, y, 0) = image(x, y);
				canvas(x, y, 1) = image(x, y);
				canvas(x, y, 2) = image(x, y);
			}
		}

		// Change color space to HSV
		canvas = IOUtility::rgb_to_hsv(canvas);

		// Draw elliptical regions (shape-adaptive patches) at the points of interest
		vector<Point> points_of_interest = iohelpers::read_points(points_filename);
		if (points_of_interest.size() > 0) {
			// Use points of interest from the provided text file
			for (auto p_it = points_of_interest.begin(); p_it != points_of_interest.end(); ++p_it) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, *p_it, MaskFx());
				vector<Point> region = structure_tensor->calculate_region(tensor, *p_it, image.size());

				// Draw single elliptical region
				for (auto it = region.begin(); it != region.end(); ++it) {
					canvas(*it, 0) = hue;
					canvas(*it, 1) = saturation;
					canvas(*it, 2) = canvas(*it, 2) * 1.2f;  // multiply value to see boundaries better
				}
			}
		} else {
			if (!points_filename.empty()) {
				std::cout << "WARNING: points file '" << points_filename
						  << "' is specified, but no points were loaded." << std::endl;
			}

			// Use points of interest that are distributed regularly
			for (uint y = 0; y < image.size_y(); y += step) {
				for (uint x = 0; x < image.size_x(); x += step) {
					Matrix2f tensor = structure_tensor->calculate(dyadics, Point(x, y), MaskFx());
					vector<Point> region = structure_tensor->calculate_region(tensor, Point(x, y), image.size());

					// Draw single elliptical region
					for (auto it = region.begin(); it != region.end(); ++it) {
						canvas(*it, 0) = hue;
						canvas(*it, 1) = saturation;
						canvas(*it, 2) = canvas(*it, 2) * 1.2f;  // multiply value to see boundaries better
					}
				}
			}
		}

		// Change color space back to RGB
		canvas = IOUtility::hsv_to_rgb(canvas);

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

		IOUtility::write_rgb_image(output_name + "_regions.png", canvas);
	} else if (mode == "ellipses+tensors") {
		std::cout << "Computing and drawing Affine Covariant Regions (with extra Structure Tensors output)..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Prepare an image to draw over it
		Image<float> canvas(image.size(), 3, 0.0f);
		canvas.set_color_space(ColorSpaces::RGB);
		for (uint y = 0; y < image.size_y(); y++) {
			for (uint x = 0; x < image.size_x(); x++) {
				canvas(x, y, 0) = image(x, y);
				canvas(x, y, 1) = image(x, y);
				canvas(x, y, 2) = image(x, y);
			}
		}

		// Change color space to HSV
		canvas = IOUtility::rgb_to_hsv(canvas);

		// Prepare structure for structure tensors
		vector<pair<Point, Matrix2f> > tensors;

		// Draw elliptical regions (shape-adaptive patches) at the points of interest
		vector<Point> points_of_interest = iohelpers::read_points(points_filename);
		if (points_of_interest.size() > 0) {
			tensors.reserve(points_of_interest.size());

			// Use points of interest from the provided text file
			for (auto p_it = points_of_interest.begin(); p_it != points_of_interest.end(); ++p_it) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, *p_it, MaskFx());
				vector<Point> region = structure_tensor->calculate_region(tensor, *p_it, image.size());

				// Store structure tensor
				tensors.push_back({*p_it, tensor});

				// Draw single elliptical region
				for (auto it = region.begin(); it != region.end(); ++it) {
					canvas(*it, 0) = hue;
					canvas(*it, 1) = saturation;
					canvas(*it, 2) = canvas(*it, 2) * 1.2f;  // multiply value to see boundaries better
				}
			}
		} else {
			if (!points_filename.empty()) {
				std::cout << "WARNING: points file '" << points_filename
						  << "' is specified, but no points were loaded." << std::endl;
			}

			tensors.reserve(image.size_x() * image.size_y());

			// Use points of interest that are distributed regularly
			for (uint y = 0; y < image.size_y(); y += step) {
				for (uint x = 0; x < image.size_x(); x += step) {
					Point p(x, y);
					Matrix2f tensor = structure_tensor->calculate(dyadics, p, MaskFx());
					vector<Point> region = structure_tensor->calculate_region(tensor, p, image.size());

					// Store structure tensor
					tensors.push_back({p, tensor});

					// Draw single elliptical region
					for (auto it = region.begin(); it != region.end(); ++it) {
						canvas(*it, 0) = hue;
						canvas(*it, 1) = saturation;
						canvas(*it, 2) = canvas(*it, 2) * 1.2f;  // multiply value to see boundaries better
					}
				}
			}
		}

		// Change color space back to RGB
		canvas = IOUtility::hsv_to_rgb(canvas);

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

		IOUtility::write_rgb_image(output_name + "_regions.png", canvas);
		iohelpers::save_tensors(output_name + "_structure_tensors.txt", tensors);
	} else {	// if default mode
		std::cout << "Computing Affine Covariant Structure Tensors..." << std::endl;
		auto time_start = std::chrono::system_clock::now();

		// Compute affine covariant structure tensors for all the points
		Image<Matrix2f> tensors(image.size_x(), image.size_y());
		#pragma omp parallel for schedule(dynamic,1) collapse(2) shared(tensors)
		for (uint y = 0; y < image.size_y(); ++y) {
			for (uint x = 0; x < image.size_x(); ++x) {
				Matrix2f tensor = structure_tensor->calculate(dyadics, Point(x, y), MaskFx());
				tensors(x, y) = tensor;
			}
		}

		auto time_end = std::chrono::system_clock::now();
		std::chrono::duration<double> elapsed_seconds = time_end - time_start;
		std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

		iohelpers::save_tensors(output_name + "_structure_tensors.txt", tensors);
	}
}