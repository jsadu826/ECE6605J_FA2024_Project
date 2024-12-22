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
#include "ellipse_normalization.h"
#include "structure_tensor_bundle.h"
#include "affine_patch_distance.h"
#include "io_helpers.h"

using std::vector;
using std::string;

int main(int argc, char* argv[])
{
	// Declare command line arguments
	TCLAP::CmdLine cmd("Compute similarity (distance) map for a given point of interest in the source image and all the points in the target image.", ' ', "1.0");
	TCLAP::ValueArg<float> size_limit_arg("", "size-limit", "Set the maximum allowed radius of an elliptical region (circle) shall it appear in a uniform region.", false, 0.0f, "float", cmd);
	TCLAP::ValueArg<int> grid_size_arg("", "grid", "Set the interpolation grid size. Default: 21.", false, 21, "int", cmd);
	TCLAP::ValueArg<float> gamma_arg("g", "gamma", "Set the mixing coefficient for the experimental scheme of Structure Tensors computation. Should be in range (0.0, 1.0], where 1.0 corresponds to the original scheme. Default: 1.0.", false, 1.0f, "float", cmd);
	TCLAP::ValueArg<int> iterations_arg("i", "iter", "Set the number of iterations for Structure Tensors. Default: 60.", false, 60, "int", cmd);
	TCLAP::ValueArg<float> scale_arg("t", "scale", "Set the t ('scale') parameter. Default: 0.0001.", false, 0.0001f, "float", cmd);
	TCLAP::ValueArg<float> radius_arg("r", "radius", "Set the R ('radius') parameter. Default: 100.0.", false, 100.0f, "float", cmd);
	TCLAP::ValueArg<float> viz_arg("v", "viz", "Set the visualization coefficient for similarity values. Default: 3.0.", false, 3.0f, "float", cmd);
	TCLAP::SwitchArg raw_arg("", "raw", "Output raw distances in txt format.", cmd);
	TCLAP::ValueArg<string> output_arg("o", "output", "Set the name for output file(s) without extension.", false, "out", "string", cmd);
	TCLAP::ValueArg<string> point_arg("p", "point", "Set the point of interest (e.g. '-p 86:70').", true, "", "x:y", cmd);
	TCLAP::UnlabeledValueArg<string> source_image_arg("source", "Source image containing a point of interest.", true, string(), "file name", cmd);
	TCLAP::UnlabeledValueArg<string> target_image_arg("target", "Target image for which similarity map should be computed. If omitted, source image is used instead.", false, string(), "file name", cmd);

	// Parse command line arguments
	try {
		cmd.parse(argc, argv);
	} catch (TCLAP::ArgException &e) {  // catch any exceptions
		std::cerr << "Error: " << e.error() << " for arg " << e.argId() << std::endl;
		return 1;
	}

	// Get names
	string source_image_name = source_image_arg.getValue();
	string target_image_name = (target_image_arg.isSet()) ? target_image_arg.getValue() : source_image_name;
	string output_name = output_arg.getValue();

	bool distinct_images = source_image_name != target_image_name;

	// Get parameters
	Point point = iohelpers::parse_point(point_arg.getValue());
	float radius = radius_arg.getValue();
	float scale = scale_arg.getValue();
	int number_of_iterations = iterations_arg.getValue();
	float gamma = gamma_arg.getValue();
	float max_size_limit = size_limit_arg.getValue();
	int grid_size = grid_size_arg.getValue();
	float viz = viz_arg.getValue();
	bool is_raw_output = raw_arg.getValue();

	// Read the images
	Image<float> source_image = IOUtility::read_mono_image(source_image_name);
	Image<float> target_image = (distinct_images) ? IOUtility::read_mono_image(target_image_name) : source_image;

	if (!source_image) {
		std::cerr << "Could not open image '" << source_image_name << "'" << std::endl;
		return 1;
	}
	if (distinct_images && !target_image) {
		std::cerr << "Could not open image '" << target_image_name << "'" << std::endl;
		return 1;
	}

	if (point.x < 0) {
		point.x = source_image.size_x() / 2;
		point.y = source_image.size_y() / 2;
		std::cout << "Wrong format of point argument: '" << point_arg.getValue() << "'.\n";
		std::cout << "Use central point " << point.x << ":" << point.y << " instead.\n";
	}

	auto time_start = std::chrono::system_clock::now();

	// Create StructureTensor calculator
	msas::StructureTensor structure_tensor(radius, number_of_iterations, gamma);
    structure_tensor.set_max_size_limit(max_size_limit);

	// Create Structure Tensor bundles (fields)
	msas::StructureTensorBundle source_bundle(source_image, structure_tensor);
	msas::StructureTensorBundle *target_bundle = (distinct_images) ?
												 new msas::StructureTensorBundle(target_image, structure_tensor) :
												 &source_bundle;

	// Create patch distance calculator
	msas::AffinePatchDistance patch_distance(grid_size);
	patch_distance.set_scale(scale);
	patch_distance.precompute_normalized_patches(source_bundle);
	if (distinct_images) {
		patch_distance.precompute_normalized_patches(*target_bundle);
	}

	// Compute distances
	Image<float> distances(source_image.size());
	float min_distance = std::numeric_limits<float>::max();
	float max_distance = 0.0f;
	#pragma omp parallel for schedule(dynamic,1) collapse(2) shared(distances) reduction(max:max_distance), reduction(min:min_distance)
	for (uint y = 0; y < source_image.size_y(); ++y) {
		for (uint x = 0; x < source_image.size_x(); ++x) {
			msas::DistanceInfo distance_info = patch_distance.calculate(source_bundle, point, *target_bundle, Point(x, y));
			float distance = std::sqrt(distance_info.distance);

			if (x != point.x && y != point.y) {
				min_distance = std::min(min_distance, distance);
			}
			max_distance = std::max(max_distance, distance);

			distances(x, y) = distance;
		}
	}

	// Release target bundle, if it does not point to source bundle
	if (distinct_images) {
		delete target_bundle;
	}

	auto time_end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = time_end - time_start;
	std::cout << "Computation has finished in " << elapsed_seconds.count() << " seconds." << std::endl;

	if (!is_raw_output) {
		// Convert distances into similarities for visualization
		Image<float> similarities(source_image.size());
		similarities.set_color_space(ColorSpaces::mono);
		float *sim_data = similarities.raw();
		float *dist_data = distances.raw();
		float sigma = (max_distance - min_distance) / viz;
		float denominator = 2.0f * sigma * sigma;
		for (int i = 0; i < distances.raw_length(); ++i) {
			sim_data[i] = 255.0f * std::exp(-(dist_data[i] - min_distance) * (dist_data[i] - min_distance) / denominator);
		}

		// Draw similarities
		string sim_name = output_name + ".png";
		IOUtility::write_mono_image(sim_name, similarities);
	} else {
		// Output raw distances
		string dist_name = output_name + ".txt";
		iohelpers::write_as_text(dist_name, distances);
	}

	return 0;
}