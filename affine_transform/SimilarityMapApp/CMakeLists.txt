cmake_minimum_required(VERSION 2.8)
project(SimilarityMapApp)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif ()

# By default build in Release
if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, set by default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# Specify target's our own include directories.
include_directories(include)
include_directories(../Eigen)

# Specify all the source files.
set(SOURCE_FILES
		main.cpp
		include/io_helpers.h)

# Specify all the targets, this target depends on.
set(DEPENDENCIES
		104Core
		104IO
		104Algo
		MSAS)   # NOTE: these are other project to be built, NOT all the libs this target depends on.

# Specify other libs this target depends on.
SET(LIBS ${DEPENDENCIES}
		tiff
		png
		jpeg)

# Configure the target.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)     # location for the target
add_executable(${PROJECT_NAME} ${SOURCE_FILES})                     # executable
target_link_libraries(${PROJECT_NAME} ${LIBS})

# Add dependencies themself and include directories from the dependencies.
foreach (DEP ${DEPENDENCIES})
	add_dependencies(${PROJECT_NAME} ${DEP})
	include_directories(${${DEP}_INCLUDE_DIRS})
endforeach (DEP)