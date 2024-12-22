cmake_minimum_required(VERSION 2.8)
project(104IO)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

# By default build in Release
if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, set by default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif ()

# Increase optimization level for RelWithDebInfo build type
if (CMAKE_C_FLAGS_RELWITHDEBINFO)
	string(REPLACE "-O2" "-O3" CMAKE_C_FLAGS_RELWITHDEBINFO "${CMAKE_C_FLAGS_RELWITHDEBINFO}")
	string(REPLACE "-O2" "-O3" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
endif ()

# Specify target's our own include directories.
include_directories(iio)
include_directories(include)

# Specify all the source files.
set(SOURCE_FILES
		iio/iio.c
		iio/iio.h
		include/tclap/Arg.h
		include/tclap/ArgException.h
		include/tclap/ArgTraits.h
		include/tclap/CmdLine.h
		include/tclap/CmdLineInterface.h
		include/tclap/CmdLineOutput.h
		include/tclap/Constraint.h
		include/tclap/DocBookOutput.h
		include/tclap/HelpVisitor.h
		include/tclap/IgnoreRestVisitor.h
		include/tclap/MultiArg.h
		include/tclap/MultiSwitchArg.h
		include/tclap/OptionalUnlabeledTracker.h
		include/tclap/StandardTraits.h
		include/tclap/StdOutput.h
		include/tclap/SwitchArg.h
		include/tclap/UnlabeledMultiArg.h
		include/tclap/UnlabeledValueArg.h
		include/tclap/ValueArg.h
		include/tclap/ValuesConstraint.h
		include/tclap/VersionVisitor.h
		include/tclap/Visitor.h
		include/tclap/XorHandler.h
		include/tclap/ZshCompletionOutput.h
		include/tinydir.h
		include/io_utility.h
		io_utility.cpp)

# Override properties for specific files.
set_source_files_properties(iio/iio.c PROPERTIES COMPILE_FLAGS -std=c99)

# Specify all the targets, this target depends on.
set(DEPENDENCIES
		104Core)   # NOTE: these are other project to be built, NOT all the libs this target depends on.

# Specify other libs this target depends on.
SET(LIBS ${DEPENDENCIES}
		)

# Configure the target.
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)     # location for the target
add_library(${PROJECT_NAME} STATIC ${SOURCE_FILES})                 # static library
target_link_libraries(${PROJECT_NAME} ${LIBS})
get_filename_component(ABSOLUTE_INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include ABSOLUTE)
set(${PROJECT_NAME}_INCLUDE_DIRS ${ABSOLUTE_INCLUDE_PATH} CACHE INTERNAL "${PROJECT_NAME}: Include Directories" FORCE)

# Add dependencies themself and include directories from the dependencies.
foreach (DEP ${DEPENDENCIES})
	add_dependencies(${PROJECT_NAME} ${DEP})
	include_directories(${${DEP}_INCLUDE_DIRS})
endforeach (DEP)
