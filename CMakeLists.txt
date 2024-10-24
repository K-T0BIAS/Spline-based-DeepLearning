cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(SplineNetLib VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optionally enable warnings for all compilers
if(MSVC)
    add_compile_options(/W4)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Add the include directory so other projects can use headers
include_directories(${PROJECT_SOURCE_DIR}/include)

# Add the library from the source files
add_library(SplineNetLib
    src/SplineNet.cpp
    src/layers.cpp
    src/splines.cpp
)

# Specify the include directories for the library target
target_include_directories(SplineNetLib PUBLIC ${PROJECT_SOURCE_DIR}/include)

# Add an example or test executable (optional)
add_executable(SplineNetExample examples/example_network.cpp)

# Link the example with the library
target_link_libraries(SplineNetExample PRIVATE SplineNetLib)

# Optional: Install the library and headers
install(TARGETS SplineNetLib DESTINATION lib)
install(DIRECTORY include/SplineNetLib DESTINATION include)

# Include CMakePackageConfigHelpers for package configuration
include(CMakePackageConfigHelpers)

# Generate the version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/SplineNetLibConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Install the package configuration file
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/SplineNetLibConfigVersion.cmake" DESTINATION lib/cmake/SplineNetLib)