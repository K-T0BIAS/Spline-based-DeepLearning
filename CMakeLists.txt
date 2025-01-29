cmake_minimum_required(VERSION 3.10)

# Set the project name and version
project(SplineNetLib VERSION 1.0)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

#include fetch content for catch2
include(FetchContent)

# fetch catch2 and make usable
FetchContent_Declare(
    Catch2
    GIT_REPOSITORY https://github.com/catchorg/Catch2.git
    GIT_TAG v3.4.0
)
FetchContent_MakeAvailable(Catch2)

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

optional(ENABLE_TESTS "allow catch2 install and tests to run" OFF)

if(ENABLE_TESTS)

    #enable testing
    enable_testing()
    
    #Add test exe
    add_executable(SplineNetTests
        tests/unit_tests/spline_tests.cpp
    )
    
    #link test exe with library
    target_link_libraries(SplineNetTests PRIVATE 
        SplineNetLib
        Catch2::Catch2WithMain
    )
    
    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)
    include(Catch)
    catch_discover_tests(SplineNetTests)
 endif()   

# Add an example or test executable 
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


