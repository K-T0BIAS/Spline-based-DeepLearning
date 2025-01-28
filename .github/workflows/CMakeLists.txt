# CMakeLists.txt for Tests

#Add more test files like this
#set(TEST_SOURCES 
#    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/network_tests.cpp
#    ${CMAKE_CURRENT_SOURCE_DIR}/unit_tests/other_tests.cpp
#)

# Set the source files for the tests in a variable (you can easily add more test files here later)
set(TEST_SOURCES 
    ${CMAKE_CURRENT_SOURCE_DIR}tests/unit_tests/network_tests.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}tests/unit_tests/spline_tests.cpp
)

# Add an executable for the tests (it will compile all the test sources)
add_executable(SplineNetLibTests ${TEST_SOURCES})

# Link the executable to your main library
target_link_libraries(SplineNetLibTests PRIVATE SplineNetLib)

# Enable testing
include(CTest)
enable_testing()

# Add the test to CTest
add_test(NAME SplineNetLibTests COMMAND SplineNetLibTests)