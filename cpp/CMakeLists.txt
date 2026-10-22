cmake_minimum_required(VERSION 3.18)
project(bevdet_metis LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(yaml-cpp REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(OpenSSL REQUIRED COMPONENTS Crypto)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OpenBLAS REQUIRED IMPORTED_TARGET openblas)

if(DEFINED ENV{AXRUNTIME_ROOT})
    set(AXRUNTIME_ROOT "$ENV{AXRUNTIME_ROOT}" CACHE PATH "AxRuntime installation root")
else()
    set(AXRUNTIME_ROOT "/opt/axelera" CACHE PATH "AxRuntime installation root")
endif()
file(GLOB AXRUNTIME_INSTALLATIONS LIST_DIRECTORIES TRUE "${AXRUNTIME_ROOT}/runtime-*")
list(SORT AXRUNTIME_INSTALLATIONS COMPARE NATURAL ORDER DESCENDING)
find_path(AXRUNTIME_INCLUDE_DIR axruntime/axruntime.h
    HINTS ${AXRUNTIME_ROOT} ${AXRUNTIME_INSTALLATIONS} PATH_SUFFIXES include)
find_library(AXRUNTIME_LIBRARY axruntime
    HINTS "${AXRUNTIME_INCLUDE_DIR}/.." ${AXRUNTIME_ROOT} ${AXRUNTIME_INSTALLATIONS} PATH_SUFFIXES lib lib64)
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(AxRuntime REQUIRED_VARS AXRUNTIME_INCLUDE_DIR AXRUNTIME_LIBRARY)
add_library(axruntime::axruntime UNKNOWN IMPORTED)
set_target_properties(axruntime::axruntime PROPERTIES IMPORTED_LOCATION "${AXRUNTIME_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${AXRUNTIME_INCLUDE_DIR}")

add_library(bevdet_cpp STATIC
    src/common/config.cpp src/common/types.cpp src/common/dataset.cpp
    src/inference/aipu_context.cpp src/inference/aipu_inference.cpp
    src/preprocessing.cpp src/view_transform.cpp src/feature_resize.cpp
    src/postprocessing.cpp src/pipeline.cpp)
target_compile_features(bevdet_cpp PUBLIC cxx_std_17)
set_target_properties(bevdet_cpp PROPERTIES CXX_EXTENSIONS OFF)
target_include_directories(bevdet_cpp PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(bevdet_cpp SYSTEM PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(bevdet_cpp PUBLIC axruntime::axruntime opencv_core opencv_imgproc opencv_imgcodecs
    PRIVATE yaml-cpp::yaml-cpp nlohmann_json::nlohmann_json OpenSSL::Crypto PkgConfig::OpenBLAS)
target_compile_options(bevdet_cpp PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(bevdet_metis app/main.cpp)
target_link_libraries(bevdet_metis PRIVATE bevdet_cpp nlohmann_json::nlohmann_json)
target_compile_options(bevdet_metis PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
