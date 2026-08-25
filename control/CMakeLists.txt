cmake_minimum_required(VERSION 3.15...3.27)
project(duke_humanoid_v2_control
  LANGUAGES C CXX)
set (CMAKE_CXX_STANDARD 17)

set(CMAKE_POSITION_INDEPENDENT_CODE ON) 


# Find Python and required components
if (CMAKE_VERSION VERSION_LESS 3.18)
  set(DEV_MODULE Development)
else()
  set(DEV_MODULE Development.Module)
endif()

find_package(Python 3.12
  COMPONENTS Interpreter ${DEV_MODULE} REQUIRED
  OPTIONAL_COMPONENTS Development.SABIModule)

# message(STATUS "Python executable: ${PYTHON_EXECUTABLE}")

# Set default build type to Release
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)

# echo "nanobind_ROOT: ${nanobind_ROOT}"
message(STATUS "nanobind_ROOT: ${nanobind_ROOT}")
find_package(nanobind CONFIG REQUIRED)

find_package(Eigen3 CONFIG REQUIRED)
find_package(CSerialPort CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
# 2026-08-22: TBB (find_package + the vcpkg port) dropped — no source or target
# ever used it; every fresh checkout was compiling it from source for nothing.

add_executable(sleep_test ipc/sleep_test.cpp)

# Hardware bindings (motor, imu, ft_servo): sources inlined at control/hardware_bindings/
# (internally a git submodule). Built via add_subdirectory; each .abi3.so lands NEXT TO
# its sources (hardware_bindings/<motor|imu|ft_servo>/), where the Python wrappers load it.
add_subdirectory(hardware_bindings)
