cmake_minimum_required(VERSION 3.20)
project(srqalv LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
find_package(Python 3.11 COMPONENTS Interpreter Development.Module REQUIRED)
execute_process(COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  COMMAND_ERROR_IS_FATAL ANY)
find_package(nanobind CONFIG REQUIRED)

add_library(btkernel STATIC src/wigner6j.cpp src/left_comb.cpp
  src/bt_kernel.cpp src/volume_srq.cpp)
set_target_properties(btkernel PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(btkernel PUBLIC include)
target_compile_options(btkernel PRIVATE
  $<$<CXX_COMPILER_ID:AppleClang,Clang,GNU>:-O3>)

nanobind_add_module(_core src/python_bindings.cpp)
target_link_libraries(_core PRIVATE btkernel)
install(TARGETS _core LIBRARY DESTINATION srqalv/_native
  RUNTIME DESTINATION srqalv/_native)
