include(CheckLanguage)
check_language(CUDA)
enable_language(CUDA)

project(TRTInferTests LANGUAGES CXX CUDA)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")

set( CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib )
set( CMAKE_CUDA_FLAGS "-cudart static -O2" )

add_library(InferLib SHARED src/trtinfer.cu)
target_link_libraries(InferLib nvinfer nvonnxparser)
set_target_properties(InferLib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(InferLib PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)

include(GNUInstallDirs)

find_package(CUDA REQUIRED)

include_directories("${CUDA_INCLUDE_DIRS}")
include_directories("${PROJECT_SOURCE_DIR}/include")

link_directories(/lib/aarch64-linux-gnu/)
link_directories("${CMAKE_BINARY_DIR}/lib")

write_file(${PROJECT_SOURCE_DIR}/include/sourceLoc.h "#ifndef SOURCELOC_H\n#define SOURCELOC_H\nnamespace file_locations\n{\n\tchar project_source_dir[] = \"${PROJECT_SOURCE_DIR}/data/\";\n}\n#endif")

add_executable(1xDLAHybridInfer ${PROJECT_SOURCE_DIR}/tests/1xDLAHybridInfer.cu )
add_executable(2xDLAHybridInfer ${PROJECT_SOURCE_DIR}/tests/2xDLAHybridInfer.cu)
add_executable(GPUOnlyInfer ${PROJECT_SOURCE_DIR}/tests/GPUOnlyInfer.cu)
add_executable(GPUwith1xDLAHybrid ${PROJECT_SOURCE_DIR}/tests/GPUwith1xDLAHybrid.cu)
add_executable(GPUwith2xDLAHybrid ${PROJECT_SOURCE_DIR}/tests/GPUwith2xDLAHybrid.cu)
add_executable(MaxThroughputFPSCount ${PROJECT_SOURCE_DIR}/tests/MaxThroughputFPSCount.cu)

target_link_libraries(1xDLAHybridInfer InferLib)
target_link_libraries(2xDLAHybridInfer InferLib)
target_link_libraries(GPUOnlyInfer InferLib)
target_link_libraries(GPUwith1xDLAHybrid InferLib)
target_link_libraries(GPUwith2xDLAHybrid InferLib)
target_link_libraries(MaxThroughputFPSCount InferLib)