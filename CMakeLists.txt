cmake_minimum_required(VERSION 3.0)
project(mobilenetknn)

SET(Caffe_DIR /home/rahul/caffe)

include_directories(${Caffe_DIR}/include ${CMAKE_CURRENT_SOURCE_DIR}/include)

file(GLOB_RECURSE SOURCE_O ${CMAKE_CURRENT_SOURCE_DIR}/lib/*.o SRC src/*.cpp)

add_library(${PROJECT_NAME}
	    STATIC ${SRC}
            ${SOURCE_O})

add_definitions(-DCPU_ONLY)
target_link_libraries(${PROJECT_NAME} ${Caffe_DIR}/lib/libcaffe.a)

SET(EXAMPLES
         program
)
# License source files
find_package(OpenMP)
if(OPENMP_FOUND)
	set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
find_package( OpenCV REQUIRED)
set(CMAKE_BUILD_TYPE release)
set(CMAKE_CXX_STANDARD 11)

add_definitions(-DCPU_ONLY)

find_package(HDF5 COMPONENTS HL REQUIRED)
include_directories(SYSTEM ${HDF5_INCLUDE_DIRS} ${HDF5_HL_INCLUDE_DIR})
list(APPEND Caffe_LINKER_LIBS ${HDF5_LIBRARIES})
list(APPEND Caffe_LINKER_LIBS ${HDF5_HL_LIBRARIES})

include_directories(${OpenCV_DIR}/include)
foreach (example ${EXAMPLES})
    add_executable(${example} ${example}.cpp)
    TARGET_LINK_LIBRARIES(${example} ${OpenCV_LIBS} ${CMAKE_CURRENT_LIST_DIR}/build/libmobilenetknn.a ${HDF5_LIBRARIES} ${HDF5_HL_LIBRARIES} -lopenblas -lprotobuf -lboost_system -lglog -lgflags -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_ml -lopencv_imgcodecs -lopencv_videoio  -lboost_system -lboost_regex -lboost_filesystem -lboost_thread -lboost_log)
endforeach()


