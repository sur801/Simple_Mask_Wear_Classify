cmake_minimum_required(VERSION 3.10)

PROJECT(Mask_Classify)

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O3 -Ofast")
ADD_COMPILE_OPTIONS( -g -fPIC )

include_directories(
	${CMAKE_SOURCE_DIR}/common/include/libdrm
	${CMAKE_SOURCE_DIR}/common/include/libdrm/libdrm
	${CMAKE_SOURCE_DIR}/common/include
	${CMAKE_SOURCE_DIR}/common/include/libv4l2
	${CMAKE_SOURCE_DIR}/common/include/rga
	${CMAKE_SOURCE_DIR}/inc
)


set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/common/libs)
set(RKNN_API_LIB ${RKNN_API_PATH}/librknn_api.so)
set(RGA_API_PATH ${CMAKE_SOURCE_DIR}/common/libs)
set(RGA_API_LIB ${RGA_API_PATH}/librga.so)
set(DRM_API_PATH ${CMAKE_SOURCE_DIR}/common/libs)
set(DRM_API_LIB ${DRM_API_PATH}/libdrm.so)
set(LIBV4L2_PATH ${CMAKE_SOURCE_DIR}/common/libs)
set(LIBV4L2_LIB ${LIBV4L2_PATH}/libv4l2.so ${LIBV4L2_PATH}/libv4lconvert.so)



file(GLOB_RECURSE SRC_FILES
	${CMAKE_SOURCE_DIR}/common/src/*.cc
	${CMAKE_SOURCE_DIR}/src/*.c
	${CMAKE_SOURCE_DIR}/src/*.cc
	${CMAKE_SOURCE_DIR}/src/*.cpp
)

add_executable(${CMAKE_PROJECT_NAME} 
	 ${SRC_FILES}
)

target_link_libraries(${CMAKE_PROJECT_NAME}
    ${DRM_API_LIB} 
    ${RKNN_API_LIB} 
    ${USB_API_LIB} 
    ${RGA_API_LIB}
    ${LIBV4L2_LIB}
    ${LIBCONFIG_LIB}
    ${LIBAIO_LIB}
    ${LIBUSBGX_LIB}
    pthread
    rt  
    dl 
 ) 
