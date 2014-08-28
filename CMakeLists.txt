CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(RawCalc)
SET(PRJ_NAME RawCalc)

FIND_PACKAGE(CUDA REQUIRED)

#TODO:Find solution for platform-independent way to find helper cuda utilities from CUDA samples directory instead of 
# using local copy of these files..
SET(CUDA_HELPER_DIR ${CMAKE_SOURCE_DIR}/cuda)
MESSAGE(STATUS ${CUDA_HELPER_DIR})


SET(NVCC_GPU_ARCH
    -gencode arch=compute_20,code=sm_20
    -gencode arch=compute_30,code=sm_30
    #   -gencode arch=compute_32,code=sm_32
    #   -gencode arch=compute_35,code=sm_35
    #   -gencode arch=compute_37,code=sm_37
    #   -gencode arch=compute_50,code=sm_50
    )


SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};${NVCC_GPU_ARCH})

IF(UNIX)
    SET(PLATFORM_SPECIFIC_LIBRARIES ${PLATFORM_SPECIFIC_LIBRARIES} rt)
ENDIF()
IF(WIN32)
    SET(PLATFORM_SPECIFIC_LIBRARIES )
ENDIF()

ADD_SUBDIRECTORY(src)
