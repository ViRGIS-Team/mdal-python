cmake_minimum_required(VERSION 3.11.0)
project(MDAL)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
option(WITH_TESTS "Enable tests" OFF)
set(CMAKE_BUILD_TYPE "Release")

# Python-finding settings
set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_REGISTRY "LAST")
set(Python3_FIND_FRAMEWORK "LAST")
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

message(STATUS "${CMAKE_MODULE_PATH}")

# find MDAL. 
find_package(MDAL REQUIRED)

if(SKBUILD)
  find_package(PythonExtensions REQUIRED)
  find_package(Cython REQUIRED)
  message(STATUS "The project is built using scikit-build")
endif()

add_subdirectory(mdal)

# Taken and adapted from PDAL's cmake macros.cmake

function(mdal_python_target_compile_settings target)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD 11)
    set_property(TARGET ${target} PROPERTY CXX_STANDARD_REQUIRED TRUE)
    target_compile_definitions(${target} PRIVATE
        -DWIN32_LEAN_AND_MEAN)
    if (MSVC)
        # check for MSVC 8+
        if (NOT (MSVC_VERSION VERSION_LESS 1400))
            target_compile_definitions(${target} PRIVATE
                -D_CRT_SECURE_NO_DEPRECATE
                -D_CRT_SECURE_NO_WARNINGS
                -D_CRT_NONSTDC_NO_WARNING
                -D_SCL_SECURE_NO_WARNINGS
            )
            target_compile_options(${target} PRIVATE
                # Yes, we don't understand GCC pragmas
                /wd4068
                # Nitro makes use of Exception Specifications, which results in
                # numerous warnings when compiling in MSVC. We will ignore
                # them for now.
                /wd4290
                /wd4800
                # Windows warns about integer narrowing like crazy and it's
                # annoying.  In most cases the programmer knows what they're
                # doing.  A good static analysis tool would be better than
                # turning this warning off.
                /wd4267
                # Annoying warning about function hiding with virtual
                # inheritance.
                /wd4250
                # some templates don't return
#                /wd4716
                # unwind semantics
#                /wd4530
                # Standard C++-type exception handling.
                /EHsc
                )
        endif()

    endif()
endfunction()




if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # For newer versions of python (3.8+), C extensions don't link against
    # libpython and instead get symbol definitions from the python interpreter
    # executable. PDAL plugins need to link against libpython, but if a plugin
    # is loaded inside a python process, it must resolve symbols from the python
    # executable instead of libpython. Using flat namespace allows that.
    set(PYTHON_LINK_LIBRARY ${PYTHON_LINK_LIBRARY} -Wl,-flat_namespace)
endif()



if (WITH_TESTS)
    enable_testing()
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    add_subdirectory(pdal/test/gtest)
    enable_testing()
    include_directories(pdal/test/gtest/include .. ${CMAKE_CURRENT_BINARY_DIR})




    PDAL_PYTHON_ADD_TEST(pdal_io_numpy_test
        FILES
            ./pdal/test/NumpyReaderTest.cpp
            ./pdal/test/Support.cpp
            ./pdal/plang/Invocation.cpp
            ./pdal/plang/Environment.cpp
            ./pdal/plang/Redirector.cpp
            ./pdal/plang/Script.cpp
        LINK_WITH
            ${numpy_reader}
            ${Python3_LIBRARIES}
            ${PDAL_LIBRARIES}
            ${CMAKE_DL_LIBS}
        SYSTEM_INCLUDES
            ${PDAL_INCLUDE_DIRS}
            ${Python3_INCLUDE_DIRS}
            ${Python3_NumPy_INCLUDE_DIRS}
    )

    PDAL_PYTHON_ADD_TEST(pdal_filters_python_test
        FILES
            ./pdal/test/PythonFilterTest.cpp
            ./pdal/test/Support.cpp
            ./pdal/plang/Invocation.cpp
            ./pdal/plang/Environment.cpp
            ./pdal/plang/Redirector.cpp
            ./pdal/plang/Script.cpp
        LINK_WITH
            ${python_filter}
            ${Python3_LIBRARIES}
            ${PDAL_LIBRARIES}
            ${CMAKE_DL_LIBS}
        SYSTEM_INCLUDES
            ${PDAL_INCLUDE_DIRS}
            ${Python3_INCLUDE_DIRS}
            ${Python3_NumPy_INCLUDE_DIRS}
    )
endif (WITH_TESTS)
