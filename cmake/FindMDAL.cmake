# Find MDAL
# ~~~~~~~~~
# Copyright (c) 2018, Peter Petrik <zilolv at gmail dot com>
# Redistribution and use is allowed according to the terms of the BSD license.
# For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#
#
# Once run this will define: 
#  MDAL_FOUND - System has MDAL
#  MDAL_INCLUDE_DIRS - The MDAL include directories
#  MDAL_LIBRARIES - The libraries needed to use MDAL
#  MDAL_DEFINITIONS - Compiler switches required for using MDAL

FIND_PACKAGE(PkgConfig)
PKG_CHECK_MODULES(PC_MDAL QUIET mdal libmdal)
SET(MDAL_DEFINITIONS ${PC_MDAL_CFLAGS_OTHER})

FIND_PATH(MDAL_INCLUDE_DIR mdal.h
          PATHS $ENV{CONDA_PREFIX}/include $ENV{LIB_DIR}/include ${PC_MDAL_INCLUDEDIR} ${PC_MDAL_INCLUDE_DIRS} $ENV{PREFIX}\\Library\\include $ENV{PREFIX}/include
          PATH_SUFFIXES libmdal )

message(STATUS "MDAL_INCLUDE=${MDAL_INCLUDE_DIR}")

FIND_LIBRARY(MDAL_LIBRARY NAMES mdal
             PATHS $ENV{CONDA_PREFIX}/lib $ENV{LIB_DIR}/lib ${PC_MDAL_LIBDIR} ${PC_MDAL_LIBRARY_DIRS} $ENV{PREFIX}\\Library\\lib $ENV{PREFIX}/lib)

message(STATUS "MDAL_LIBRARY=${MDAL_LIBRARY}")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(MDAL DEFAULT_MSG
                                  MDAL_LIBRARY MDAL_INCLUDE_DIR)

MARK_AS_ADVANCED(MDAL_INCLUDE_DIR MDAL_LIBRARY )

SET(MDAL_LIBRARIES ${MDAL_LIBRARY} )
SET(MDAL_INCLUDE_DIRS ${MDAL_INCLUDE_DIR} )
