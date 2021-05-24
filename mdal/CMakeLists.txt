set(EXTENSION_SRC
    PyMesh.cpp
    PyMesh.hpp
    )

set(extension "libmdalpython")
add_cython_target(${extension} "libmdalpython.pyx" CXX PY3)

add_library(${extension} MODULE ${EXTENSION_SRC} libmdalpython)
target_include_directories( ${extension}
    PRIVATE
    .
    ${MDAL_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${Python3_NumPy_INCLUDE_DIRS})

target_link_libraries(${extension} ${MDAL_LIBRARIES})
python_extension_module(${extension})

install(TARGETS ${extension} LIBRARY DESTINATION "mdal")


