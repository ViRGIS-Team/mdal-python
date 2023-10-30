Changes
--------------------------------------------------------------------------------

2.0.0
-----

UNDER DEVELOPMENT

- fix memory leaks and inconsistencies around the Datagroup object
- remove Open3D integration to separate package

1.0.0
-----

First Read / Write Release

- read and write all MDAL compatible file formats
- access vertex, face, edge and volume data as numpy arrays
- write vertex, face, edge and volume data from numpy arrays
- access and write scalar and vector datasets
- beta level read and write integration with meshio
- beta level read integration with Open3D


0.9.0
-----

First release. This is beta software and has not been completely tested yet:

Currently, this integration can:

- read all MDAL compatible file formats,
- access the metadata for the source,
- access the vertex, face and edge data as numpy arrays,
- access 'double' datasets (both scalar and vector) as numpy arrays, and
- convert the MDAL source mesh into a `meshio`_ mesh object (with some restrictions currently).

This version does not currently allow the MDAL source mesh to be written or ammended.
