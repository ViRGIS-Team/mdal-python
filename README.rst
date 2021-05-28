================================================================================
MDAL Python Integration
================================================================================

Basics
------

MDAL Python integration allows you to access and manipulation geospatial mesh data sets using `MDAL`_ in Python.

Currently, this integration can:

- read all MDAL compatible file formats,
- access the metadata for the source,
- access the vertex, face and edge data as numpy arrays,
- access 'double' datasets (both scalar and vector) as numpy arrays, and
- convert the MDAL source mesh into a `meshio`_ mesh object (with some restrictions currently).

This version does not currently allow the MDAL source mesh to be written or ammended.

.. _MDAL: https://www.mdal.xyz/
.. _meshio: https://github.com/nschloe/meshio

Drivers
.......



Installation
------------

Conda
................................................................................

MDAL Python support is installable via Conda:

.. code-block::

    conda install mdal-python

GitHub
................................................................................

The repository for MDAL's Python extension is available at https://github.com/ViRGIS-Team/mdal-python

Usage
--------------------------------------------------------------------------------

The basic usage can be seen in this code snippet:

.. code-block:: python

    import mdal
    from mdal import Datasource

    print(f"MDAL Version:  {mdal.getVersionString()}")
    print(f"MDAL Driver Count :{mdal.getDriverCount()}")
    print(mdal.getLastStatus())

    for driver in mdal.getDrivers():
        print(driver.long_name)


    ds = Datasource("data/ply/test_mesh.ply")
    print(ds.meshes)

    mesh = ds.load(0)
    print(f"Driver : {mesh.driverName}")
    print(f"Vertex Count : {mesh.vertexCount}")
    print(f"Face Count : {mesh.faceCount}")
    print(f"Largest Face: {mesh.largestFace}")
    print(f"Edge Count : {mesh.edgeCount}")
    print(f"CRS : {mesh.projection}")
    print(f"Mesh extent : {mesh.extent}")
    print(f"DatasetGroup Count : {mesh.groupCount}")
    print("")

    mesh = ds.load(ds.meshes[0])
    print(f"Driver : {mesh.driverName}")
    print(f"Vertex Count : {mesh.vertexCount}")
    print(f"Face Count : {mesh.faceCount}")
    print(f"Largest Face: {mesh.largestFace}")
    print(f"Edge Count : {mesh.edgeCount}")
    print(f"CRS : {mesh.projection}")
    print(f"Mesh extent : {mesh.extent}")

    vertex = mesh.getVertices()
    print(f"Vertex Array Shape : {vertex.shape}")

    faces = mesh.getFaces()
    print(f"Face Array Shape : {faces.shape}")

    edges = mesh.getEdges()
    print(f"Edges Array Shape : {edges.shape}")

    print("")

    group = mesh.getGroup(0)
    print(f"DatasetGroup Name : {group.name}")
    print(f"DatasetGroup Location : {group.location}")
    print(f"Dataset Count : {group.datasetCount}")
    print(f"Group has scalar values : {group.hasScalar}")
    print(f"Group has temporal values : {group.isTemporal}")
    print(f"Reference Time : {group.referenceTime}")
    print(f"Maximum Vertical Level Count : {group.levelCount}")
    print(f"Minimum / Maximum ; {group.minmax}")
    print(f"Metadata : {group.getMetadata()}")

    print("")

    data = group.getDataAsDouble(0)
    print(f"Dataset Shape : {data.shape}")

Documentation
-------------

The documentation is currently WIP and can be found at https://virgis-team.github.io/mdal-python/


Requirements
------------

* MDAL 0.8.0 +
* Python >=3.6
* Cython (eg :code:`pip install cython`)
* Numpy (eg :code:`pip install numpy`)
* Packaging (eg :code:`pip install packaging`)
* scikit-build (eg :code:`pip install scikit-build`)


Credit
------

This package borrowed heavily from the `PDAL-Python`_ package.

.. _PDAL-Python:  https://github.com/PDAL/python
