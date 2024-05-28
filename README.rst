================================================================================
MDAL Python Integration
================================================================================

.. image:: https://img.shields.io/conda/vn/conda-forge/mdal-python.svg
   :target: https://anaconda.org/conda-forge/mdal-python
   
.. image:: https://badge.fury.io/py/mdal.svg
   :target: https://badge.fury.io/py/mdal

Basics
------

MDAL Python integration allows you to access and manipulation geospatial mesh data sets using `MDAL`_ in Python.

Currently, this integration can:

- read and write all MDAL compatible file formats
- access vertex, face, edge and volume data as numpy arrays
- write vertex, face, edge and volume data from numpy arrays
- access and write scalar and vector datasets
- beta level read and write integration with `meshio`_
- beta level read integration with `Open3D`_


.. _MDAL: https://www.mdal.xyz/
.. _meshio: https://github.com/nschloe/meshio
.. _Open3D: http://www.open3d.org/

Drivers
.......

['2DM Mesh File', 'XMS Tin Mesh File', 'Selafin File', 'Esri TIN', 'Stanford PLY Ascii Mesh File', 'Flo2D', 'HEC-RAS 2D', 'TUFLOW FV', 'AnuGA', 'UGRID Results', 'GDAL NetCDF', 'GDAL Grib', 'DAT', 'Binary DAT', 'TUFLOW XMDF', 'XDMF']

Installation
------------

Conda
................................................................................

MDAL Python support is installable via Conda:

.. code-block::

    conda install -c conda-forge mdal-python

PyPI
...............................................................................

MDAL Python support can be installed using `pip`

.. note::

    The previous mdal-python package is deprecated and will not updated beyond 1.0.3. Use the mdal package instead.

.. code-block::

   pip install mdal
   
This will ONLY work if there is a valid and working installation of MDAL on the device and accessible through the device library search path.

.. note::

    As of version 1.1.0, this package has been written to the NumPy 2.0 ABI. This should work with any version of NumPy >= 1.25 and Python >=3.9


GitHub
................................................................................

The repository for MDAL's Python extension is available at https://github.com/ViRGIS-Team/mdal-python

Usage
--------------------------------------------------------------------------------

The basic usage can be seen in this code snippet:

.. code-block:: python


    from mdal import Datasource, Info, last_status, PyMesh, drivers, MDAL_DataLocation

    print(f"MDAL Version:  {Info.version}")
    print(f"MDAL Driver Count :{Info.driver_count}")
    print(last_status().name)

    for driver in Info.drivers:
        print(driver)


    ds = Datasource("data/ply/test_mesh.ply")
    print(ds.meshes)

    with ds.load(0) as mesh:
        print(f"Driver : {mesh.driver_name}")
        print(f"Format : {mesh.get_metadata('format')}")
        print(f"Vertex Count : {mesh.vertex_count}")
        print(f"Face Count : {mesh.face_count}")
        print(f"Largest Face: {mesh.largest_face}")
        print(f"Edge Count : {mesh.edge_count}")
        print(f"CRS : {mesh.projection}")
        print(f"Mesh extent : {mesh.extent}")
        print(f"Metadata : {mesh.metadata}")
        print(f"CRS Metadata : {mesh.get_metadata('crs')}")
        mesh.add_metadata("test", "value")
        print(f"Metadate set eqiuality : {mesh.get_metadata('test') == 'value'}")

        vertex = mesh.vertices
        print(f"Vertex Array Shape : {vertex.shape}")

        faces = mesh.faces
        print(f"Face Array Shape : {faces.shape}")

        edges = mesh.edges
        print(f"Edges Array Shape : {edges.shape}")

        print("")

        group = mesh.group(0)
        print(f"DatasetGroup Name : {group.name}")
        print(f"DatasetGroup Location : {group.location.name}")
        print(f"Dataset Count : {group.dataset_count}")
        print(f"Group has scalar values : {group.has_scalar}")
        print(f"Group has temporal values : {group.is_temporal}")
        print(f"Reference Time : {group.reference_time}")
        print(f"Maximum Vertical Level Count : {group.level_count}")
        print(f"Minimum / Maximum ; {group.minmax}")
        print(f"Metadata : {group.metadata}")
        print(f"Name Metadata : {group.get_metadata('name')}")
        group.add_metadata("test", "value")
        print(
            f"Metadate set eqiuality : {group.get_metadata('test') == 'value'}")

        print("")
        for i in range(0, group.dataset_count):
            data = group.data(i)
            time = group.dataset_time(i)
            print(f"Dataset Shape for time {time} : {data.shape}")

        print("")

        test = PyMesh()
        test.vertices = mesh.vertices
        test.faces = mesh.faces
        test.edges = mesh.edges
        print(f"Mesh Copy Equality : {test == mesh}")
        print(
            f"Mesh Vertex Size equality: {test.vertex_count == mesh.vertex_count}")
        print(f"Mesh Face Size equality: {test.face_count == mesh.face_count}")
        test.save("data/save_test.nc")

        test2 = PyMesh(drivers()[0])
        print(f"Mesh created by Driver : {test2.driver_name}")

        ds2 = Datasource("data/save_test.nc")
        test4 = ds2.load(0)
        print(f"Save equality : {test4 == test}")

        del(test)
        del(test4)
        mesh.save("save_test.ply")

    with Datasource("data/ply/all_features.ply").load(0) as mesh:
        mesh.save("save_test_2.ply")

        with Datasource("save_test_2.ply").load(0) as mesh2:
            print(f"Save equality 2 : {mesh == mesh2}")

    with Datasource("data/tuflowfv/withMaxes/trap_steady_05_3D.nc").load() as mesh:
        group = mesh.groups[1]
        a, b, c = group.volumetric(0)

        ds2 = Datasource("test_vol.ply")
        with ds2.add_mesh() as mesh2:
            mesh2.vertices = mesh.vertices
            mesh2.faces = mesh.faces

            print(f"Vertex Count :{mesh.vertex_count}")
            print(f"Face Count : {mesh.face_count}")

            group2 = mesh2.add_group(
                "test", location=MDAL_DataLocation.DataOnVolumes)
            group2.add_volumetric(group.data(), a, b)

            print(f"Level Count: {group2.level_count}")
            print(f"Location: {group2.location}")
            print(f"MinMax: {group2.minmax}")

            print(f"Dataset Count: {group2.dataset_count}")

            data = group2.data(0)
            print(f"Data Value Count: {len(data)}")
            print(f"{data}")

            print(f"{group2.volumetric(0)}")

            a, b, c = group2.volumetric(0)
            print(f"Number of Extrusion values : {len(b)}")
            mesh2.save()
            with ds2.load() as mesh3:
                mesh3.info()
                group3 = mesh3.groups[1]
                print(f"{group3.location}")
                d, e, f = group3.volumetric(0)
                print(f"{group3.volumetric(0)}")
                print(f"{group3.data(0)}")
                print("Mesh Equality : {mesh2 == mesh3}")


    """deep copy test"""

    with Datasource("data/ply/all_features.ply").load() as mesh:
        with ds.add_mesh("test") as mesh2:
            mesh2.deep_copy(mesh)
            mesh2.data_copy(mesh)
            print(f"{mesh2.info()}")


    print("all finished !")


Integration with meshio
-----------------------

There is read and write integration with the meshio package. Any MDAL mesh
can be converted to a meshio object and vice versa.

This integration is beta at the moment.

There are the following constraints:

- MDAL_transform.to_meshio can take as an argument either a Mesh or a Dataset Group,
- Only scalar MDAL datasets can be converted to meshio,
- Volumetric data must be passed as a Dataset Group,
- Volumetric meshio meshes and data are not currently converted, and
- MDAL_transform.from_meshio only converts cells of types ["line", "triangle", "quad"].

.. code-block:: python

    from mdal import Datasource,MDAL_transform

    """meshio tests"""
    with Datasource("data/ply/all_features.ply").load() as mesh:

        mio = MDAL_transform.to_meshio(mesh)
        print(f"{mio}")
        mio.write("test.vtk")

        group = mesh.group(1)

        mio2 = MDAL_transform.to_meshio(group)
        print(f"{mio2}")
        
        mesh2 = MDAL_transform.from_meshio(mio)
        print(f"{mesh2.info()}")
        print(f"{mesh2.group(0).data()}")
        print(f"{mesh2.vertex_count}")
        print(f"{mesh2.face_count}")

    with Datasource("test_vol.ply").load() as mesh:
        group = mesh.group(1)
        mio2 = MDAL_transform.to_meshio(group)
        print(f"{mio2}")


    print("all finished !")

Integration with Open3D
-----------------------

There is read-only integration with Open3D.

The MDAL_transform.to_triangle_mesh function converts any MDAL mesh to an Open3D TriangleMesh. The function
can take as an argument an MDAL mesh or Dataset Group. In the former case 
if there are colour Datasets then these are converted to the TraingleMesh colours.
In the later case, the data is converted to a false colur using a simple process -
scalar data is loaded into the red values and vector data to
the red and blue values.

The MDAL_transform.to_point_cloud converts a MDAL
volumetric DatasetGroup to an Open3D PointCloud with the data values
converted to color as above.

.. note::
    Open3D is NOT loaded as dependency. If these commands are used in an environment without Open3D, they will fail with a user friendly message.

This integration is beta at the moment.

.. code-block:: python

    from mdal import Datasource, MDAL_transform

    import numpy as np
    import open3d as o3d

    """
    Open3d Tests
    """
    with Datasource("data/ply/test_mesh.ply").load() as mesh:
        tm = MDAL_transform.to_triangle_mesh(mesh)
        print(tm)
        tm2 = o3d.io.read_triangle_mesh("data/ply/test_mesh.ply")
        tmc = np.asarray(tm.vertex_colors)
        tmc2 = np.asarray(tm2.vertex_colors)
        for i in range(len(tmc)):
            value = tmc[i] - tmc2[i]
            if not (value == [0, 0, 0]).all():
                print(value)
                break

    with Datasource("test_vol.ply").load() as mesh:
        pc = MDAL_transform.to_point_cloud(mesh.group(1))
        print(pc)


    print("all finished !")

.. note::

    About Python Versions. MDAL supports 3.8, 3.9 and 3.10. Open3D supports 3.6, 3.7 and 3.8. Therefore, 
    if you want to use Open3D, the Python version should be pinned to 3.8 before you start.


Documentation
-------------

The documentation is currently WIP and can be found at https://virgis-team.github.io/mdal-python/html/index.html


Requirements
------------

* MDAL 0.9.0 +
* Python >=3.8
* Cython (eg :code:`pip install cython`)
* Numpy (eg :code:`pip install numpy`)
* Packaging (eg :code:`pip install packaging`)
* scikit-build (eg :code:`pip install scikit-build`)


Credit
------

This package borrowed heavily from the `PDAL-Python`_ package.

.. _PDAL-Python:  https://github.com/PDAL/python
