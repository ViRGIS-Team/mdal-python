"""
/******************************************************************************
* Copyright (c) 2021, Runette Software Ltd (www.runette.co.uk)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Runette Software nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/
"""

from mdal import Datasource, Info, last_status, PyMesh, drivers

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

    meshio = mesh.meshio()
    mesh.save("save_test.ply")

ds2 = Datasource("data/ply/all_features.ply")

with ds2.load(0) as mesh:
    mesh.save("save_test_2.ply")

    ds3 = Datasource("save_test_2.ply")

    with ds3.load(0) as mesh2:
        print(f"Save equality 2 : {mesh == mesh2}")

print(meshio)
print(mesh)
print(mesh.vertex_count)
