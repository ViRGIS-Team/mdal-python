from mdal import Datasource, Info, last_status

print(f"MDAL Version:  {Info.version}")
print(f"MDAL Driver Count :{Info.driver_count}")
print(last_status())

for driver in Info.drivers:
    print(driver)


ds = Datasource("data/ply/test_mesh.ply")
print(ds.meshes)

with ds.load(0) as mesh:
    print(f"Driver : {mesh.driver_name}")
    print(f"Vertex Count : {mesh.vertex_count}")
    print(f"Face Count : {mesh.face_count}")
    print(f"Largest Face: {mesh.largest_face}")
    print(f"Edge Count : {mesh.edge_count}")
    print(f"CRS : {mesh.projection}")
    print(f"Mesh extent : {mesh.extent}")
    print(f"DatasetGroup Count : {mesh.group_count}")

print("finish first pass")
print(f"Loading uri {ds.meshes[0]}")


with ds.load(ds.meshes[0]) as mesh:
    print(f"Driver : {mesh.driver_name}")
    print(f"Vertex Count : {mesh.vertex_count}")
    print(f"Face Count : {mesh.face_count}")
    print(f"Largest Face: {mesh.largest_face}")
    print(f"Edge Count : {mesh.edge_count}")
    print(f"CRS : {mesh.projection}")
    print(f"Mesh extent : {mesh.extent}")

    vertex = mesh.vertices
    print(f"Vertex Array Shape : {vertex.shape}")

    faces = mesh.faces
    print(f"Face Array Shape : {faces.shape}")

    edges = mesh.edges
    print(f"Edges Array Shape : {edges.shape}")

    print("")

    group = mesh.group(0)
    print(f"DatasetGroup Name : {group.name}")
    print(f"DatasetGroup Location : {group.location}")
    print(f"Dataset Count : {group.dataset_count}")
    print(f"Group has scalar values : {group.has_scalar}")
    print(f"Group has temporal values : {group.is_temporal}")
    print(f"Reference Time : {group.reference_time}")
    print(f"Maximum Vertical Level Count : {group.level_count}")
    print(f"Minimum / Maximum ; {group.minmax}")
    print(f"Metadata : {group.metadata}")

    print("")
    for i in range(0, group.dataset_count):
        data = group.data_as_double(i)
        time = group.dataset_time(i)
        print(f"Dataset Shape for time {time} : {data.shape}")

    print("")

    meshio = mesh.meshio()
    print(meshio)
