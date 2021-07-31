from mdal import Datasource, MDAL_DataLocation

ds = Datasource("data/tuflowfv/withMaxes/trap_steady_05_3D.nc")
mesh = ds.load()
group = mesh.groups[1]
a, b, c = group.volumetric(0)

ds2 = Datasource("test_vol.ply")
mesh2 = ds2.add_mesh()
mesh2.vertices = mesh.vertices
mesh2.faces = mesh.faces

print(f"Vertex Count :{mesh.vertex_count}")
print(f"Face Count : {mesh.face_count}")

group2 = mesh2.add_group("test", location=MDAL_DataLocation.DataOnVolumes)
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
mesh3 = ds2.load()
mesh3.info()
group3 = mesh3.groups[1]
print(f"{group3.location}")
d, e, f = group3.volumetric(0)
print(f"{group3.volumetric(0)}")
print(f"{group3.data(0)}")
print("Mesh Equality : {mesh2 == mesh3}")
