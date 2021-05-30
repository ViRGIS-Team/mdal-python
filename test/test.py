from mdal import Datasource, Info, getLastStatus

print(f"MDAL Version:  {Info.version}")
print(f"MDAL Driver Count :{Info.driverCount}")
print(getLastStatus())

for driver in Info.drivers:
    print(driver)


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
for i in range(0, group.datasetCount):
    data = group.getDataAsDouble(i)
    time = group.getDatasetTime(i)
    print(f"Dataset Shape for time {time} : {data.shape}")

print("")

meshio = mesh.getMeshio()
print(meshio)
