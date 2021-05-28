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

data = group.getDataAsDouble()
print(f"Dataset Shape : {data.shape}")
