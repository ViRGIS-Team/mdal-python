import mdal
from mdal import Datasource

print(f"MDAL Version:  {mdal.getVersionString()}")
print(f"MDAL Driver Count :{mdal.getDriverCount()}")
print(mdal.getLastStatus())

for driver in mdal.getDrivers():
    print(driver.long_name)


ds = Datasource("data/ply/test_mesh.ply")
print (ds.meshes)

mesh = ds.load(0)
print (f"Driver : {mesh.driverName}")
print (f"Vertex Count : {mesh.vertexCount}")
print (f"Face Count : {mesh.faceCount}")
print ( f"Largest Face: {mesh.largestFace}")
print (f"Edge Count : {mesh.edgeCount}")
print ( f"CRS : {mesh.projection}")
print ( f"Mesh extent : {mesh.extent}")
print ( f"DatasetGroup Count : {mesh.groupCount}")
print ("")

mesh = ds.load(ds.meshes[0])
print (f"Driver : {mesh.driverName}")
print (f"Vertex Count : {mesh.vertexCount}")
print (f"Face Count : {mesh.faceCount}")
print ( f"Largest Face: {mesh.largestFace}")
print (f"Edge Count : {mesh.edgeCount}")
print ( f"CRS : {mesh.projection}")
print ( f"Mesh extent : {mesh.extent}")

vertex = mesh.getVerteces()
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

print("")

data = group.getDataset(0)
print(f"Dataset is valid : {data.isValid}")
print(f"Dataset value count : {data.valueCount}")
