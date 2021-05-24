import mdal
from mdal import Datasource

print(mdal.getVersionString())
print(mdal.getDriverCount())
print(mdal.getLastStatus())

for driver in mdal.getDrivers():
    print(driver.long_name)


ds = Datasource("data/ply/test_mesh.ply")
print (ds.meshes)

mesh = ds.load(0)
print (f"Driver : {mesh.driverName}")
print (f"Vertex Count : {mesh.vertexCount}")
print (f"Face Count : {mesh.faceCount}")
print (f"Edge Count : {mesh.edgeCount}")
print ( f"CRS : {mesh.projection}")
print ( f"Mesh extent : {mesh.extent}")
print ("")

mesh = ds.load(ds.meshes[0])
print (f"Driver : {mesh.driverName}")
print (f"Vertex Count : {mesh.vertexCount}")
print (f"Face Count : {mesh.faceCount}")
print (f"Edge Count : {mesh.edgeCount}")
print ( f"CRS : {mesh.projection}")
print ( f"Mesh extent : {mesh.extent}")

mesh.getVerteces()
