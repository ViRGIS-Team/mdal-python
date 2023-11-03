from mdal import Datasource
from pathlib import Path
import gc

mypath = "/Users/paulharwood/Library/CloudStorage/GoogleDrive-runette@gmail.com/My Drive/Mesh Project/gpr_new.ply"
ds = Datasource(mypath)

with ds.load() as mesh:
    print("mesh loaded")
    for i in range(0,1000):
        print(i)
        group = mesh.group(3)
        data = group.data(0)
        print(data)