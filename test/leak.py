from mdal import Datasource
from pathlib import Path
import gc

mypath = "C:\\Users\\runes\\Documents\\GitHub\\mdal-python\\test\\results_3di.nc"
ds = Datasource(mypath)

for i in range(0,100):
    print(i)
    with ds.load() as mesh:
        print("mesh loaded")
        group = mesh.group(3)
        print("group Loaded")
        for j in range(0, group.dataset_count):
            data = group.data(j)
            print(data)
            # make very sure the variable gets taken out of memory:
            del(data)