from mdal import Datasource
from pathlib import Path
import gc

mypath = "C:\\Users\\runes\\Documents\\GitHub\\mdal-python\\test\\results_3di.nc"
ds = Datasource(mypath)

with ds.load() as mesh:
    print("mesh loaded")
    group = mesh.group(3)
    for i in range(1000):
        for j in range(0, group.dataset_count):
            print(j)
            data = group.data(j)
            # make very sure the variable gets taken out of memory:
            del(data)
            gc.collect()