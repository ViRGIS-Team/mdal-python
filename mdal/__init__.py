__version__='0.9.0'

#from .array import Array

from mdal.libmdalpython import getVersionString, \
                               getDriverCount, \
                               getLastStatus, \
                               Driver, \
                               getDrivers, \
                               Datasource, \
                               PyMesh, \
                               DatasetGroup


class Info(object):
    version = getVersionString()

info = Info()
