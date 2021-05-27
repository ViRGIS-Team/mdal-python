__version__='0.9.0'

from mdal.libmdalpython import getVersionString, \
                               getDriverCount, \
                               getLastStatus, \
                               Driver, \
                               getDrivers, \
                               Datasource, \
                               PyMesh, \
                               DatasetGroup


class Info(object):
    """Information on MDAL"""
    version = getVersionString()
    driverCount = getDriverCount()
    drivers = [driver.long_name for driver in getDrivers()]

info = Info()