from mdal.libmdalpython import version_string, \
    driver_count, \
    last_status, \
    Driver, \
    drivers, \
    Datasource, \
    PyMesh, \
    DatasetGroup, \
    MDAL_Status, \
    MDAL_DataLocation

from mdal.transform import MDAL_transform

__version__ = '1.0.1'


class Info(object):
    """Information on MDAL"""
    version = version_string()
    driver_count = driver_count()
    drivers = [driver.long_name for driver in drivers()]


info = Info()
