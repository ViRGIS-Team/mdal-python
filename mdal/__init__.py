from mdal.libmdalpython import version_string, \
    driver_count, \
    last_status, \
    Driver, \
    drivers, \
    Datasource, \
    PyMesh, \
    DatasetGroup
__version__ = '0.9.2'


class Info(object):
    """Information on MDAL"""
    version = version_string()
    driver_count = driver_count()
    drivers = [driver.long_name for driver in drivers()]


info = Info()
