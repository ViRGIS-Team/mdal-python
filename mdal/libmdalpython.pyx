# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cpython.version cimport PY_MAJOR_VERSION
cimport numpy as np
np.import_array()

from cpython cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "mdal.h":
    ctypedef void* MDAL_DriverH;
    ctypedef void* MDAL_DatasetGroupH;
    ctypedef void* MDAL_DatasetH;

    cpdef enum MDAL_Status:
        Err_NotEnoughMemory = 1,
        Err_FileNotFound = 2,
        Err_UnknownFormat = 3,
        Err_IncompatibleMesh = 4,
        Err_InvalidData = 5,
        Err_IncompatibleDataset = 6,
        Err_IncompatibleDatasetGroup = 7,
        Err_MissingDriver = 8,
        Err_MissingDriverCapability = 9,
        Err_FailToWriteToDisk = 10,
        Err_UnsupportedElement = 11,
        Warn_InvalidElements = 12,
        Warn_ElementWithInvalidNode = 13,
        Warn_ElementNotUnique = 14,
        Warn_NodeNotUnique = 15,
        Warn_MultipleMeshesInFile = 16

    cpdef enum MDAL_DataLocation:
        DataInvalidLocation = 0,
        DataOnVertices = 1,
        DataOnFaces = 2,
        DataOnVolumes = 3,
        DataOnEdges = 4

    cdef string MDAL_Version()
    cdef MDAL_Status MDAL_LastStatus()
    cdef int MDAL_driverCount()
    cdef MDAL_DriverH MDAL_driverFromIndex(int index)
    cdef MDAL_DriverH MDAL_driverFromName( string name )
    cdef bool MDAL_DR_meshLoadCapability( MDAL_DriverH driver )
    cdef bool MDAL_DR_writeDatasetsCapability( MDAL_DriverH driver, MDAL_DataLocation location )
    cdef string MDAL_DR_writeDatasetsSuffix( MDAL_DriverH driver )
    cdef bool MDAL_DR_saveMeshCapability( MDAL_DriverH driver )
    cdef string MDAL_DR_name( MDAL_DriverH driver )
    cdef string MDAL_DR_longName( MDAL_DriverH driver )
    cdef string MDAL_DR_filters( MDAL_DriverH driver )
    cdef string MDAL_MeshNames(  char* uri )
    cdef MDAL_DataLocation MDAL_G_dataLocation(MDAL_DatasetGroupH group)
    cdef string MDAL_G_name(MDAL_DatasetGroupH group)
    cdef int MDAL_G_datasetCount(MDAL_DatasetGroupH group)
    cdef MDAL_DatasetH MDAL_G_dataset(MDAL_DatasetGroupH group, int index)
    cdef bool MDAL_G_hasScalarData(MDAL_DatasetGroupH group)
    cdef int MDAL_G_maximumVerticalLevelCount(MDAL_DatasetGroupH group)
    cdef bool MDAL_G_isTemporal(MDAL_DatasetGroupH group)
    cdef const char *MDAL_G_referenceTime(MDAL_DatasetGroupH group)
    cdef void MDAL_G_minimumMaximum(MDAL_DatasetGroupH group, double* min, double* max)
    cdef double MDAL_D_time(MDAL_DatasetH dataset)


def getVersionString():
    return MDAL_Version()

def getLastStatus():
    return MDAL_LastStatus()

def getDriverCount():
    return MDAL_driverCount()

def getDrivers():
    ret = []
    for i in range(0, getDriverCount()):
        ret.append(Driver(i))
    return ret

cdef extern from "PyMesh.hpp" namespace "mdal::python":
    cdef cppclass Mesh:
        Mesh() except +
        Mesh(char* uri) except +
        void *getVerteces() except +
        void *getFaces() except +
        void *getEdges() except +
        int vertexCount() except +
        int edgeCount() except +
        int faceCount() except +
        int maxFaceVertex() except +
        string getProjection() except +
        void getExtent(double* minX, double* maxX, double* minY, double* maxY) except +
        string getDriverName() except +
        int groupCount() except +
        MDAL_DatasetGroupH getGroup(int index) except +


cdef class Driver:
    cdef MDAL_DriverH thisptr # hold the pointer to the driver instance we are wrapping

    def __cinit__(self, int index):
        self.thisptr = MDAL_driverFromIndex(index);

    property name:

        def __get__(self):
            return MDAL_DR_name(self.thisptr)

    property long_name:

        def __get__(self):
            return MDAL_DR_longName(self.thisptr)

    property filters:

        def __get__(self):
            return MDAL_DR_filters(self.thisptr)

    property saveMeshCapability:

        def __get__(self):
            return MDAL_DR_saveMeshCapability(self.thisptr)

    def getWriteDatasetCapability(self, location):
        return MDAL_DR_writeDatasetsCapability(self.thisptr, location)

    property writeDatasetsSuffix:

        def __get__(self):
            return MDAL_DR_writeDatasetsSuffix(self.thisptr)

    property meshLoadCapability:

        def __get__(self):
            return MDAL_DR_meshLoadCapability(self.thisptr)

cdef class Datasource:

    cdef string uri  # hold the uri reference for the datasource

    def __cinit__(self, string uri):
        self.uri = uri

    property meshes:

        def __get__(self):
         return self.getMeshNames().split(";;")

    def getMeshNames(self):
        ret = MDAL_MeshNames(bytes(self.uri, 'utf-8'))
        status = getLastStatus()
        if status != 0:
            raise IndexError("No Meshes Found" + str(status))
        return ret

    def load(self, arg):
        if type(arg) is int:
            arg = self.meshes[arg]
        return PyMesh.load(arg)
    

cdef class PyMesh:
    cdef Mesh* thisptr

    @classmethod
    def load(cls, uri):
        ret = PyMesh()
        ret.thisptr = new Mesh(bytes(uri, 'utf-8'))
        return ret

    property vertexCount:

        def __get__(self):
            return self.thisptr.vertexCount()

    property faceCount:

        def __get__(self):
            return self.thisptr.faceCount()

    property edgeCount:

        def __get__(self):
            return self.thisptr.edgeCount()

    property largestFace:

        def __get__(self):
            return self.thisptr.maxFaceVertex()

    property projection:

        def __get__(self):
            return self.thisptr.getProjection();

    property extent:

        def __get__(self):
            minX = <double>0
            minY = <double>0
            maxX = <double>0
            maxY = <double>0
            self.thisptr.getExtent(&minX, &maxX, &minY, &maxY)
            return (minX, maxX, minY, maxY)

    property driverName:
    
        def __get__(self):
            return self.thisptr.getDriverName()

    property groupCount:

        def __get__(self):
            return self.thisptr.groupCount()

    def getVerteces(self):
        return <object>self.thisptr.getVerteces()
    
    def getFaces(self):
        return <object>self.thisptr.getFaces()

    def getEdges(self):
        return <object>self.thisptr.getEdges()
    
    def getGroup(self, index):
        ret = DatasetGroup()
        ret.thisptr = <MDAL_DatasetGroupH>self.thisptr.getGroup(index)
        ret.thisdata = new Data(ret.thisptr)
        return ret

    def getGroups(self):
        ret = []
        for i in range(0,self.groupCount):
            ret.append(self.getGroup(i))
        return ret;


cdef extern from "DatasetGroup.hpp" namespace "mdal::python":
    cdef cppclass Data:
        Data() except +
        Data(MDAL_DatasetGroupH data) except +
        dict getMetadata() except +
        void* getDataAsDouble(int index) except +

cdef class DatasetGroup:
    cdef MDAL_DatasetGroupH thisptr
    cdef Data* thisdata # cpp class instance used to marshall the data values

    property location:

        def __get__(self):
            return MDAL_G_dataLocation(self.thisptr)

    property name:

        def __get__(self):
            return MDAL_G_name(self.thisptr)

    property datasetCount:

        def __get__(self):
            return MDAL_G_datasetCount(self.thisptr)

    property hasScalar:

        def __get__(self):
            return MDAL_G_hasScalarData(self.thisptr)

    property isTemporal:

        def __get__(self):
            return MDAL_G_isTemporal(self.thisptr)

    property referenceTime:

        def __get__(self):
            return MDAL_G_referenceTime(self.thisptr)

    property levelCount:

        def __get__(self):
            return MDAL_G_maximumVerticalLevelCount(self.thisptr)

    property minmax:

        def __get__(self):
            min = <double>0
            max = <double>0
            MDAL_G_minimumMaximum(self.thisptr, &min, &max)
            return (min, max)

    def getMetadata(self):
        return self.thisdata.getMetadata()

    def getDataAsDouble(self, index=0):
        return <object>self.thisdata.getDataAsDouble(index)

    def getDatasetTime(self, index):
        return MDAL_D_time(MDAL_G_dataset(self.thisptr, index))


