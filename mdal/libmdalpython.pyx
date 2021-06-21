# distutils: language = c++
# cython: c_string_type=unicode, c_string_encoding=utf8, embedsignature=True

"""
/******************************************************************************
* Copyright (c) 2021, Runette Software Ltd (www.runette.co.uk)
*
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following
* conditions are met:
*
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in
*       the documentation and/or other materials provided
*       with the distribution.
*     * Neither the name of Runette Software nor the
*       names of its contributors may be used to endorse or promote
*       products derived from this software without specific prior
*       written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
* "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
* LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
* FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
* COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
* INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
* BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
* OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
* AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
* OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
* OF SUCH DAMAGE.
****************************************************************************/
"""

from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint32_t, int64_t
from libcpp cimport bool
from cpython.version cimport PY_MAJOR_VERSION
cimport numpy as np
import numpy as npy
np.import_array()
import meshio
import typing
from typing import Union

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


def version_string():
    """Returns MDAL version"""
    return MDAL_Version()

def last_status():
    """Returns last status message"""
    ret = MDAL_LastStatus()
    if ret != 0:
        return MDAL_Status(ret)
    else:
        return None

def driver_count():
    """Returns count of registed MDAL drivers"""
    return MDAL_driverCount()

def drivers():
    """Returns the list of Drivers"""
    ret = []
    for i in range(0, driver_count()):
        ret.append(Driver(i))
    return ret

cdef extern from "PyMesh.hpp" namespace "mdal::python":
    cdef cppclass Mesh:
        Mesh() except +
        Mesh(char* uri) except +
        Mesh(MDAL_DriverH drv) except +
        void *getVertices() except +
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
        bool addVertices(np.ndarray vertices) except +
        bool addFaces(np.ndarray faces, long count) except +
        bool addEdges(np.ndarray faces) except +
        bool save(char* uri) except +
        bool save(char* uri, char* drv) except +


cdef class Driver:
    """Wrapper for the MDAL Driver.

    Init : Driver(index: int)
    """
    cdef MDAL_DriverH thisptr # hold the pointer to the driver instance we are wrapping

    def __cinit__(self, int index):
        self.thisptr = MDAL_driverFromIndex(index);

    property name:
        """Driver Short Name"""

        def __get__(self):
            return MDAL_DR_name(self.thisptr)

    property long_name:
        """Driver Long Name"""
        def __get__(self):
            return MDAL_DR_longName(self.thisptr)

    property filters:

        def __get__(self):
            return MDAL_DR_filters(self.thisptr)

    property save_mesh_capability:

        def __get__(self):
            return MDAL_DR_saveMeshCapability(self.thisptr)
            
    
    def write_dataset_capability(self, location):
        return MDAL_DR_writeDatasetsCapability(self.thisptr, location)

    property write_datasets_suffix:

        def __get__(self):
            return MDAL_DR_writeDatasetsSuffix(self.thisptr)

    property mesh_load_capability:

        def __get__(self):
            return MDAL_DR_meshLoadCapability(self.thisptr)

cdef class Datasource:
    """Wrapper for a Source of MDAL data - e.g. a file.
    
    Init: Datasource(uri: str | PosixPath)
    """
    cdef str uri  # hold the uri reference for the datasource

    def __cinit__(self, uri):   
        self.uri = str(uri)
        
    def __eq__(self,other):
        return self.uri == other.uri

    property meshes:
        """Returns a list of mesh uris"""
        def __get__(self):
         return self.mesh_name_string().split(";;")

    def mesh_name_string(self):
        ret = MDAL_MeshNames(bytes(self.uri, 'utf-8'))
        if last_status():
            raise ValueError(last_status().name)
        if len(ret) == 0:
            raise IndexError("No Meshes Found")
        return ret

    def load(self, arg: Union(int, str)) -> PyMesh:
        """Returns an mdal.PyMesh object wrapping an instance of an MDAL mesh.

        Usage: 
        
        ds.load(uri: str) -> PyMesh
        ds.load(index: int) -> PyMesh
        """
        if type(arg) is int:
            arg = self.meshes[arg]
        return PyMesh(arg)
    

cdef class PyMesh:
    cdef Mesh* thisptr
    cdef bool valid

    def __cinit__(self, *args):
        if args:
            try:
                self.thisptr = new Mesh(bytes(args[0], 'utf-8'))
            except:
                driver: Driver = args[0]
                ptr: MDAL_DriverH = driver.thisptr
                self.thisptr = new Mesh(ptr)
        else:
            self.thisptr = new Mesh()
        self.valid = True

    def __dealloc__(self):
        if self.valid:
            del self.thisptr

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        del self.thisptr
        self.valid = False
        return False
        
    def __eq__(self, other):
        return npy.array_equal(self.vertices,other.vertices) and npy.array_equal(self.faces, other.faces) and npy.array_equal(self.edges,other.edges)

    property vertex_count:

        def __get__(self):
            if self.valid:
                return self.thisptr.vertexCount()

    property face_count:

        def __get__(self):
            if self.valid:
                return self.thisptr.faceCount()

    property edge_count:

        def __get__(self):
            if self.valid:
                return self.thisptr.edgeCount()

    property largest_face:

        def __get__(self):
            if self.valid:
                return self.thisptr.maxFaceVertex()

    property projection:

        def __get__(self):
            if self.valid:
                return self.thisptr.getProjection();

    property extent:

        def __get__(self):
            if self.valid:
                minX = <double>0
                minY = <double>0
                maxX = <double>0
                maxY = <double>0
                self.thisptr.getExtent(&minX, &maxX, &minY, &maxY)
                return (minX, maxX, minY, maxY)

    property driver_name:
    
        def __get__(self):
            if self.valid:
                return self.thisptr.getDriverName()

    property group_count:

        def __get__(self):
            if self.valid:
                return self.thisptr.groupCount()

    property vertices:
    
        def __get__(self):
            if self.valid:
                return <object>self.thisptr.getVertices()
                
        def __set__(self, np.ndarray vertices):
            if self.valid:
                if not self.thisptr.addVertices(vertices):
                    raise ValueError(last_status().name)
    
    property faces:
    
        def __get__(self):
            if self.valid:
                return <object>self.thisptr.getFaces()
                
        def __set__(self, np.ndarray faces):
            if self.valid:
                b = npy.array([list(faces[i]) for i in range(faces.shape[0])])
                if not self.thisptr.addFaces(faces, npy.sum(b, axis=0)[0]):
                    raise ValueError(last_status().name)

    property edges:
    
        def __get__(self):
            if self.valid:
                return <object>self.thisptr.getEdges()
            
        def __set__(self, np.ndarray edges):
            if self.valid:
                if not self.thisptr.addEdges(edges):
                    raise ValueError(last_status().name)
    
    def group(self, index):
        if self.valid:
            if type(index) is str:
                try:
                    return [group for group in self.getGroups() if group.name == index][0]
                except Exception:
                    return None
            ret = DatasetGroup()
            ret.thisptr = <MDAL_DatasetGroupH>self.thisptr.getGroup(index)
            ret.thisdata = new Data(ret.thisptr)
            return ret

    property groups:

        def __get__(self):
            if self.valid:
                ret = []
                for i in range(0,self.group_count):
                    ret.append(self.group(i))
                return ret
        
    def meshio(self):
        if self.valid:
            vertices = self.vertices
            if self.face_count == 0:
                if self.edge_count == 0:
                    return None
                edges = self.edges
                cells =[
                    ("line", npy.stack((edges['START'],edges['END']),1))
                ]
            else:
                faces = self.faces
                lines = faces[faces['Vertices'] == 2]
                tris = faces[faces['Vertices'] == 3]
                quads = faces[faces['Vertices'] == 4]
                cells = []
                if len(lines) > 0:
                    cells.append(("line",npy.stack((faces['V0'], faces['V1']), 1)))
                if len(tris) > 0:
                    cells.append(("triangle",npy.stack((faces['V0'], faces['V1'], faces['V2']), 1)))
                if len(quads) > 0:
                    cells.append(("quad",npy.stack((faces['V0'], faces['V1'], faces['V2'], faces['V3']), 1)))
                    
            point_data = {}
            cell_data = {}
            for group in self.groups:
                if group.location == 1 and group.has_scalar:
                    point_data.update({group.name: group.data_as_double(0)['U']})
                elif group.location == 2 and group.has_scalar:
                    cell_data.update({group.name: group.data_as_double(0)['U']})
                elif group.location == 4 and group.has_scalar:
                    cell_data.update({group.name: group.aata_as_double(0)['U']})
            return meshio.Mesh(
                npy.stack((vertices['X'], vertices['Y'], vertices['Z']), 1),
                cells,
                point_data,
                cell_data
            )
    
    def save(self, uri: str, drv: Driver = None):
        if drv:
            if not self.thisptr.save(bytes(uri, 'utf-8'), bytes(drv.name, 'utf-8')):
                raise ValueError(last_status().name)
        else:
            if ":" in uri:
                spl = uri.split(":")
                if not self.thisptr.save(spl[1],spl[0]):
                    raise ValueError(last_status().name)
            else:
                if not self.thisptr.save(bytes(uri, 'utf-8')):
                    raise ValueError(last_status().name)
                    
        

cdef extern from "DatasetGroup.hpp" namespace "mdal::python":
    cdef cppclass Data:
        Data() except +
        Data(MDAL_DatasetGroupH data) except +
        dict getMetadata() except +
        void* getDataAsDouble(int index) except +

cdef class DatasetGroup:
    cdef MDAL_DatasetGroupH thisptr
    cdef Data* thisdata # cpp class instance used to marshall the data values

    def __dealloc__(self):
        del self.thisdata

    property location:

        def __get__(self):
            return MDAL_DataLocation(MDAL_G_dataLocation(self.thisptr))

    property name:

        def __get__(self):
            return MDAL_G_name(self.thisptr)

    property dataset_count:

        def __get__(self):
            return MDAL_G_datasetCount(self.thisptr)

    property has_scalar:

        def __get__(self):
            return MDAL_G_hasScalarData(self.thisptr)

    property is_temporal:

        def __get__(self):
            return MDAL_G_isTemporal(self.thisptr)

    property reference_time:

        def __get__(self):
            return MDAL_G_referenceTime(self.thisptr)

    property level_count:

        def __get__(self):
            return MDAL_G_maximumVerticalLevelCount(self.thisptr)

    property minmax:

        def __get__(self):
            min = <double>0
            max = <double>0
            MDAL_G_minimumMaximum(self.thisptr, &min, &max)
            return (min, max)

    property metadata:

        def __get__(self):
            return self.thisdata.getMetadata()

    def data_as_double(self, index=0):
        return <object>self.thisdata.getDataAsDouble(index)

    def dataset_time(self, index):
        return MDAL_D_time(MDAL_G_dataset(self.thisptr, index))


