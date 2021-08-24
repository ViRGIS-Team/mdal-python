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
import typing
from typing import Union
from pathlib import Path

from cpython cimport PyObject, Py_INCREF
from cython.operator cimport dereference as deref, preincrement as inc

cdef extern from "mdal.h":
    ctypedef void* MDAL_DriverH;
    ctypedef void* MDAL_DatasetGroupH;
    ctypedef void* MDAL_DatasetH;
    ctypedef void* MDAL_MeshH;

    cpdef enum MDAL_Status:
        No_Err "None" = 0,
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
    cdef void MDAL_ResetStatus()
    cdef int MDAL_driverCount()
    cdef MDAL_DriverH MDAL_driverFromIndex(int index)
    cdef MDAL_DriverH MDAL_driverFromName( const char* name )
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
    cdef void MDAL_G_closeEditMode(MDAL_DatasetGroupH group)
    cdef double MDAL_D_time(MDAL_DatasetH dataset)
    cdef bool MDAL_G_isInEditMode(MDAL_DatasetGroupH group)
    cdef string MDAL_G_driverName( MDAL_DatasetGroupH group )
    cdef void MDAL_G_setReferenceTime( MDAL_DatasetGroupH group, const char* referenceTimeISO8601 )
    cdef string MDAL_G_uri( MDAL_DatasetGroupH group )
    cdef MDAL_MeshH MDAL_G_mesh( MDAL_DatasetGroupH group )
    cdef int MDAL_M_vertexCount( MDAL_MeshH mesh )
    cdef int MDAL_M_faceCount( MDAL_MeshH mesh )    
    cdef int MDAL_M_edgeCount( MDAL_MeshH mesh )

def version_string():
    """Returns MDAL version"""
    return MDAL_Version()

def last_status():
    """Returns last status message"""
    return MDAL_Status(MDAL_LastStatus())

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
        MDAL_Status setProjection(const char* proj) except +
        void getExtent(double* minX, double* maxX, double* minY, double* maxY) except +
        string getDriverName() except +
        int groupCount() except +
        MDAL_DatasetGroupH getGroup(int index) except +
        bool addMesh( MDAL_MeshH mesh ) except +
        bool addVertices(np.ndarray vertices) except +
        bool addFaces(np.ndarray faces, long count) except +
        bool addEdges(np.ndarray faces) except +
        bool save(char* uri) except +
        bool save(char* uri, char* drv) except +
        MDAL_DatasetGroupH addGroup(const char* name, MDAL_DataLocation loc, bool hasScalar, const char* file) except +
        MDAL_DatasetGroupH addGroup(const char* name, MDAL_DataLocation loc, bool hasScalar, const char* uri, MDAL_DriverH drv) except +
        dict getMetadata() except +
        MDAL_Status setMetadata(dict md, const char* encoding) except +


cdef class Driver:
    """Wrapper for the MDAL Driver.

    Init : 

    Driver(index: int)
    Driver(name : str)
    """
    cdef MDAL_DriverH thisptr # hold the pointer to the driver instance we are wrapping

    def __cinit__(self, *args):
        if type(args[0]) == str:
            self.thisptr = MDAL_driverFromName(args[0])
        else:
            self.thisptr = MDAL_driverFromIndex(args[0])

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
    cdef str uri  # hold the uri for the datasource
    cdef str driver_name  # hold the driver name for the file

    def __cinit__(self, uri: Path, driver_name : str = None):   
        self.uri = str(uri)
        if type(uri) != Path:
            path = Path(uri)
        if driver_name:
            self.driver_name = driver_name
            return
        if path.is_file():
            try:
                meshes = self.meshes
                if len(meshes) > 0:
                    self.driver_name = meshes[0].split(":")[0]
                    return
            except Exception:
                pass
        ex = path.suffix
        for driver in drivers():
            if ex in driver.filters:
                self.driver_name = driver.name
                return
        raise ValueError("Cannot Create Mesh")
        
    def __eq__(self,other):
        return self.uri == other.uri

    property meshes:
        """Returns a list of mesh uris"""
        def __get__(self):
            try:
                return self.mesh_name_string().split(";;")
            except:
                return []

    def mesh_name_string(self):
        ret = MDAL_MeshNames(bytes(self.uri, 'utf-8'))
        return ret

    def load(self, arg: Union(int, str) = 0) -> PyMesh:
        """Returns an mdal.PyMesh object wrapping an instance of an MDAL mesh.

        Usage: 
        
        ds.load(uri: str) -> PyMesh
        ds.load(index: int) -> PyMesh
        """
        if type(arg) is int:
            arg = self.meshes[arg]
        return PyMesh(arg)
    
    def add_mesh(self, name: str = None):
        """Adds a mesh to the datasource. The mesh must be persisted by mesh.save()
        
        Usage:
        
        ds.add_mesh("test") -> PyMesh
        """
        uri = f'{self.driver_name}:"{self.uri}"'
        if name:
            uri += f':{name}'
        mesh = PyMesh(self.driver_name)
        mesh.uri = uri
        return mesh
    
cdef class PyMesh:
    """Class that wraps an MDAL Mesh instance.
    
    
    
    """
    cdef Mesh* thisptr
    cdef bool valid
    cdef str file
    cdef str name

    def __cinit__(self, *args):
        MDAL_ResetStatus()
        if args:
            try:
                if type(args[0]) != str:
                    driver: Driver = args[0]
                    ptr: MDAL_DriverH = driver.thisptr
                    self.thisptr = new Mesh(ptr)
                elif args[0] in [driver.name for driver in drivers()]:
                    driver: Driver = Driver(args[0])
                    ptr: MDAL_DriverH = driver.thisptr
                    self.thisptr = new Mesh(ptr)
                else:
                    self.uri = args[0]
                    self.thisptr = new Mesh(bytes(args[0], 'utf-8'))
            except:
                raise ValueError("Mesh Creation Error")
            if last_status() != MDAL_Status.No_Err:
                raise ValueError(last_status().name)
        else:
            self.thisptr = new Mesh(Driver("PLY").thisptr)
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
        
    def info(self):
        print(f"Driver : {self.driver_name}")
        print(f"File : {self.file}")
        print(f"Mesh Name : {self.name}")
        print(f"Is valid ? : {self.valid}")
        print(f"Metadata : {self.metadata}")
        print(f"CRS : {self.projection}")
        print(f"Groups : {[item.name for item in self.groups]}")
    
    property uri:
    
        def __get__(self):
            uri = f'{self.driver_name}:"{self.file}"'
            if self.name:
                uri += f":{self.name}"
            return uri
            
        def __set__(self, uri):
            if ":" in uri:
                spl = uri.split(":")
                self.file = spl[1].strip('"')
                try:
                    self.name = spl[2]
                except Exception:
                    pass

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
                return self.thisptr.getProjection()

        def __set__(self, proj:str):
            if self.valid:
                status = self.thisptr.setProjection(proj)
                if status != 0:
                    raise ValueError(MDAL_Status(status).name)
                return

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
            ret.mesh = self
            return ret

    property groups:

        def __get__(self):
            if self.valid:
                ret = []
                for i in range(0,self.group_count):
                    ret.append(self.group(i))
                return ret
                
    property metadata:

        def __get__(self):
            return self.thisptr.getMetadata()
        
        def __set__(self, metadata):
            if self.thisptr.setMetadata(metadata, "utf-8") != MDAL_Status.No_Err:
                raise ValueError(last_status().name)
                
    def get_metadata(self, key: str, encoding: str = 'UTF-8'):
        value = self.metadata.get(key.encode(encoding=encoding))
        if value:
            value = value.decode(encoding=encoding)
        return value

    def add_metadata(self, key: str, value: str, encoding: str = 'utf-8' ):
        if self.valid:
            if self.thisptr.setMetadata({key: value}, encoding) != MDAL_Status.No_Err:
                raise ValueError(last_status().name)
    
    def save(self, uri: str = None, drv: Driver = None):
        MDAL_ResetStatus()
        if drv:
            if not self.thisptr.save(bytes(uri, 'utf-8'), bytes(drv.name, 'utf-8')):
                raise ValueError(last_status().name)
        else:
            if uri:
                self.uri = uri
            else:
                uri = self.uri
            if ":" in uri:
                spl = uri.split(":")
                driver: str = spl[0]
                file: str = spl[1].strip('"')
                if not self.thisptr.save(bytes(file, 'utf-8'),bytes(driver, 'utf-8')):
                    raise ValueError(last_status().name)
            else:
                if not self.thisptr.save(bytes(uri, 'utf-8')):
                    raise ValueError(last_status().name)
                    
        self.valid = True

    def add_group(self,  arg, **kwargs):
        MDAL_ResetStatus()
        if type(arg) == DatasetGroup:
            ret = DatasetGroup()
            spl = self.uri.split(":")
            file: str = spl[1].strip('"')
            ret.thisptr = <MDAL_DatasetGroupH>self.thisptr.addGroup(arg.name, arg.location, arg.has_scalar, arg.uri)
            ret.thisdata = new Data(ret.thisptr)
            ret.mesh = self
            for i in range(0, arg.dataset_count ):
                ret.add_data(arg.data(i), arg.dataset_time(i))
        else:
            if "location" in kwargs:
                location: MDAL_DataLocation = kwargs.get("location")
            else:
                location = MDAL_DataLocation.DataOnVertices
            if "uri" in kwargs and ":" in kwargs.get("uri"):
                spl = kwargs.get("uri").split(":")
                driver: str = spl[0]
                file: str = spl[1].strip('"')
            elif "file" in kwargs:
                file: str = kwargs.get("file")
                if "driver" in kwargs:
                    driver: str = kwargs.get("driver")
                else:
                    driver: str = self.driver_name
            elif self.valid:
                spl = self.uri.split(":")
                driver: str = spl[0]
                file: str = spl[1].strip('"')
            else:
                raise ValueError("No valid place to save data")
            if "vector" in kwargs:
                scalar: bool = not kwargs.get("vector")
            else:
                scalar: bool = True
            ret = DatasetGroup()
            ret.thisptr = <MDAL_DatasetGroupH>self.thisptr.addGroup(arg, location, scalar, file, MDAL_driverFromName(bytes(driver, 'utf-8')))
            ret.thisdata = new Data(ret.thisptr)
            if last_status() != MDAL_Status.No_Err:
                raise ValueError(f"Group creation error : {last_status().name}")
        return ret
        
    def deep_copy(self, other: PyMesh):
        self.vertices = other.vertices
        self.faces = other.faces
        self.edges = other.edges
        self.projection = other.projection
        
    def data_copy(self, other: PyMesh):
        for group in other.groups:
            self.add_group(group)

cdef extern from "DatasetGroup.hpp" namespace "mdal::python":
    cdef cppclass Data:
        Data() except +
        Data(MDAL_DatasetGroupH data) except +
        dict getMetadata() except +
        void* getDataAsDouble(int index) except +
        void* getDataAsVolumeIndex(int index) except +
        void* getDataAsLevelCount(int index) except +
        void* getDataAsLevelValue(int index) except +
        MDAL_Status setDataAsDouble(np.ndarray data, double time) except +
        MDAL_Status setDataAsVolume(np.ndarray data, np.ndarray verticalLevelCounts, np.ndarray verticalLevels, double time ) except +
        MDAL_Status setMetadata(dict md, const char* encoding ) except +

cdef class DatasetGroup:
    cdef MDAL_DatasetGroupH thisptr
    cdef Data* thisdata # cpp class instance used to marshall the data values
    cdef PyMesh _mesh
 
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
            
    property driver_name:
    
        def __get__(self):
            return MDAL_G_driverName(self.thisptr)

    property has_scalar:

        def __get__(self):
            return MDAL_G_hasScalarData(self.thisptr)

    property is_temporal:

        def __get__(self):
            return MDAL_G_isTemporal(self.thisptr)

    property reference_time:

        def __get__(self):
            return MDAL_G_referenceTime(self.thisptr)
            
        def __set__(self, referenceTimeISO8601: str):
            MDAL_ResetStatus()
            MDAL_G_setReferenceTime( self.thisptr, referenceTimeISO8601 )
            if last_status() != 0:
                raise ValueError(last_status().name)
            return

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
            
        def __set__(self, metadata):
            if self.thisdata.setMetadata(metadata, 'utf-8') != MDAL_Status.No_Err:
                raise ValueError(last_status().name)
            
    property vertex_count:

        def __get__(self):
            return MDAL_M_vertexCount( <MDAL_MeshH>MDAL_G_mesh( self.thisptr ) )

    property face_count:

        def __get__(self):
            return MDAL_M_faceCount( <MDAL_MeshH>MDAL_G_mesh( self.thisptr ) )

    property edge_count:

        def __get__(self):
           return MDAL_M_edgeCount( <MDAL_MeshH>MDAL_G_mesh( self.thisptr ) )
           
    property mesh:
    
        def __get__(self):
            return self._mesh
            
        def __set__(self, mesh):
            self._mesh = mesh
                
    def get_metadata(self, key: str, encoding: str = 'UTF-8'):
        value = self.metadata.get(key.encode(encoding=encoding))
        if value:
            value = value.decode(encoding=encoding)
        return value
        
    def add_metadata(self, key: str, value: str, encoding: str = 'utf-8' ):
        if self.thisdata.setMetadata({key: value}, encoding) != MDAL_Status.No_Err:
            raise ValueError(last_status().name)
            
    property edit_mode:
    
        def __get__(self):
            return MDAL_G_isInEditMode(self.thisptr)
            
    property uri:
    
        def __get__(self):
            return MDAL_G_uri( self.thisptr )
    
    def save(self):
        MDAL_G_closeEditMode(self.thisptr)
        if last_status() != 0:
            raise ValueError(last_status().name)
        
    def add_data(self, data: npy.ndarray, time:float = 0):
        if type(time) == str:
            if time == "":
                time = 0
            else:
                time = float( time )
        if ( self.location == MDAL_DataLocation.DataInvalidLocation or
            (self.location == MDAL_DataLocation.DataOnVertices and len(data) != self.vertex_count ) or
            (self.location == MDAL_DataLocation.DataOnFaces and len(data) != self.face_count ) or
            (self.location == MDAL_DataLocation.DataOnEdges and len(data) != self.edge_count )):
                raise ValueError("Invalid Dataset")
        self.thisdata.setDataAsDouble( data, time)
        if last_status() != 0:
            raise ValueError(last_status().name)

    def data(self, index=0):
        ret =  <object>self.thisdata.getDataAsDouble(index)
        if last_status() != 0:
            raise ValueError(last_status().name)
        return ret

    def dataset_time(self, index):
        return MDAL_D_time(MDAL_G_dataset(self.thisptr, index))
        
    def volumetric(self, index: int = 0):
        if (self.location != MDAL_DataLocation.DataOnVolumes):
            raise ValueError("Group has no data on volumes")
        face_ind = <object>self.thisdata.getDataAsVolumeIndex(index)
        level_counts = <object>self.thisdata.getDataAsLevelCount(index)
        level_values = <object>self.thisdata.getDataAsLevelValue(index)
        if last_status() != MDAL_Status.No_Err:
            raise ValueError(last_status().name)
        return (level_counts, level_values, face_ind)

    def add_volumetric(self, data: npy.ndarray, level_counts: npy.ndarray, level_values: npy.ndarray, time:float = 0):
        self.thisdata.setDataAsVolume( data, level_counts, level_values, time)
        if last_status() != 0:
            raise ValueError(last_status().name)
