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

#include "PyMesh.hpp"

#include <numpy/ndarrayobject.h>
#include <string>
#include <cmath>
#include <cstring>
#include <iostream>


namespace mdal
{
namespace python
{

PyObject* defaultObject()
{
    const npy_intp dims = 1;
    return PyArray_SimpleNew(1, &dims, 1);
}
    
std::string toString(PyObject *pname)
{
    PyObject* r = PyObject_Str(pname);
    if (!r) {}
        //throw pdal_error("couldn't make string representation value");
    Py_ssize_t size;
    return std::string(PyUnicode_AsUTF8AndSize(r, &size));
}

// Create new empty mesh
//

Mesh::Mesh()
{
    hasMesh = false;
    if (_import_array() < 0)
        {}
        //throw pdal_error("Could not import numpy.core.multiarray.");
}

Mesh::Mesh(MDAL_DriverH drv) : m_vertices(nullptr), m_faces(nullptr), m_edges(nullptr), m_mdalMesh(nullptr)
{
    hasMesh = false;
    if (_import_array() < 0)
        {}
        //throw pdal_error("Could not import numpy.core.multiarray.");
    m_mdalMesh = MDAL_CreateMesh(drv);
}

Mesh::~Mesh()
{
    if (m_vertices)
        Py_XDECREF((PyObject *)m_vertices);
    if (m_faces)
        Py_XDECREF((PyObject *)m_faces);
    if (m_edges)
        Py_XDECREF((PyObject *)m_edges);
    if (m_mdalMesh)
        MDAL_CloseMesh(m_mdalMesh);
}

//
// Load from uri
//
Mesh::Mesh(const char* uri) : m_vertices(nullptr), m_faces(nullptr), m_edges(nullptr), m_mdalMesh(nullptr)
{
    if (_import_array() < 0)
        return;
    m_mdalMesh = MDAL_LoadMesh(uri);
    if (MDAL_LastStatus() == MDAL_Status::None){
        hasMesh=true;
    }
}

//
// add existing Mesh
//

bool Mesh::addMesh( MDAL_MeshH mesh )
{
    m_mdalMesh = std::move(mesh);
    return false;
}

//
// Save the mesh//
bool Mesh::save(const char* uri)
{
    return save(uri, MDAL_M_driverName(m_mdalMesh));
}

bool Mesh::save(const char* uri, const char* drv)
{
    MDAL_ResetStatus();
    MDAL_SetStatus(MDAL_LogLevel::Debug, MDAL_Status::None, uri);
    //if (! m_vertices) 
    //{
        //MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_InvalidData, "Mesh must have vertices");
        //return false;
    //}
    MDAL_SaveMesh(m_mdalMesh, uri, drv);
    if (MDAL_LastStatus() != MDAL_Status::None)
    {
        return false;
    }
    return true;
}

PyArrayObject *Mesh::getVertices() 
{
    if (! m_vertices) {
        PyObject* dict = PyDict_New();
        PyObject* formats = PyList_New(3);
        PyObject* titles = PyList_New(3);

        PyList_SetItem(titles, 0, PyUnicode_FromString("X"));
        PyList_SetItem(formats, 0, PyUnicode_FromString("f8"));
        PyList_SetItem(titles, 1, PyUnicode_FromString("Y"));
        PyList_SetItem(formats, 1, PyUnicode_FromString("f8"));
        PyList_SetItem(titles, 2, PyUnicode_FromString("Z"));
        PyList_SetItem(formats, 2, PyUnicode_FromString("f8"));

        PyDict_SetItemString(dict, "names", titles);
        PyDict_SetItemString(dict, "formats", formats);

        PyArray_Descr *dtype = nullptr;
        if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
            {}
    //         throw pdal_error("Unable to build numpy dtype");
        Py_XDECREF(dict);

        npy_intp size = (npy_intp)vertexCount();

        // This is a 1 x size array.
        m_vertices = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

        MDAL_MeshVertexIteratorH vi = MDAL_M_vertexIterator(m_mdalMesh);

        double* buffer = new double[3072];
        size_t count = 0;
        size_t t_count = 0;

        while(t_count < vertexCount())
        {
            count = MDAL_VI_next(vi, 1024, buffer);
            int idx = 0;
            for (int i = 0; i < count; i++) 
            {
                char* p = (char *)PyArray_GETPTR1(m_vertices, t_count + i);
                
                double x = buffer[idx];
                idx++;
                double y = buffer[idx];
                idx++;
                double z = buffer[idx];
                idx++;
                std::memcpy(p, &x, 8);
                std::memcpy(p + 8, &y, 8);
                std::memcpy(p + 16, &z,  8);
            }
            t_count += count;
        }
        delete [] buffer;
        MDAL_VI_close(vi);
    }
    return m_vertices;
}

PyArrayObject *Mesh::getFaces() 
{
    if (! m_faces) {
        PyObject* dict = PyDict_New();
        PyObject* formats = PyList_New(maxFaceVertex() + 1);
        PyObject* titles = PyList_New(maxFaceVertex() + 1);

        PyList_SetItem(titles, 0, PyUnicode_FromString("Vertices"));
        PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));

        for (int i = 0; i < maxFaceVertex();i++)
        {
            std::string name = "V" + std::to_string(i);
            PyList_SetItem(titles, i + 1, PyUnicode_FromString(name.c_str()));
            PyList_SetItem(formats, i + 1, PyUnicode_FromString("u4"));
        }

        PyDict_SetItemString(dict, "names", titles);
        PyDict_SetItemString(dict, "formats", formats);

        PyArray_Descr *dtype = nullptr;
        if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
            {}
    //         throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);

        npy_intp size = (npy_intp)faceCount();

        // This is a 1 x size array.
        m_faces = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

        MDAL_MeshEdgeIteratorH fi = MDAL_M_faceIterator(m_mdalMesh);

        int* vbuffer = new int[1024];
        int* obuffer = new int[1024];
        size_t count = 0;
        size_t tcount = 0;

        while (tcount < faceCount())
        {
            count = MDAL_FI_next(fi, 1024, obuffer, 1024, vbuffer);
            for (int i = 0; i < count; i++) 
            {
                char* p = (char *)PyArray_GETPTR1(m_faces, tcount + i);
                size_t offset;
                if (i==0)
                {
                    offset = 0;
                } else 
                {
                    offset = (size_t)obuffer[i-1];
                }
                uint32_t vert = (uint32_t)obuffer[i] - offset;

                std::memcpy(p, &vert, 4);

                for (int k=0; k<vert; k++)
                {
                    uint32_t v = (uint32_t)vbuffer[offset + k];
                    std::memcpy(p + ((k + 1) * 4), &v, 4);
                }
            }
            tcount += count;
        }
        delete [] vbuffer;
        delete [] obuffer;
        MDAL_FI_close(fi);
    }
    return m_faces;
}

PyArrayObject *Mesh::getEdges() 
{
    if (! m_edges) {
        PyObject* dict = PyDict_New();
        PyObject* formats = PyList_New(2);
        PyObject* titles = PyList_New(2);

        PyList_SetItem(titles, 0, PyUnicode_FromString("START"));
        PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));
        PyList_SetItem(titles, 1, PyUnicode_FromString("END"));
        PyList_SetItem(formats, 1, PyUnicode_FromString("u4"));

        PyDict_SetItemString(dict, "names", titles);
        PyDict_SetItemString(dict, "formats", formats);

        PyArray_Descr *dtype = nullptr;
        if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
            {}
    //         throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);

        npy_intp size = (npy_intp)edgeCount();

        // This is a 1 x size array.
        m_edges = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

        MDAL_MeshEdgeIteratorH ei = MDAL_M_edgeIterator(m_mdalMesh);

        int* sbuffer = new int[1024];
        int* ebuffer = new int[1024];
        size_t count = 0;
        size_t t_count = 0;

        while (t_count < edgeCount())
        {
            count = MDAL_EI_next(ei, 1024, sbuffer, ebuffer);
            for (int i = 0; i < count; i++) 
            {
                char* p = (char *)PyArray_GETPTR1(m_edges, t_count + i);
                
                uint32_t s = (uint32_t)sbuffer[i];
                uint32_t e = (uint32_t)ebuffer[i];
                std::memcpy(p, &s, 4);
                std::memcpy(p + 4, &e, 4);
            }
            t_count += count;
        }
        delete [] sbuffer;
        delete [] ebuffer;
        MDAL_EI_close(ei);
    }
    return m_edges;
}

bool Mesh::addVertices(PyArrayObject* vertices){
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return false;
    }
    
    Py_XINCREF(vertices);
    
    m_vertices = vertices;
    
    int xyz = 0;

    PyArray_Descr *dtype = PyArray_DTYPE(m_vertices);
    npy_intp ndims = PyArray_NDIM(m_vertices);
    npy_intp *shape = PyArray_SHAPE(m_vertices);
    size_t size = shape[0];
    PyObject* names_dict = PyDataType_FIELDS(dtype);
    int numFields = (names_dict == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(names_dict));

    PyObject *names = PyDict_Keys(names_dict);
    PyObject *values = PyDict_Values(names_dict);
    if (!names || !values)
    {
        return false;
    }
    //throw pdal_error("Bad field specification in numpy array.");

    size_t x_idx;
    size_t y_idx;
    size_t z_idx;
    
    for (int i = 0; i < numFields; ++i)
    {
        std::string name = toString(PyList_GetItem(names, i));
        if (name == "X")
        {
            xyz |= 1;
            x_idx = i;
        }
        else if (name == "Y")
        {
            xyz |= 2;
            y_idx = i;
        }
        else if (name == "Z")
        {
            xyz |= 4;
            z_idx = i;
        }
    }

    if (xyz != 0 && xyz != 7) 
    {
        return false;
    }
        //throw pdal_error("Array fields must contain all or none "
        //    "of X, Y and Z");
    if (xyz == 0 && ndims != 3)
    {
        return false;
    }
        //throw pdal_error("Array without named X/Y/Z fields "
        //        "must have three dimensions.");
    if (xyz == 0)
    {
        x_idx = 0;
        y_idx = 1;
        z_idx = 2;
    }
    
    double* v_array = new double[3 * size];
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(m_vertices, i);
        
        size_t idx = 3 * i;
        double* x = &v_array[idx];
        idx++;
        double* y = &v_array[idx];
        idx++;
        double* z = &v_array[idx];
        
        std::memcpy(x,p + ( 8 * x_idx ), 8);
        std::memcpy(y,p + ( 8 * y_idx ), 8);
        std::memcpy(z,p + ( 8 * z_idx ), 8);

    }
    MDAL_M_addVertices(m_mdalMesh,size, v_array);
    MDAL_Status status =  MDAL_LastStatus();
    if (status != MDAL_Status::None) 
        return false;
    return true;
}
    
bool Mesh::addFaces(PyArrayObject* faces, long int count){
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return false;
    }
    
    Py_XINCREF(faces);
    
    m_faces = faces;

    PyArray_Descr *dtype = PyArray_DTYPE(m_faces);
    npy_intp ndims = PyArray_NDIM(m_faces);
    npy_intp *shape = PyArray_SHAPE(m_faces);
    size_t size = shape[0];
    PyObject* names_dict = PyDataType_FIELDS(dtype);
    int numFields = (names_dict == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(names_dict));

    PyObject *names = PyDict_Keys(names_dict);
    PyObject *values = PyDict_Values(names_dict);
    if (!names || !values) {
        //throw pdal_error("Bad field specification in numpy array.");
    }

    if (numFields< 3) {}
        //throw pdal_error("Faces"
        //        "must have three dimensions.");
    int* f_array = new int[count];
    int* index_array = new int[size];
    size_t idx = 0;
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(m_faces, i);
        
        int face_size;
        std::memcpy(&face_size, p, 4);
        index_array[i] = face_size;
        
        for (int j = 0; j < face_size; j++)
        {
            std::memcpy(&f_array[idx], p + (j + 1) * 4, 4);
            idx++;
        }
    }
    
    MDAL_M_addFaces(m_mdalMesh, (int)size, index_array, f_array );
    MDAL_Status status =  MDAL_LastStatus();
    if (status != MDAL_Status::None) 
        return false;
    return true;
}

bool Mesh::addEdges(PyArrayObject* edges){
    if (_import_array() < 0) {}
            //throw pdal_error("Could not import numpy.core.multiarray.");
    
    Py_XINCREF(edges);
    
    m_edges = edges;

    PyArray_Descr *dtype = PyArray_DTYPE(m_edges);
    npy_intp ndims = PyArray_NDIM(m_edges);
    npy_intp *shape = PyArray_SHAPE(m_edges);
    size_t size = shape[0];
    PyObject* names_dict = PyDataType_FIELDS(dtype);
    int numFields = (names_dict == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(dtype->fields));

    PyObject *names = PyDict_Keys(names_dict);
    PyObject *values = PyDict_Values(names_dict);
    if (!names || !values) {}
        //throw pdal_error("Bad field specification in numpy array.");

    if (numFields != 2) {}
        //throw pdal_error("Array without named X/Y/Z fields "
        //        "must have three dimensions.");
        
    int ab;
    int start = 0;
    int end = 0;
        
    for (int i = 0; i < numFields; ++i)
    {
        std::string name = toString(PyList_GetItem(names, i));
        if (name == "START")
        {
            ab |= 1;
            start = i;
        }
        else if (name == "END")
        {
            ab |= 2;
            end = i;
        }
        PyObject *tup = PyList_GetItem(values, i);

        // Get offset.
        size_t offset = PyLong_AsLong(PySequence_Fast_GET_ITEM(tup, 1));

        if (ab != 0 && ab != 3) {}
            //throw pdal_error("Array fields must contain all or none "
            //    "of X, Y and Z");
        if (ab == 0 && ndims != 2) {}
            //throw pdal_error("Array without named X/Y/Z fields "
            //        "must have three dimensions.");
    }
    int* start_array = new int[size];
    int* end_array = new int[size];
    size_t idx = 0;
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(m_edges, i);
        
        std::memcpy(&start_array[i], p + ( start * 4 ), 4);
        std::memcpy(&end_array[i],p + (end * 4 ), 4);
    }
    MDAL_M_addEdges(m_mdalMesh, (int)size, start_array, end_array );
    MDAL_Status status =  MDAL_LastStatus();
    if (status != MDAL_Status::None) 
        return false;
    return true;
}


MDAL_DatasetGroupH Mesh::addGroup(const char* name, MDAL_DataLocation loc, bool hasScalar, const char* file)
{
    return addGroup(name, loc, hasScalar, file, MDAL_driverFromName(MDAL_M_driverName(m_mdalMesh)));
}
MDAL_DatasetGroupH Mesh::addGroup(const char* name, MDAL_DataLocation loc, bool hasScalar, const char* file, MDAL_DriverH drv)
{
    return MDAL_M_addDatasetGroup(m_mdalMesh, name, loc, hasScalar, drv, file);
}

int Mesh::edgeCount() 
{
    if (m_mdalMesh)
        return MDAL_M_edgeCount(m_mdalMesh);
    return 0;
}

int Mesh::vertexCount() 
{
    if (m_mdalMesh) 
        return MDAL_M_vertexCount(m_mdalMesh);
    return 0;
}

int Mesh::faceCount() 
{
    if (m_mdalMesh)
        return MDAL_M_faceCount(m_mdalMesh);
    return 0;
}

int Mesh::groupCount()
{
    if (m_mdalMesh)
        return MDAL_M_datasetGroupCount(m_mdalMesh);
    return 0;
}

int Mesh::maxFaceVertex(){
    if (m_mdalMesh)
        return MDAL_M_faceVerticesMaximumCount(m_mdalMesh);
    return 0;
}


const char* Mesh::getProjection() 
{
    if (m_mdalMesh)
        return MDAL_M_projection(m_mdalMesh);
    return nullptr;
}

MDAL_Status Mesh::setProjection(const char* proj)
{
    MDAL_ResetStatus();
    if (m_mdalMesh)
    {
        MDAL_M_setProjection(m_mdalMesh, proj);
    } else 
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_IncompatibleMesh, "Invalid Mesh (null)");
    }
    return MDAL_LastStatus();
}

void Mesh::getExtent(double* minX, double* maxX, double* minY, double* maxY)
{
    if (m_mdalMesh)
        MDAL_M_extent(m_mdalMesh, minX, maxX, minY, maxY);
    return;
}

const char* Mesh::getDriverName() 
{
    if (m_mdalMesh) 
        return MDAL_M_driverName(m_mdalMesh);
    return NULL;
}

MDAL_DatasetGroupH Mesh::getGroup(int index) 
{
    if (m_mdalMesh)
        return MDAL_M_datasetGroup(m_mdalMesh, index);
    return nullptr;
}

PyObject* Mesh::getMetadata() 
{
    if (! m_mdalMesh) 
        return defaultObject();
    PyObject* dict = PyDict_New();
    int count = MDAL_M_metadataCount(m_mdalMesh);
    for (int i =0; i < count; i++)
    {
        PyObject *key_py = PyBytes_FromString(MDAL_M_metadataKey(m_mdalMesh, i));
        PyObject *value_py = PyBytes_FromString(MDAL_M_metadataValue(m_mdalMesh,i));
        PyDict_SetItem(dict, key_py, value_py );
    }
    return dict;
}

MDAL_Status Mesh::setMetadata(PyObject* dict, const char* encoding )
{
    MDAL_ResetStatus();
    if (! m_mdalMesh)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_IncompatibleMesh, "Invalid Mesh (null)");
        return MDAL_LastStatus();
    }
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    PyObject* str;
    
    while (PyDict_Next(dict, &pos, &key, &value)) {
        const char *keyc = PyBytes_AS_STRING(PyUnicode_AsEncodedString(key, encoding, "~E~"));
        const char *valuec = PyBytes_AS_STRING(PyUnicode_AsEncodedString(value, encoding, "~E~"));
        MDAL_M_setMetadata( m_mdalMesh, keyc, valuec );
    }
    return MDAL_LastStatus();
}

} // namespace python
} // namespace mdal

