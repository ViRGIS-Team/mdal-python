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

#include "DatasetGroup.hpp"

#include <numpy/ndarrayobject.h>
#include <string>
#include <cmath>
#include <cstring>
#include <iostream>


namespace mdal
{
namespace python
{
    
PyObject* defaultArray()
{
    const npy_intp dims = 1;
    return PyArray_SimpleNew(1, &dims, 1);
}

// Create new empty dataset
//
Data::Data() : m_data(nullptr), m_dataset(nullptr)
{
        if (_import_array() < 0)
        {}
}

Data::~Data()
{
    if (m_dataset)
        Py_XDECREF((PyObject *)m_dataset);
}

// Create dataset object
//
Data::Data(MDAL_DatasetGroupH data) : m_data(nullptr), m_dataset(nullptr)
{
        if (_import_array() < 0)
            return;
        m_data = data;
}

PyObject* Data::getMetadata() 
{
    if (! m_data) 
        return defaultArray();
    PyObject* dict = PyDict_New();
    int count = MDAL_G_metadataCount(m_data);

    for (int i =0; i < count; i++)
    {
        PyObject *key_py = PyBytes_FromString(MDAL_G_metadataKey(m_data, i));
        PyObject *value_py = PyBytes_FromString(MDAL_G_metadataValue(m_data,i));
        PyDict_SetItem(dict, key_py, value_py );
    }
    return dict;
}

MDAL_Status Data::setMetadata(PyObject* dict, const char* encoding )
{
    MDAL_ResetStatus();
    if (! m_data)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_InvalidData, "Invalid Data Group (null)");
        return MDAL_LastStatus();
    }
    PyObject *key, *value;
    Py_ssize_t pos = 0;
    PyObject* str;
    
    while (PyDict_Next(dict, &pos, &key, &value)) {
        const char *keyc = PyBytes_AS_STRING(PyUnicode_AsEncodedString(key, encoding, "~E~"));
        const char *valuec = PyBytes_AS_STRING(PyUnicode_AsEncodedString(value, encoding, "~E~"));
        MDAL_G_setMetadata( m_data, keyc, valuec );
    }
    return MDAL_LastStatus();
}

PyArrayObject* Data::getDataAsDouble(int index) 
{
    if (! m_data) 
        return (PyArrayObject*)defaultArray();
    
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return (PyArrayObject*)defaultArray();
    }

    MDAL_DatasetH datasetH = MDAL_G_dataset(m_data, index);
    if ( MDAL_LastStatus() != MDAL_Status::None)
    {
        return (PyArrayObject*)defaultArray();
    }
    
    npy_intp valueCount = (npy_intp)MDAL_D_valueCount( datasetH);
    int dims;
    MDAL_DataType type;
    if (MDAL_G_hasScalarData(m_data)) 
    {
        dims = 1;
        switch(MDAL_G_dataLocation(m_data)) 
        {
            case MDAL_DataLocation::DataOnVertices:
                type = MDAL_DataType::SCALAR_DOUBLE;
                break;
            case MDAL_DataLocation::DataOnFaces:
                type = MDAL_DataType::SCALAR_DOUBLE;
                break;
            case MDAL_DataLocation::DataOnEdges:
                type = MDAL_DataType::SCALAR_DOUBLE;
                break; 
            case MDAL_DataLocation::DataOnVolumes:
                type = MDAL_DataType::SCALAR_VOLUMES_DOUBLE;
                break;
            case MDAL_DataLocation::DataInvalidLocation:
                return (PyArrayObject*)defaultArray();
        }
    } else 
    {
        dims = 2;
        switch(MDAL_G_dataLocation(m_data)) 
        {
            case MDAL_DataLocation::DataOnVertices:
                type = MDAL_DataType::VECTOR_2D_DOUBLE;
                break;
            case MDAL_DataLocation::DataOnFaces:
                type = MDAL_DataType::VECTOR_2D_DOUBLE;
                break;
            case MDAL_DataLocation::DataOnEdges:
                type = MDAL_DataType::VECTOR_2D_DOUBLE;
                break; 
            case MDAL_DataLocation::DataOnVolumes:
                type = MDAL_DataType::VECTOR_2D_VOLUMES_DOUBLE;
                break;
            case MDAL_DataLocation::DataInvalidLocation:
                return (PyArrayObject*)defaultArray();
        }
    }
    PyObject* dict = PyDict_New();
    PyObject* formats = PyList_New(dims);
    PyObject* titles = PyList_New(dims);

    PyList_SetItem(titles, 0, PyUnicode_FromString("U"));
    PyList_SetItem(formats, 0, PyUnicode_FromString("f8"));
    if (dims == 2) 
    {
        PyList_SetItem(titles, 1, PyUnicode_FromString("V"));
        PyList_SetItem(formats, 1, PyUnicode_FromString("f8"));
    }

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Unable to build numpy dtype");
        return (PyArrayObject*)defaultArray();
    }

    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);
    Py_XDECREF(m_dataset);

    // This is a valueCount array.
    m_dataset = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &valueCount, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    double* buffer = new double[dims * 1024];
    size_t count = 0;
    int indexStart = 0;
    int next = 1024;

    while(true)
    {
        int remain = valueCount - indexStart;
        if (remain < 1024) next = remain;
        if (remain <= 0 ) break;
        count = MDAL_D_data(datasetH, indexStart, next , type, buffer);
        if (count != next) 
        {
            delete [] buffer;
            return (PyArrayObject *)defaultArray();
        }
        int idx = 0;
        for (int i = 0; i < count; i++) 
        {
            char* p = (char *)PyArray_GETPTR1(m_dataset, indexStart + i);
                
            for (int l =0; l < dims; l++)
            {
                double val = buffer[idx];
                idx++;
                std::memcpy(p, &val, 8);
                p += 8;
            }
        }
        indexStart += count;
    }
    delete [] buffer;
    
    return m_dataset;
}

PyArrayObject* Data::getDataAsVolumeIndex(int index)
{
    if (! m_data) 
        return (PyArrayObject*)defaultArray();
    
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return (PyArrayObject*)defaultArray();
    }

    if ( MDAL_LastStatus() != MDAL_Status::None)
    {
        return (PyArrayObject*)defaultArray();
    }
    
    npy_intp valueCount = (npy_intp)MDAL_M_faceCount( MDAL_G_mesh(m_data) );
    
    PyObject* dict = PyDict_New();
    PyObject* formats = PyList_New(1);
    PyObject* titles = PyList_New(1);

    PyList_SetItem(titles, 0, PyUnicode_FromString("value"));
    PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Unable to build numpy dtype");
        return (PyArrayObject*)defaultArray();
    }

    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);
    Py_XDECREF(m_dataset);

    // This is a dsCount x valueCount array.
    m_dataset = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &valueCount, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    int* buffer = new int[1024];
    size_t count = 0;
    int indexStart = 0;
    int next = 1024;

    while(true)
    {
        int remain = valueCount - indexStart;
        if (remain < 1024) next = remain;
        if (remain <= 0 ) break;
        count = MDAL_D_data( MDAL_G_dataset(m_data, index), indexStart, next , MDAL_DataType::FACE_INDEX_TO_VOLUME_INDEX_INTEGER, buffer);
        if (count != next) 
        {
            delete [] buffer;
            return (PyArrayObject *)defaultArray();
        }
        int idx = 0;
        for (int i = 0; i < count; i++) 
        {
            char* p = (char *)PyArray_GETPTR1(m_dataset, indexStart + i);
            uint32_t val = (uint32_t)buffer[idx];
            idx++;
            std::memcpy(p, &val, 4);
        }
        indexStart += count;
    }
    delete [] buffer;
    
    return m_dataset;
}

PyArrayObject* Data::getDataAsLevelCount(int index)
{
    if (! m_data) 
        return (PyArrayObject*)defaultArray();
    
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return (PyArrayObject*)defaultArray();
    }

    MDAL_DatasetH datasetH = MDAL_G_dataset(m_data, index);
    if ( MDAL_LastStatus() != MDAL_Status::None)
    {
        return (PyArrayObject*)defaultArray();
    }
    
    npy_intp valueCount = (npy_intp)MDAL_M_faceCount( MDAL_G_mesh(m_data));
    
    PyObject* dict = PyDict_New();
    PyObject* formats = PyList_New(1);
    PyObject* titles = PyList_New(1);

    PyList_SetItem(titles, 0, PyUnicode_FromString("value"));
    PyList_SetItem(formats, 0, PyUnicode_FromString("u4"));

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Unable to build numpy dtype");
        return (PyArrayObject*)defaultArray();
    }

    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);
    Py_XDECREF(m_dataset);

    // This is a dsCount x valueCount array.
    m_dataset = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &valueCount, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    int* buffer = new int[1024];
    size_t count = 0;
    int indexStart = 0;
    int next = 1024;

    while(true)
    {
        int remain = valueCount - indexStart;
        if (remain < 1024) next = remain;
        if (remain <= 0 ) break;
        count = MDAL_D_data( datasetH, indexStart, next , MDAL_DataType::VERTICAL_LEVEL_COUNT_INTEGER, buffer);

        if (count != next) 
        {
            delete [] buffer;
            return (PyArrayObject *)defaultArray();
        }
        int idx = 0;
        for (int i = 0; i < count; i++) 
        {
            char* p = (char *)PyArray_GETPTR1(m_dataset, indexStart + i);
            uint32_t val = (uint32_t)buffer[idx];
            idx++;
            std::memcpy(p, &val, 4);
        }
        indexStart += count;
    }
    delete [] buffer;
    
    return m_dataset;
}

PyArrayObject* Data::getDataAsLevelValue(int index)
{
    if (! m_data) 
        return (PyArrayObject*)defaultArray();
    
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return (PyArrayObject*)defaultArray();
    }

    MDAL_DatasetH datasetH = MDAL_G_dataset(m_data, index);
    if ( MDAL_LastStatus() != MDAL_Status::None)
    {
        return (PyArrayObject*)defaultArray();
    }
    
    npy_intp valueCount = (npy_intp)MDAL_M_faceCount( MDAL_G_mesh(m_data)) + (npy_intp)MDAL_D_valueCount( MDAL_G_dataset(m_data, index));
    
    PyObject* dict = PyDict_New();
    PyObject* formats = PyList_New(1);
    PyObject* titles = PyList_New(1);

    PyList_SetItem(titles, 0, PyUnicode_FromString("value"));
    PyList_SetItem(formats, 0, PyUnicode_FromString("f8"));

    PyDict_SetItemString(dict, "names", titles);
    PyDict_SetItemString(dict, "formats", formats);

    PyArray_Descr *dtype = nullptr;
    if (PyArray_DescrConverter(dict, &dtype) == NPY_FAIL)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Unable to build numpy dtype");
        return (PyArrayObject*)defaultArray();
    }

    Py_XDECREF(dict);
    Py_XDECREF(titles);
    Py_XDECREF(formats);
    Py_XDECREF(m_dataset);

    // This is a dsCount x valueCount array.
    m_dataset = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &valueCount, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    double* buffer = new double[1024];
    size_t count = 0;
    int indexStart = 0;
    int next = 1024;

    while(true)
    {
        int remain = valueCount - indexStart;
        if (remain < 1024) next = remain;
        if (remain <= 0 ) break;
        count = MDAL_D_data( datasetH, indexStart, next , MDAL_DataType::VERTICAL_LEVEL_DOUBLE, buffer);

        if (count != next) 
        {
            delete [] buffer;
            return (PyArrayObject *)defaultArray();
        }
        int idx = 0;
        for (int i = 0; i < count; i++) 
        {
            char* p = (char *)PyArray_GETPTR1(m_dataset, indexStart + i);
            double val = buffer[idx];
            idx++;
            std::memcpy(p, &val, 8);
        }
        indexStart += count;
    }
    delete [] buffer;
    return m_dataset;
}

MDAL_Status Data::setDataAsDouble(PyArrayObject* data, double time)
{
    if (! m_data)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_IncompatibleDatasetGroup, "NULL: No Group");
        return MDAL_LastStatus();
    }
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return MDAL_LastStatus();
    }
    
    Py_XINCREF(data);

    PyArray_Descr *dtype = PyArray_DTYPE(data);
    npy_intp ndims = PyArray_NDIM(data);
    npy_intp *shape = PyArray_SHAPE(data);
    size_t size = shape[0];
    PyObject *names_dict = PyDataType_FIELDS(dtype);
    int numFields = (names_dict == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(names_dict));

    PyObject *names = PyDict_Keys(names_dict);
    PyObject *values = PyDict_Values(names_dict);
    if (!names || !values) 
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Bad field specification in numpy array.");
    }
    
    double* d_array = new double[size * numFields];
    size_t idx = 0;
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(data, i);
        
        for (int j = 0; j < numFields; j++)
        {
            std::memcpy(&d_array[idx], p + (j) * 8, 8);
            idx++;
        }
    }
    
    MDAL_G_addDataset(m_data, time, d_array, nullptr);

    Py_XDECREF(data);
    
    return MDAL_LastStatus();
}

MDAL_Status Data::setDataAsVolume(PyArrayObject* data, PyArrayObject* verticalLevelCounts, PyArrayObject* verticalLevels, double time )
{
    if (! m_data)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_IncompatibleDatasetGroup, "NULL: No Group");
        return MDAL_LastStatus();
    }
    if (! m_data)
        {
            MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_IncompatibleDatasetGroup, "NULL: No Group");
            return MDAL_LastStatus();
        }
    MDAL_ResetStatus();
    if (_import_array() < 0)
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_FailToWriteToDisk, "Could not import numpy.core.multiarray.");
        return MDAL_LastStatus();
    }

    PyArray_Descr *dtype = PyArray_DTYPE(verticalLevelCounts);
    npy_intp ndims = PyArray_NDIM(verticalLevelCounts);
    npy_intp *shape = PyArray_SHAPE(verticalLevelCounts);
    size_t size = shape[0];

    PyObject *names_dict = PyDataType_FIELDS(dtype);
    PyObject *names = PyDict_Keys(names_dict);
    PyObject *values = PyDict_Values(names_dict);
    if (!names || !values) 
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Bad field specification in numpy array.");
    }
    
    int* vl_array = new int[size];
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(verticalLevelCounts, i);
        
        std::memcpy(&vl_array[i], p, 4);
    }
    
    dtype = PyArray_DTYPE(verticalLevels);
    ndims = PyArray_NDIM(verticalLevels);
    shape = PyArray_SHAPE(verticalLevels);
    size = shape[0];

    names_dict = PyDataType_FIELDS(dtype);
    names = PyDict_Keys(names_dict);
    values = PyDict_Values(names_dict);
    if (!names || !values) 
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Bad field specification in numpy array.");
    }
    
    double* v_array = new double[size];
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(verticalLevels, i);
        
        std::memcpy(&v_array[i], p, 8);
    }
    
    Py_XINCREF(data);
    Py_XINCREF(verticalLevelCounts);
    Py_XINCREF(verticalLevels);

    dtype = PyArray_DTYPE(data);
    ndims = PyArray_NDIM(data);
    shape = PyArray_SHAPE(data);
    size = shape[0];
    names_dict = PyDataType_FIELDS(dtype);
    int numFields = (names_dict == Py_None) ?
        0 :
        static_cast<int>(PyDict_Size(names_dict));

    names = PyDict_Keys(names_dict);
    values = PyDict_Values(names_dict);
    if (!names || !values) 
    {
        MDAL_SetStatus(MDAL_LogLevel::Error, MDAL_Status::Err_UnsupportedElement, "Bad field specification in numpy array.");
    }
    
    double* d_array = new double[size * numFields];
    size_t idx = 0;
    
    for (int i = 0; i < size; i++) 
    {
        char* p = (char *)PyArray_GETPTR1(data, i);
        
        for (int j = 0; j < numFields; j++)
        {
            std::memcpy(&d_array[idx], p + (j) * 8, 8);
            idx++;
        }
    }
    
    MDAL_G_addDataset3D(m_data, time, d_array, vl_array, v_array);

    Py_XDECREF(data);
    Py_XDECREF(verticalLevelCounts);
    Py_XDECREF(verticalLevels);
    return MDAL_LastStatus();
}

} // namespace python
} // namespace mdal