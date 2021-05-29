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


namespace mdal
{
namespace python
{

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
        return nullptr;
    PyObject* dict = PyDict_New();
    int count = MDAL_G_metadataCount(m_data);

    for (int i =0; i < count; i++)
    {
        PyDict_SetItemString(dict, MDAL_G_metadataKey(m_data, i), PyBytes_FromString(MDAL_G_metadataValue(m_data,i)));
    }
    return dict;
}

PyArrayObject* Data::getDataAsDouble(int index) 
{
    if (! m_data) 
        return nullptr;
    int dsCount = MDAL_G_datasetCount(m_data);
    npy_intp valueCount = (npy_intp)MDAL_D_valueCount( MDAL_G_dataset(m_data, index));
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
                return nullptr;
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
                return nullptr;
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
        {}
//         throw pdal_error("Unable to build numpy dtype");
    Py_XDECREF(dict);

    // This is a dsCount x valueCount array.
    PyArrayObject* dataset = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
            1, &valueCount, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

    double* buffer = new double[dims * valueCount];
    size_t count = 0;
    int bufs = std::ceil(valueCount/1024);
    if (bufs == 0) bufs = 1;
    int indexStart = 0;
    int next = 1024;

    for (int j = 0; j < bufs; j++)
    {
        int remain = valueCount - j*1024;
        if (remain < 1024) next = remain;
        count = MDAL_D_data(MDAL_G_dataset(m_data, index), indexStart, next , type, buffer);
        if (count != next) return nullptr;
        int idx = 0;
        for (int i = 0; i < count; i++) 
        {
            char* p = (char *)PyArray_GETPTR1(dataset, (1024 * j) + i);
                
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
    
    return dataset;
}

} // namespace python
} // namespace mdal