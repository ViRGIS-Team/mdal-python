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


namespace mdal
{
namespace python
{

// Create new empty mesh
//
Mesh::Mesh() : m_verteces(nullptr), m_faces(nullptr), m_edges(nullptr), m_mdalMesh(nullptr)
{
    hasMesh = false;
    if (_import_array() < 0)
        {}
        //throw pdal_error("Could not import numpy.core.multiarray.");
}

Mesh::~Mesh()
{
    if (m_verteces)
        Py_XDECREF((PyObject *)m_verteces);
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
Mesh::Mesh(const char* uri) : m_verteces(nullptr), m_faces(nullptr), m_edges(nullptr), m_mdalMesh(nullptr)
{
    if (_import_array() < 0)
        return;
    m_mdalMesh = MDAL_LoadMesh(uri);
    if (MDAL_LastStatus() == MDAL_Status::None){
        hasMesh=true;
    }
}

PyArrayObject *Mesh::getVertices() 
{
    if (! m_verteces) {
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
        m_verteces = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

        MDAL_MeshVertexIteratorH vi = MDAL_M_vertexIterator(m_mdalMesh);

        double* buffer = new double[3072];
        size_t count = 0;
        int bufs = std::ceil(size/1024);
        if (bufs == 0) bufs = 1;

        for (int j = 0; j < bufs; j++)
        {
            count = MDAL_VI_next(vi, 1024, buffer);
            int idx = 0;
            for (int i = 0; i < count; i++) 
            {
                char* p = (char *)PyArray_GETPTR1(m_verteces, (1024 * j) + i);
                
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
        }
        delete [] buffer;
        MDAL_VI_close(vi);
    }
    return m_verteces;
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

        npy_intp size = (npy_intp)edgeCount();

        // This is a 1 x size array.
        m_edges = (PyArrayObject *)PyArray_NewFromDescr(&PyArray_Type, dtype,
                1, &size, 0, nullptr, NPY_ARRAY_CARRAY, nullptr);

        MDAL_MeshEdgeIteratorH ei = MDAL_M_edgeIterator(m_mdalMesh);

        int* sbuffer = new int[1024];
        int* ebuffer = new int[1024];
        size_t count = 0;
        int bufs = std::ceil(size/1024);
        if (bufs == 0) bufs = 1;

        for (int j = 0; j < bufs; j++)
        {
            count = MDAL_EI_next(ei, 1024, sbuffer, ebuffer);
            for (int i = 0; i < count; i++) 
            {
                char* p = (char *)PyArray_GETPTR1(m_edges, (1024 * j) + i);
                
                uint32_t s = (uint32_t)sbuffer[i];
                uint32_t e = (uint32_t)ebuffer[i];
                std::memcpy(p, &s, 4);
                std::memcpy(p + 4, &e, 4);
            }
        }
        delete [] sbuffer;
        delete [] ebuffer;
        MDAL_EI_close(ei);
    }
    return m_edges;
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
    return NULL;
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


bool Mesh::rowMajor() const
{
    return m_rowMajor;
}

Mesh::Shape Mesh::shape() const
{
    return m_shape;
}


MeshIter& Mesh::iterator()
{
    MeshIter *it = new MeshIter(*this);
    m_iterators.push_back(std::unique_ptr<MeshIter>(it));
    return *it;
}

MeshIter::MeshIter(Mesh& mesh)
{
    m_iter = NpyIter_New(mesh.getVertices(),
        NPY_ITER_EXTERNAL_LOOP | NPY_ITER_READONLY | NPY_ITER_REFS_OK,
        NPY_KEEPORDER, NPY_NO_CASTING, NULL);
    //if (!m_iter)
        //throw pdal_error("Unable to create numpy iterator.");

    char *itererr;
    m_iterNext = NpyIter_GetIterNext(m_iter, &itererr);
    if (!m_iterNext)
    {
        NpyIter_Deallocate(m_iter);
        //throw pdal_error(std::string("Unable to create numpy iterator: ") +
            //itererr);
    }
    m_data = NpyIter_GetDataPtrArray(m_iter);
    m_stride = NpyIter_GetInnerStrideArray(m_iter);
    m_size = NpyIter_GetInnerLoopSizePtr(m_iter);
    m_done = false;
}

MeshIter::~MeshIter()
{
    NpyIter_Deallocate(m_iter);
}

MeshIter& MeshIter::operator++()
{
    if (m_done)
        return *this;

    if (--(*m_size))
        *m_data += *m_stride;
    else if (!m_iterNext(m_iter))
        m_done = true;
    return *this;
}

MeshIter::operator bool () const
{
    return !m_done;
}

char * MeshIter::operator * () const
{
    return *m_data;
}


} // namespace python
} // namespace mdal

