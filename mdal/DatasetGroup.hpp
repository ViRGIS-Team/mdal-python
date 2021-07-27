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
*     * Neither the name of Runette SOftware. nor the
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

#pragma once

#include <numpy/ndarraytypes.h>
#include "mdal.h"

#include <utility>
#include <memory>
#include <array>
#include <vector>

namespace mdal
{
namespace python
{
class Data
{
public:
    Data(MDAL_DatasetGroupH data);
    Data();

    ~Data();

    PyObject* getMetadata();
    MDAL_Status setMetadata(PyObject* dict,  const char* encoding );
    PyArrayObject* getDataAsDouble(int index);
    MDAL_Status setDataAsDouble(PyArrayObject* data, double time);
    PyArrayObject* getDataAsVolumeIndex(int index);
    PyArrayObject* getDataAsLevelCount(int index);
    PyArrayObject* getDataAsLevelValue(int index);
    MDAL_Status setDataAsVolume(PyArrayObject* data, PyArrayObject* verticalLevelCounts, PyArrayObject* verticalLevels, double time );

private:

    MDAL_DatasetGroupH m_data;
    PyArrayObject* m_dataset;


}; // class Data
} // namespace python
} // namespace mdal