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

from mdal.libmdalpython import PyMesh, DatasetGroup, MDAL_DataLocation
import meshio
import numpy as np
from typing import Union, List

try:
    import open3d as o3d
    o3d_flag = True
except:
    o3d_flag = False


class MDAL_transform:

    @classmethod
    def to_meshio(cls, arg: Union[PyMesh, DatasetGroup], time = 0) -> meshio.Mesh:
        """attmempts to convert an MDAL mesh or dataset into a meshio mesh

        CONSTRAINT : only converts scalar datasets.

        CONSTRAINT : will convert volumetric datasets if given as a group argument

        NOTE: All MDAL_transform utilities are beta

        """
        if type(arg) == PyMesh:
            mesh = arg
        else:
            mesh = arg.mesh
        vertices = mesh.vertices
        lines = []
        faces = []
        tris = []
        quads = []
        if mesh.face_count == 0:
            if mesh.edge_count == 0:
                return None
            #  address all 1 D meshes as lines
            lines = mesh.edges
            cells = [
                ("line", np.stack((lines['START'], lines['END']), 1))
            ]
        else:
            faces = mesh.faces
            lines = mesh.edges
            tris = faces[faces['Vertices'] == 3]
            quads = faces[faces['Vertices'] == 4]
            cells = []
            # address all 2D meshes with 2, 3 or 4 sides as lines / triangles / quads
            if len(lines) > 0:
                cells.append(
                    ("line", np.stack((lines['START'], lines['END']), 1)))
            if len(tris) > 0:
                cells.append(("triangle", np.stack(
                    (tris['V0'], tris['V1'], tris['V2']), 1)))
            if len(quads) > 0:
                cells.append(
                    ("quad", np.stack((quads['V0'], quads['V1'], quads['V2'], quads['V3']), 1)))
        points = np.stack((vertices['X'], vertices['Y'], vertices['Z']), 1)
        point_data = {}
        cell_data = {}
        for group in mesh.groups:
            if group.name == "Bed Elevation":
                continue
            if type(arg) == DatasetGroup and arg.name != group.name:
                continue
            if group.location == 1 and group.has_scalar:
                point_data.update({group.name: group.data(time)['U']})
            elif group.location == 2 and group.has_scalar:
                #  Face data gets written only to triangles and quads but NaN also needs to be written to lines
                data_out = []
                data_in = group.data(time)['U']
                if len(lines) > 0:
                    data_out.append([float("NaN") for item in lines])
                if len(tris) > 0:
                    data_out.append(data_in[faces['Vertices'] == 3])
                if len(quads) > 0:
                    data_out.append(data_in[faces['Vertices'] == 4])
                cell_data.update({group.name: data_out})
            elif group.location == 3 and group.has_scalar:
                if type(arg) == PyMesh:
                    continue
                #  create the mesh as a mixture of Wedge and Hexahedron
                (points, cells, cell_data) = cls._vol_to_voxel(group, time)

            elif group.location == 4 and group.has_scalar:
                #  Edge data gets written only to lines but NaN also needs to be written to triangles and quads
                data_out = []
                if len(lines) > 0:
                    data_out.append(group.data(time)['U'])
                if len(tris) > 0:
                    data_out.append(
                        [float("NaN") for item in faces[faces['Vertices'] == 3]])
                if len(quads) > 0:
                    data_out.append(
                        [float("NaN") for item in faces[faces['Vertices'] == 4]])
                cell_data.update({group.name: data_out})
        return meshio.Mesh(
            points,
            cells,
            point_data,
            cell_data
        )

    @classmethod
    def from_meshio(cls, mesh: meshio.Mesh):
        """attempts to convert meshio.Mesh to mdal.PyMesh

        CONSTRAINT : only converts scalar datasets.

        CONSTRAINT : will not convert volumetric datasets

        NOTE: All MDAL_transform utilities are beta

        """
        mdal_mesh = PyMesh()
        mdal_mesh.vertices = np.fromiter(map(tuple, mesh.points), np.dtype(
            [('X', '<f8'), ('Y', '<f8'), ('Z', '<f8')]))
        faces = np.empty((0), np.dtype(
            [('Vertices', '<u4'), ('V0', '<u4'), ('V1', '<u4'), ('V2', '<u4'), ('V3', '<u4')]))
        for cell_type in mesh.cells:
            if len(cell_type) > 0:
                if cell_type.type == "line":
                    mdal_mesh.edges = np.fromiter(map(tuple, cell_type.data), np.dtype([
                                                  ('START', '<u4'), ('END', '<u4')]))
                    continue
                if cell_type.type == "triangle":
                    tris = np.fromiter(map(lambda x: tuple(np.insert(np.insert(x, 3, 0), 0, 3)), cell_type.data), np.dtype(
                        [('Vertices', '<u4'), ('V0', '<u4'), ('V1', '<u4'), ('V2', '<u4'), ('V3', '<u4')]))
                    faces = np.append(faces, tris)
                    continue
                if cell_type.type == "quad":
                    quads = np.fromiter(map(lambda x: tuple(np.insert(x, 0, 4)), cell_type.data), np.dtype(
                        [('Vertices', '<u4'), ('V0', '<u4'), ('V1', '<u4'), ('V2', '<u4'), ('V3', '<u4')]))
                    faces = np.append(faces, quads)
                    continue
                if cell_type.type == "wedge":
                    continue
                if cell_type.type == "hexahedron":
                    continue
        if len(faces) > 0:
            mdal_mesh.faces = faces
        return mdal_mesh

    @classmethod
    def _vol_to_voxel(cls, group: DatasetGroup, index: int = 0):
        """internal method to convert a volumetric DatasetGroup into 3D vertices and voxels with data

        NOTE: All MDAL_transform utilities are beta

        TODO : convert to C++
        """
        # TODO - convert this to cpp
        if group.location != MDAL_DataLocation.DataOnVolumes:
            raise ValueError("Not a Volumetric Data set")
        mesh = group.mesh
        vertices2D = mesh.vertices
        faces = mesh.faces
        wedges = []
        wedge_data = []
        hexa = []
        hexa_data = []
        vertices3D = []
        v_index = []
        (level_counts, level_values, face_ind) = group.volumetric(index)
        data = group.data(index)
        #  create adjancency matrix
        for i in range(0, len(vertices2D)):
            neighbours = []
            for j in range(0, len(faces)):
                face = faces[j]
                for l in range(0, face['Vertices']):
                    if face[l + 1] == i:
                        neighbours.append(j)
                        break
            v_index.append(len(vertices3D))
            # case of a single neighbour
            if len(neighbours) == 1:
                curr_level = 0
                for j in range(0, level_counts[neighbours[0]][0]):
                    if level_values[face_ind[neighbours[0]][0] + neighbours[0] + j][0] != curr_level:
                        curr_level = level_values[face_ind[neighbours[0]]
                                                  [0] + neighbours[0] + j][0]
                        vertices3D.append(
                            [vertices2D[i]['X'], vertices2D[i]['Y'], curr_level]
                        )
                continue
            bott_buff = []
            z = []
            for face_id in neighbours:
                bott_buff.append(
                    level_values[face_ind[face_id][0] + face_id][0])
                z.append(None)
            # find the lowest cells
            min_idx = bott_buff.index(min(bott_buff))
            z[min_idx] = -1
            last_level = 0
            while True:
                # check which new columes match this voxel
                curr_level = level_values[face_ind[neighbours[min_idx]][0] + neighbours[min_idx] +
                                          z[min_idx]][0]
                this_z = []
                for face_id in range(0, len(neighbours)):
                    if z[face_id] is None:
                        # find if the current cell is the closest to this bottom cell
                        this_level = bott_buff[face_id]
                        next_level = level_values[face_ind[neighbours[min_idx]][0] + neighbours[min_idx]
                                                  + z[min_idx] + 1][0]
                        if min(curr_level, next_level) <= this_level <= max(curr_level, next_level):
                            z[face_id] = 0
                            this_z.append(this_level)
                    else:
                        if z[face_id] < level_counts[neighbours[face_id]][0]:
                            z[face_id] += 1
                            this_z.append(
                                level_values[face_ind[neighbours[face_id]][0] + neighbours[face_id] + z[min_idx]][0])
                # the z value for this vertex is the average z values for the neighbours
                z_val = sum(this_z) / len(this_z)
                if z_val != last_level:
                    vertices3D.append(
                        [vertices2D[i]['X'], vertices2D[i]['Y'], z_val])
                    last_level = z_val
                # cope with reaching the top of a colume
                if z[min_idx] == level_counts[neighbours[min_idx]][0]:
                    min_idx = None
                    for face_id in range(0, len(neighbours)):
                        # to cope with columes that do not overlap
                        if z[face_id] is None:
                            min_idx = face_id
                            continue
                        # to cope with columes that overlap
                        if z[face_id] < level_counts[neighbours[face_id]][0]:
                            min_idx = face_id
                            break
                #  nothing more to process
                if min_idx is None:
                    break
                # if there is no overlap - avoiding pumping None into the alogrithm
                if z[min_idx] == None:
                    z[min_idx] = 0

        for i in range(0, len(faces)):
            last_level = level_values[face_ind[i][0] + i][0]
            count = 1
            face = faces[i]
            for j in range(0, level_counts[i][0]):
                level = level_values[face_ind[i][0] + i + j + 1][0]
                if level != last_level:
                    if face[0] == 3:
                        wedges.append([
                                      v_index[face[1] + count - 1],
                                      v_index[face[2] + count - 1],
                                      v_index[face[3] + count - 1],
                                      v_index[face[1] + count],
                                      v_index[face[2] + count],
                                      v_index[face[3] + count]
                                      ])
                        wedge_data.append(data[face_ind[i][0] + j][0])
                    if face[0] == 4:
                        hexa.append([v_index[face[1] + count - 1],
                                     v_index[face[2] + count - 1],
                                     v_index[face[3] + count - 1],
                                     v_index[face[4] + count - 1],
                                     v_index[face[1] + count],
                                     v_index[face[2] + count],
                                     v_index[face[3] + count],
                                     v_index[face[4] + count]
                                     ])
                        hexa_data.append(data[face_ind[i][0] + j][0])
        cells = []
        cell_data = []
        if len(wedges) > 0:
            cells.append(("wedge", wedges))
            cell_data.append(wedge_data)
        if len(hexa) > 0:
            cells.append(("hexahedron", hexa))
            cell_data.append(hexa_data)
        return (vertices3D, cells, {group.name: cell_data})

    @classmethod
    def to_triangular_mesh(cls, group: DatasetGroup, index: int = 0):
        if not o3d_flag:
            raise ImportError(
                "Could not find Open3D. Try `pip install open3d`")

    @classmethod
    def to_point_cloud(cls, group: DatasetGroup, index: int = 0):
        if not o3d_flag:
            raise ImportError(
                "Could not find Open3D. Try `pip install open3d`")
        # TODO - convert this to cpp
        if group.location != MDAL_DataLocation.DataOnVolumes:
            raise ValueError("Not a Volumetric Data set")
        mesh = group.mesh
        data = group.data(index)
        points = np.empty((data.shape[0], 3))
        (level_counts, level_values, face_ind) = group.volumetric(index)
        faces = mesh.faces
        for i in range(0, len(faces)):
            face = faces[i]
            x = []
            y = []
            z = []
            for j in range(1, face[0]):
                x.append(mesh.vertices[face[j]]['X'])
                y.append(mesh.vertices[face[j]]['Y'])
                z.append(mesh.vertices[face[j]]['Z'])
            centroid = (np.average(x), np.average(y), np.average(z))
            z_last = level_values[face_ind[i][0] + i][0]
            for k in range(0, level_counts[i][0]):
                z_next = level_values[face_ind[i][0] + i + k + 1][0]
                z = (z_last + z_next) / 2
                points[face_ind[i][0] + k] = (centroid[0], centroid[1],
                                              centroid[2] + z)
        pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

        return pcd
