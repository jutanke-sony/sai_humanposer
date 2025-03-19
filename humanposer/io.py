import numpy as np
from humanposer.uv_util import map_vertex_to_texture

def load_obj(file_path):
    vertices = []
    uv_coords = []
    vertex_faces = []
    texture_faces = []

    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("v "):
                parts = line.strip().split()
                vertex = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(vertex)
            elif line.startswith("vt "):
                parts = line.strip().split()
                uv = [float(parts[1]), float(parts[2])]
                uv_coords.append(uv)
            elif line.startswith("f "):
                parts = line.strip().split()
                vertex_face = []
                texture_face = []
                for part in parts[1:]:
                    indices = part.split("/")
                    vertex_face.append(int(indices[0]))
                    texture_face.append(int(indices[1]))
                vertex_faces.append(vertex_face)
                texture_faces.append(texture_face)

    return (
        np.array(vertices, dtype=np.float32),
        np.array(uv_coords, dtype=np.float32),
        np.array(vertex_faces, dtype=np.int64),
        np.array(texture_faces, dtype=np.int64),
    )
