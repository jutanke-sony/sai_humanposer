
def map_vertex_to_texture(vertex_faces, texture_faces):
    vertex_to_texture = {}
    for v_face, t_face in zip(vertex_faces, texture_faces):
        for v_idx, t_idx in zip(v_face, t_face):
            if v_idx not in vertex_to_texture:
                vertex_to_texture[v_idx.item()] = set()
            vertex_to_texture[v_idx].add(t_idx.item())
    
    vertex_to_texture_ = {}
    for k, v in vertex_to_texture.items():
        vertex_to_texture_[k] = list(sorted(v))
    return vertex_to_texture