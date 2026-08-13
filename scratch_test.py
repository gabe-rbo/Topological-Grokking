import numpy as np
import itertools

def euler_characteristic(simplices: np.ndarray) -> int:
    if len(simplices) == 0:
        return 0
    k = simplices.shape[1] - 1
    faces_by_dim = [set() for _ in range(k + 1)]
    for simplex in simplices:
        verts = tuple(sorted(int(v) for v in simplex))
        for dim in range(k + 1):
            for face in itertools.combinations(verts, dim + 1):
                faces_by_dim[dim].add(face)
    return sum((-1) ** dim * len(faces_by_dim[dim]) for dim in range(k + 1))

def _get_boundary(simplices: np.ndarray, dim: int) -> np.ndarray:
    if len(simplices) == 0:
        return np.zeros((0, dim), dtype=np.int64)
    face_counts = {}
    for s in simplices:
        for face in itertools.combinations(sorted(int(v) for v in s), dim):
            face_counts[face] = face_counts.get(face, 0) + 1
    res = [f for f, c in face_counts.items() if c == 1]
    if not res:
        return np.zeros((0, dim), dtype=np.int64)
    return np.array(res, dtype=np.int64)

def _analyze_link(simplices: np.ndarray, dim: int):
    if len(simplices) == 0:
        return True, None
    simps = [tuple(s) for s in simplices]
    if dim == 0:
        if len(simps) in (1, 2):
            return True, None
        return False, simps[-1]
        
    face_to_simplices = {}
    for i, s in enumerate(simps):
        for face in itertools.combinations(sorted(s), dim):
            face_to_simplices.setdefault(face, []).append(i)
            
    adj = {i: set() for i in range(len(simps))}
    for faces, simps_idx in face_to_simplices.items():
        if len(simps_idx) == 2:
            adj[simps_idx[0]].add(simps_idx[1])
            adj[simps_idx[1]].add(simps_idx[0])
            
    visited = set()
    components = []
    for i in range(len(simps)):
        if i not in visited:
            comp = set()
            q = [i]
            while q:
                curr = q.pop()
                if curr not in comp:
                    comp.add(curr)
                    q.extend(adj[curr] - comp)
            visited.update(comp)
            components.append(comp)
            
    if len(components) > 1:
        components.sort(key=len)
        target_idx = list(components[0])[0]
        comp_simps = np.array([simps[idx] for idx in components[0]], dtype=np.int64)
        comp_bnd = _get_boundary(comp_simps, dim)
        if len(comp_bnd) > 0:
            b_face = set(comp_bnd[0])
            for idx in components[0]:
                if b_face.issubset(set(simps[idx])):
                    return False, simps[idx]
        return False, simps[target_idx]
        
    chi = euler_characteristic(simplices)
    boundary = _get_boundary(simplices, dim)
    
    if len(boundary) == 0:
        if chi == 1 + (-1)**dim:
            return True, None
        return False, simps[0]
    else:
        if chi == 1:
            chi_b = euler_characteristic(boundary)
            if chi_b == 1 + (-1)**(dim - 1):
                return True, None
        b_face = set(boundary[0])
        for s in simps:
            if b_face.issubset(set(s)):
                return False, s
        return False, simps[0]

# Test 1-sphere
s1 = np.array([[0, 1], [1, 2], [2, 0]])
print("1-sphere:", _analyze_link(s1, 1))

# Test 1-disk
d1 = np.array([[0, 1], [1, 2]])
print("1-disk:", _analyze_link(d1, 1))

# Test two disjoint 1-disks
d2 = np.array([[0, 1], [2, 3]])
print("2 disjoint 1-disks:", _analyze_link(d2, 1))

# Test 2-sphere
# Tetrahedron boundary
s2 = np.array([[0,1,2], [0,1,3], [0,2,3], [1,2,3]])
print("2-sphere:", _analyze_link(s2, 2))

# Test 2-disk
# One triangle removed from tetrahedron
d2_surf = np.array([[0,1,2], [0,1,3], [0,2,3]])
print("2-disk:", _analyze_link(d2_surf, 2))

