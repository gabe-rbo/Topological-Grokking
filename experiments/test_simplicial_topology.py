"""
Regression tests for tangential_delaunay's simplicial-complex helpers
(_get_boundary, _analyze_link, euler_characteristic) against manifolds with
known topology. Per _analyze_link's own contract, BOTH a closed dim-sphere
(an interior vertex's link) and a dim-disk with one boundary component (a
boundary vertex's link) are valid manifold links; only a disconnected link
(more than one component, e.g. two disjoint disks) is invalid. Covered here:
a 1-sphere (closed loop) and a 1-disk (open path) are each valid; two
disjoint 1-disks are invalid; a 2-sphere (tetrahedron boundary) and a 2-disk
(one face removed from a tetrahedron) are each valid.

Originally a standalone prototype (scratch_test.py) written while designing
_analyze_link, before that logic was folded into tangential_delaunay.py;
kept here as a regression test against the shipped implementation instead
of a throwaway copy of the algorithm.

Run directly (`python -m experiments.test_simplicial_topology`) or via pytest.
"""
import numpy as np

from experiments.tangential_delaunay import _analyze_link, _get_boundary, euler_characteristic


def test_get_boundary_of_open_path():
    # Two edges forming an open path 0-1-2: boundary is the two endpoints.
    boundary = _get_boundary([(0, 1), (1, 2)], dim=1)
    assert sorted(boundary) == [(0,), (2,)]


def test_get_boundary_of_closed_triangle_is_empty():
    boundary = _get_boundary([(0, 1), (1, 2), (0, 2)], dim=1)
    assert boundary == []


def test_1_sphere_link_is_valid():
    # A circle (three edges closing up): a valid, closed 1-sphere link
    # (an interior vertex).
    is_valid, prune = _analyze_link([(0, 1), (1, 2), (0, 2)], dim=1)
    assert is_valid and prune is None


def test_1_disk_link_is_valid():
    # An open path of two edges: a valid 1-disk link (a boundary vertex) —
    # its boundary is the two endpoints, a valid 0-sphere.
    is_valid, prune = _analyze_link([(0, 1), (1, 2)], dim=1)
    assert is_valid and prune is None


def test_disjoint_1_disks_link_is_invalid():
    # Two disconnected edges: fails the connectivity check — a link must be
    # a single sphere or disk, not multiple components.
    is_valid, prune = _analyze_link([(0, 1), (2, 3)], dim=1)
    assert not is_valid and prune is not None


def test_2_sphere_link_is_valid():
    # Tetrahedron boundary: a closed, valid 2-sphere link (an interior vertex).
    simplices = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    is_valid, prune = _analyze_link(simplices, dim=2)
    assert is_valid and prune is None


def test_2_disk_link_is_valid():
    # One face removed from the tetrahedron boundary: a valid 2-disk link
    # (a boundary vertex) — its boundary is a triangle, a valid 1-sphere.
    simplices = [(0, 1, 2), (0, 1, 3), (0, 2, 3)]
    is_valid, prune = _analyze_link(simplices, dim=2)
    assert is_valid and prune is None


def test_euler_characteristic_matches_known_values():
    circle = np.array([[0, 1], [1, 2], [2, 0]])
    assert euler_characteristic(circle) == 0
    tetrahedron_boundary = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    assert euler_characteristic(tetrahedron_boundary) == 2


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"[ok] {_name}")
    print("All simplicial-topology regression tests passed.")
