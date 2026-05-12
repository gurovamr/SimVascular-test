#!/usr/bin/env python3
"""
prepare_sv_project.py
─────────────────────
Converts CoW centerline extraction output into a SimVascular project
that can be opened directly in the GUI for meshing and CFD simulation.

Both CT and MR modalities of the same patient (p025) are supported.
Run once per modality to get two independent SV projects.

Usage
-----
    # Recommended — use SimVascular's bundled Python + VTK:
    simvascular --python -- /home/gurovamr/SimVascular/SimVascular-test/scripts/prepare_sv_project.py --modality ct
    simvascular --python -- /home/gurovamr/SimVascular/SimVascular-test/scripts/prepare_sv_project.py --modality mr

    # Alternative — if system Python has VTK installed:
    python3 prepare_sv_project.py --modality ct

What the script produces
------------------------
  <output_dir>/
    simvascular.proj          ← project file (open this in SV)
    Images/
      image_information.xml   ← placeholder (no image needed for mesh-based workflow)
    Models/
      p025_ct.vtp             ← capped surface model (wall + inlet/outlet faces)
      p025_ct.mdl             ← SV model descriptor (face names and types)
    Paths/
      BA.pth, R-ICA.pth, …   ← centerline paths per vessel segment
    Simulations/
      p025_ct_job.sjb         ← simulation job with estimated boundary conditions

After running, open SimVascular GUI:
  1. File > Open Project → select output directory
  2. Models tab    → inspect face labels (wall / inlet / outlet caps)
  3. Mesh tab      → run TetGen mesher, name it p025_ct_mesh
  4. Simulations tab → open .sjb, review BCs, Create Data Files, run solver
"""

import os
import sys
import json
import argparse
import time
import numpy as np

try:
    import vtk
    from vtk.util.numpy_support import vtk_to_numpy
except ImportError:
    print("ERROR: VTK not found.")
    print("Run with:  simvascular --python -- prepare_sv_project.py --modality ct")
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration — edit paths here if needed
# ═══════════════════════════════════════════════════════════════════════════════

DATA_DIR    = "/home/gurovamr/SimVascular/SimVascular-test-local/data"
OUTPUT_BASE = "/home/gurovamr/SimVascular/SimVascular-test/simvascular"
PATIENT_ID  = "p025"

# TopCoW segment label → artery name (13-class scheme, class 0 = background)
LABEL_MAP = {
    1:  "BA",
    2:  "R-PCA",
    3:  "L-PCA",
    4:  "R-ICA",
    5:  "R-MCA",
    6:  "L-ICA",
    7:  "L-MCA",
    8:  "R-Pcom",
    9:  "L-Pcom",
    10: "Acom",
    11: "R-ACA",
    12: "L-ACA",
}

# Node keyword → inlet (flow in) or outlet (flow out)
INLET_KEYWORDS  = ["_start"]   # matches "ba_start", "ica_start"
OUTLET_KEYWORDS = ["_end"]     # matches "pca_end", "mca_end", "aca_end"

# Fluid properties (SV convention: CGS units)
FLUID_DENSITY   = 1.06   # g/cm³
FLUID_VISCOSITY = 0.04   # g/(cm·s)  [= Poise]

# Estimated steady inflow per inlet (mm³/s, negative = into domain)
# Based on published CoW flow values; adjust before running solver
INFLOW_MM3_S = {
    "BA":    -333.0,   # Basilar Artery  ≈ 200 mL/min
    "R-ICA": -500.0,   # Right ICA       ≈ 300 mL/min
    "L-ICA": -500.0,   # Left ICA        ≈ 300 mL/min
}
DEFAULT_INFLOW      = -333.0   # fallback for unlabelled inlets
DEFAULT_RESISTANCE  = 1500.0   # dyn·s/cm⁵  — outlet Windkessel resistance

# Per-endpoint sphere radius for dome removal (based on local vessel radius)
SPHERE_RADIUS_MULTIPLIER = 2.0   # × local vessel mean radius
SPHERE_RADIUS_MIN_MM     = 1.5   # floor — never smaller than this (mm)
SPHERE_RADIUS_MAX_MM     = 3.5   # ceiling — avoids cutting through Acom/short bridges

# Face colours (R,G,B) used in the .mdl file for display in SV
WALL_COLOR = ("0.705882", "0.298039", "0.701961")
CAP_COLORS = [
    ("0.223529", "0.666667", "0.250980"),
    ("0.843137", "0.254902", "0.254902"),
    ("0.831373", "0.694118", "0.247059"),
    ("0.000000", "0.447059", "0.741176"),
    ("0.850980", "0.325490", "0.098039"),
    ("0.929412", "0.694118", "0.125490"),
    ("0.494118", "0.184314", "0.556863"),
    ("0.466667", "0.674510", "0.188235"),
    ("0.301961", "0.745098", "0.933333"),
    ("0.635294", "0.078431", "0.184314"),
    ("0.121569", "0.470588", "0.705882"),
    ("0.200000", "0.627451", "0.172549"),
]


# ═══════════════════════════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════════════════════════

def read_vtp(filepath):
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filepath)
    reader.Update()
    return reader.GetOutput()


def write_vtp(polydata, filepath):
    writer = vtk.vtkXMLPolyDataWriter()
    writer.SetFileName(filepath)
    writer.SetInputData(polydata)
    writer.Write()
    print(f"    wrote: {os.path.basename(filepath)}")


def write_text(lines, filepath):
    with open(filepath, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"    wrote: {os.path.basename(filepath)}")


def make_dirs(proj_dir):
    for sub in ["Images", "Paths", "Segmentations", "Models",
                "Meshes", "MultiPhysics", "ROMSimulations", "Simulations"]:
        os.makedirs(os.path.join(proj_dir, sub), exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0 — Collect degree-1 nodes with unique keys (handles duplicate names)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_degree1_nodes(nodes_dict):
    """
    Returns a dict keyed by "seg<id>_<sanitised_type>" so that nodes with the
    same type name (e.g. "ICA start" in both seg 4 and seg 6) are not lost.

    Values: {"coords": np.array, "seg_id": str, "node_type": str, "key": str}
    """
    result = {}
    for seg_id, seg_data in nodes_dict.items():
        for node_type, node_list in seg_data.items():
            for node in node_list:
                if node["degree"] != 1:
                    continue
                sanitised = (node_type.lower()
                             .replace(" ", "_")
                             .replace("-", "_"))
                key = f"seg{seg_id}_{sanitised}"
                # Keep only unique combinations (same point may appear in
                # multiple segments as a shared boundary — skip duplicates)
                coords = np.array(node["coords"])
                already = any(
                    np.linalg.norm(v["coords"] - coords) < 0.5
                    for v in result.values()
                )
                if not already:
                    result[key] = {
                        "coords":    coords,
                        "seg_id":    seg_id,
                        "node_type": node_type,
                        "key":       key,
                    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 0b — Compute per-endpoint dome-removal sphere radii from morphometrics
# ═══════════════════════════════════════════════════════════════════════════════

def get_endpoint_sphere_radii(degree1_nodes, features):
    """
    For each degree-1 endpoint, derive a sphere radius for dome removal from
    the local vessel mean radius stored in features.json.

        sphere_radius = clamp(SPHERE_RADIUS_MULTIPLIER × vessel_mean_radius,
                              SPHERE_RADIUS_MIN_MM, SPHERE_RADIUS_MAX_MM)

    Using per-vessel radii avoids over-cutting in thin vessels (e.g. ACA/Acom)
    whose endpoints may be only a few mm apart.  A single global radius of 4 mm
    would cut through the Acom bridge between the two ACA tips (~3.4 mm apart).

    Returns dict: node_key → sphere_radius_mm
    """
    radii = {}
    for key, info in degree1_nodes.items():
        seg_id = info["seg_id"]
        vessel_radius = 1.5   # mm — fallback if features are missing

        # features keys may be ints or strings depending on the JSON loader
        seg_feats = features.get(seg_id, features.get(str(seg_id), {}))
        for feat_type, feat_list in seg_feats.items():
            if not isinstance(feat_list, list):
                continue
            for entry in feat_list:
                if isinstance(entry, dict) and "radius" in entry:
                    r = entry["radius"]
                    if isinstance(r, dict) and "mean" in r:
                        vessel_radius = float(r["mean"])
                        break   # first entry per segment is sufficient

        sphere_r = max(SPHERE_RADIUS_MIN_MM,
                       min(SPHERE_RADIUS_MAX_MM,
                           SPHERE_RADIUS_MULTIPLIER * vessel_radius))
        radii[key] = sphere_r
        # debug: print(f"    {key}: vessel_r={vessel_radius:.2f}  sphere_r={sphere_r:.2f}")

    return radii


def compute_endpoint_normals(degree1_nodes, graph_mesh):
    """
    For each degree-1 endpoint, compute an *outward* unit normal
    (direction pointing away from the vessel body, toward the terminal dome).

    This is used to define the cutting plane that removes the closed dome
    and exposes the vessel lumen.

    Returns dict: key → (endpoint_coords  np.array,
                         outward_normal    np.array)
    """
    # Build full graph adjacency (all labels together)
    adj = {}
    for ci in range(graph_mesh.GetNumberOfCells()):
        cell = graph_mesh.GetCell(ci)
        pt_ids = cell.GetPointIds()
        for j in range(pt_ids.GetNumberOfIds() - 1):
            p0, p1 = pt_ids.GetId(j), pt_ids.GetId(j + 1)
            adj.setdefault(p0, set()).add(p1)
            adj.setdefault(p1, set()).add(p0)

    pts = graph_mesh.GetPoints()
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(graph_mesh)
    locator.BuildLocator()

    result = {}
    for key, info in degree1_nodes.items():
        ep_coord = info["coords"]
        ep_id = locator.FindClosestPoint(ep_coord.tolist())
        ep_xyz = np.array(pts.GetPoint(ep_id))

        neighbors = list(adj.get(ep_id, []))
        if neighbors:
            nb_xyz = np.array(pts.GetPoint(neighbors[0]))
            # outward = away from vessel interior (endpoint → away from body)
            outward = ep_xyz - nb_xyz
        else:
            outward = np.array([0.0, 0.0, 1.0])

        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-10 else outward
        result[key] = (ep_xyz, outward)

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Cut the closed mesh open at each vessel endpoint
# ═══════════════════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════════════════
# Step 1 — Cut the closed mesh open at each vessel endpoint
# ═══════════════════════════════════════════════════════════════════════════════

def cut_mesh_at_endpoints(mesh, endpoint_normals, sphere_radii=None):
    """
    Remove the terminal dome faces near each vessel endpoint to expose the lumen.

    The surface mesh from the CoW pipeline is *closed* — each vessel end is
    covered by a dome from the marching cubes extraction.  Before CFD we must
    remove those domes to create open inlet/outlet boundaries.

    Strategy (local sphere removal — safer than global clipping for ring
    structures like the CoW):
        For each inlet/outlet endpoint:
        1. Find all triangle faces whose centroid lies within sphere_radius_mm
           of the endpoint.
        2. Among those, keep only faces on the OUTWARD side of the cutting plane
           (i.e. those forming the dome cap, with f(x) >= 0 where
           f(x) = outward · (centroid - endpoint)).
        3. Delete those faces.

    This is surgical — it only removes faces near the endpoint, leaving the
    rest of the (ring-shaped) CoW wall untouched.

    sphere_radii: dict mapping node_key → radius_mm (from get_endpoint_sphere_radii).
                  Any key absent from the dict falls back to SPHERE_RADIUS_MAX_MM.
    """
    if sphere_radii is None:
        sphere_radii = {}
    n_cells = mesh.GetNumberOfCells()
    mesh_pts = mesh.GetPoints()
    n_pts = mesh.GetNumberOfPoints()

    # Pre-compute all face centroids as a NumPy array for fast vectorised ops
    all_pts = np.array([mesh_pts.GetPoint(i) for i in range(n_pts)])
    centroids = np.zeros((n_cells, 3))
    for ci in range(n_cells):
        cell = mesh.GetCell(ci)
        n_verts = cell.GetNumberOfPoints()
        c = np.zeros(3)
        for j in range(n_verts):
            c += all_pts[cell.GetPointId(j)]
        centroids[ci] = c / n_verts

    # Decide which faces to keep (True = keep)
    keep = np.ones(n_cells, dtype=bool)

    for key, (ep_xyz, outward) in endpoint_normals.items():
        sphere_radius_mm = sphere_radii.get(key, SPHERE_RADIUS_MAX_MM)
        # Signed distance of each centroid from the cutting plane
        # (outward normal, plane passes through ep_xyz)
        disp = centroids - ep_xyz              # (N, 3)
        dist_to_ep = np.linalg.norm(disp, axis=1)   # Euclidean distance
        signed_dist = disp @ outward                  # projection on outward

        # Remove faces that are:
        #   (a) within the sphere around the endpoint, AND
        #   (b) on the dome side (signed_dist >= 0)
        to_remove = (dist_to_ep < sphere_radius_mm) & (signed_dist >= 0.0)
        removed = int(to_remove.sum())
        if removed == 0:
            print(f"    WARNING '{key}': no faces removed — check sphere radius "
                  f"or endpoint coords")
        else:
            print(f"    '{key}': removed {removed} dome faces")
        keep[to_remove] = False

    # Build new polydata with surviving faces only
    new_pd = vtk.vtkPolyData()
    new_pd.SetPoints(mesh.GetPoints())

    new_polys = vtk.vtkCellArray()
    for ci in range(n_cells):
        if keep[ci]:
            id_list = vtk.vtkIdList()
            mesh.GetCellPoints(ci, id_list)
            new_polys.InsertNextCell(id_list)

    new_pd.SetPolys(new_polys)

    # Remove unreferenced points, merge coincident points
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(new_pd)
    cleaner.SetTolerance(1e-6)
    cleaner.Update()

    return cleaner.GetOutput()


# ═══════════════════════════════════════════════════════════════════════════════
# Step 3 — Find open boundary loops of the cut mesh
# ═══════════════════════════════════════════════════════════════════════════════

def find_boundary_loops(mesh):
    """
    Find all open boundary edge loops of a triangle surface mesh.

    Uses vtkFeatureEdges to identify boundary edges, then traces each loop
    with a vertex-based walk that:
      - always prefers UNVISITED neighbours (avoids premature termination
        at T-junction vertices that were reached from a different direction)
      - closes the loop as soon as `start` is reachable

    Returns
    -------
    list of list[int]
        Each inner list is an ordered sequence of original mesh point IDs
        forming one closed boundary loop.
    """
    feat = vtk.vtkFeatureEdges()
    feat.SetInputData(mesh)
    feat.BoundaryEdgesOn()
    feat.FeatureEdgesOff()
    feat.ManifoldEdgesOff()
    feat.NonManifoldEdgesOff()
    feat.Update()

    bnd = feat.GetOutput()

    # Build adjacency (local boundary mesh point IDs)
    adj = {}
    for ci in range(bnd.GetNumberOfCells()):
        cell = bnd.GetCell(ci)
        for j in range(cell.GetNumberOfPoints() - 1):
            p0 = cell.GetPointId(j)
            p1 = cell.GetPointId(j + 1)
            adj.setdefault(p0, set()).add(p1)
            adj.setdefault(p1, set()).add(p0)

    # Trace loops
    visited = set()
    loops_local = []
    for start in adj:
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur, prev = start, None

        while True:
            candidates = adj.get(cur, set()) - ({prev} if prev is not None else set())
            if not candidates:
                break
            # Close the loop immediately if start is reachable
            if start in candidates and len(loop) > 2:
                break
            # Prefer unvisited neighbours to avoid stopping at T-junctions
            unvisited = candidates - visited
            if not unvisited:
                break
            nxt = next(iter(unvisited))
            loop.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        if len(loop) >= 3:
            loops_local.append(loop)

    # Map local boundary IDs → original mesh IDs via coordinate lookup
    locator = vtk.vtkPointLocator()
    locator.SetDataSet(mesh)
    locator.BuildLocator()

    bnd_pts = bnd.GetPoints()
    loops_orig = []
    for loop in loops_local:
        orig = [locator.FindClosestPoint(bnd_pts.GetPoint(lid)) for lid in loop]
        loops_orig.append(orig)

    return loops_orig


def loop_centroid(mesh, loop):
    pts = mesh.GetPoints()
    coords = np.array([pts.GetPoint(pid) for pid in loop])
    return coords.mean(axis=0)


def fill_residual_holes(pd):
    """
    Fill any open boundary loops that remain after the main fan-cap pass.

    These are small gaps left by the sphere-cut dome removal (e.g. a tiny
    triangular notch in the vessel wall near a bifurcation).  We detect them
    by edge-counting directly on the polydata, trace each residual loop, and
    fill with a fan from the loop centroid — same method as the main caps,
    so no FillHolesFilter duplicates.

    New fill cells get ModelFaceID = 1 (wall).
    """
    # -- count edge occurrences directly on mesh cells --
    edge_count = {}
    id_list = vtk.vtkIdList()
    for ci in range(pd.GetNumberOfCells()):
        pd.GetCellPoints(ci, id_list)
        n = id_list.GetNumberOfIds()
        for j in range(n):
            p0 = id_list.GetId(j)
            p1 = id_list.GetId((j + 1) % n)
            e = (min(p0, p1), max(p0, p1))
            edge_count[e] = edge_count.get(e, 0) + 1

    bnd_adj = {}
    for (p0, p1), c in edge_count.items():
        if c == 1:
            bnd_adj.setdefault(p0, set()).add(p1)
            bnd_adj.setdefault(p1, set()).add(p0)

    if not bnd_adj:
        return pd   # already fully closed

    # -- trace residual boundary loops --
    visited = set()
    residual_loops = []
    for start in sorted(bnd_adj.keys()):
        if start in visited:
            continue
        loop = [start]
        visited.add(start)
        cur = next(iter(bnd_adj[start]))
        while cur != start and cur not in visited:
            loop.append(cur)
            visited.add(cur)
            candidates = bnd_adj[cur] - visited
            if not candidates:
                break
            cur = next(iter(candidates))
        if len(loop) >= 3:
            residual_loops.append(loop)

    if not residual_loops:
        return pd

    # -- fill each residual loop with a centroid fan --
    old_pts = pd.GetPoints()
    old_face_arr = pd.GetCellData().GetArray("ModelFaceID")

    new_pts = vtk.vtkPoints()
    for i in range(old_pts.GetNumberOfPoints()):
        new_pts.InsertNextPoint(old_pts.GetPoint(i))

    new_cells = vtk.vtkCellArray()
    new_face_arr = vtk.vtkIntArray()
    new_face_arr.SetName("ModelFaceID")

    id_list2 = vtk.vtkIdList()
    for ci in range(pd.GetNumberOfCells()):
        pd.GetCellPoints(ci, id_list2)
        new_cells.InsertNextCell(id_list2)
        fid = int(old_face_arr.GetValue(ci)) if old_face_arr else 1
        new_face_arr.InsertNextValue(fid)

    for loop in residual_loops:
        coords = np.array([old_pts.GetPoint(pid) for pid in loop])
        centroid = coords.mean(axis=0)
        c_id = new_pts.InsertNextPoint(centroid.tolist())
        n = len(loop)
        for i in range(n):
            tri = vtk.vtkIdList()
            tri.SetNumberOfIds(3)
            tri.SetId(0, c_id)
            tri.SetId(1, loop[i])
            tri.SetId(2, loop[(i + 1) % n])
            new_cells.InsertNextCell(tri)
            new_face_arr.InsertNextValue(1)
        print(f"    Filled residual wall hole: {n} edges")

    new_pd = vtk.vtkPolyData()
    new_pd.SetPoints(new_pts)
    new_pd.SetPolys(new_cells)
    new_pd.GetCellData().AddArray(new_face_arr)
    return new_pd


    pts = mesh.GetPoints()
    coords = np.array([pts.GetPoint(pid) for pid in loop])
    return coords.mean(axis=0)


def match_loops_to_nodes(mesh, loops, endpoint_normals, degree1_nodes):
    """
    Match each boundary loop to the closest degree-1 node (inlet/outlet).

    When multiple loops match the same endpoint (caused by over-cutting that
    fragments the boundary), only the loop closest to that endpoint is kept;
    the rest are discarded as artefacts.

    Returns
    -------
    list of (loop, name) — deduplicated, one entry per unique endpoint.
    """
    # Score every loop against every endpoint
    scored = []
    for i, loop in enumerate(loops):
        centroid = loop_centroid(mesh, loop)
        best_name, best_dist = None, float("inf")
        for key, (ep_xyz, _) in endpoint_normals.items():
            dist = np.linalg.norm(centroid - ep_xyz)
            if dist < best_dist:
                best_dist = dist
                best_name = key
        if best_name is None:
            for key, info in degree1_nodes.items():
                dist = np.linalg.norm(centroid - info["coords"])
                if dist < best_dist:
                    best_dist = dist
                    best_name = key
        scored.append((i, loop, best_name or f"opening_{i}", best_dist, centroid))

    # Deduplicate: for each endpoint key, keep only the loop with smallest dist
    best_per_key = {}
    for i, loop, name, dist, centroid in scored:
        if name not in best_per_key or dist < best_per_key[name][3]:
            best_per_key[name] = (i, loop, name, dist, centroid)

    discarded = len(loops) - len(best_per_key)
    if discarded:
        print(f"    Discarded {discarded} artefact loop(s) "
              f"(over-cut fragments — reduce sphere_radius_mm if many)")

    result_loops = []
    result_names = []
    for i, loop, name, dist, centroid in best_per_key.values():
        result_loops.append(loop)
        result_names.append(name)
        print(f"    Loop → '{name}'  centroid={centroid.round(1)}  dist={dist:.1f} mm")

    return result_loops, result_names


def add_triangulated_caps(mesh, loops, loop_names):
    """
    Add fan-triangulated cap patches to close all open boundary loops.

    Unlike a single n-gon polygon per hole, fan triangulation from the loop
    centroid produces a fully triangulated surface:
        - 0 non-manifold edges  (each edge shared by exactly 2 triangles)
        - 0 free edges          (all boundaries closed)
        - 1 connected region    (as long as the wall is already connected)

    Each cap adds (len(loop)) triangles of the form:
        (centroid_id, loop[i], loop[i+1 % n])

    Face ID conventions
    -------------------
    1       = wall (all original triangle cells)
    2, 3, … = caps (one fan patch per open boundary, named cap_<loop_name>)

    Returns
    -------
    capped_pd : vtkPolyData
    face_info : list of dict  [{"id", "name", "type"}, …]
    """
    # Build a new point array: originals + one centroid per cap
    new_points = vtk.vtkPoints()
    old_pts = mesh.GetPoints()
    for i in range(old_pts.GetNumberOfPoints()):
        new_points.InsertNextPoint(old_pts.GetPoint(i))

    all_cells  = vtk.vtkCellArray()
    face_id_arr = vtk.vtkIntArray()
    face_id_arr.SetName("ModelFaceID")

    # --- original wall triangles (face ID 1) ---
    orig_polys = mesh.GetPolys()
    orig_polys.InitTraversal()
    id_list = vtk.vtkIdList()
    while orig_polys.GetNextCell(id_list):
        all_cells.InsertNextCell(id_list)
        face_id_arr.InsertNextValue(1)

    face_info = [{"id": 1, "name": "wall", "type": "wall"}]

    # --- cap fan patches ---
    for cap_idx, (loop, name) in enumerate(zip(loops, loop_names)):
        face_id = cap_idx + 2

        # Centroid of this loop → new point
        coords = np.array([old_pts.GetPoint(pid) for pid in loop])
        centroid = coords.mean(axis=0)
        centroid_id = new_points.InsertNextPoint(centroid.tolist())

        # Fan triangles: (centroid, loop[i], loop[i+1])
        n = len(loop)
        for i in range(n):
            p0 = loop[i]
            p1 = loop[(i + 1) % n]
            tri = vtk.vtkIdList()
            tri.SetNumberOfIds(3)
            tri.SetId(0, centroid_id)
            tri.SetId(1, p0)
            tri.SetId(2, p1)
            all_cells.InsertNextCell(tri)
            face_id_arr.InsertNextValue(face_id)

        face_info.append({"id": face_id, "name": f"cap_{name}", "type": "cap"})
        print(f"    cap '{name}': {n} triangles  (fan from centroid {centroid.round(1)})")

    new_pd = vtk.vtkPolyData()
    new_pd.SetPoints(new_points)
    new_pd.SetPolys(all_cells)
    new_pd.GetCellData().AddArray(face_id_arr)

    # Merge coincident/near-coincident vertices, remove degenerate cells.
    # Tolerance 1e-6 (relative) ≈ sub-nanometre for a 30 mm structure —
    # safe against floating-point noise while never merging distinct mesh points.
    cleaner = vtk.vtkCleanPolyData()
    cleaner.SetInputData(new_pd)
    cleaner.SetTolerance(1e-6)
    cleaner.Update()
    cleaned = cleaner.GetOutput()

    # Verify no free edges remain; fill any residual wall holes via our own
    # centroid-fan method (NOT vtkFillHolesFilter — it creates duplicate facets).
    feat_check = vtk.vtkFeatureEdges()
    feat_check.SetInputData(cleaned)
    feat_check.BoundaryEdgesOn()
    feat_check.FeatureEdgesOff()
    feat_check.ManifoldEdgesOff()
    feat_check.NonManifoldEdgesOff()
    feat_check.Update()
    n_open = feat_check.GetOutput().GetNumberOfCells()
    if n_open > 0:
        print(f"    NOTE: {n_open} free edge(s) remain — "
              f"filling residual wall holes")
        cleaned = fill_residual_holes(cleaned)
    else:
        print(f"    Surface is fully closed (0 free edges)")

    return cleaned, face_info


# ═══════════════════════════════════════════════════════════════════════════════
# Step 4 — Extract ordered centerline paths from the graph VTP
# ═══════════════════════════════════════════════════════════════════════════════

def extract_paths_from_graph(graph_mesh):
    """
    Extract per-label ordered point sequences from the centerline graph.

    The graph VTP stores vessel segments as polylines with a "labels" cell
    data array (values 1–12, matching LABEL_MAP).

    Returns
    -------
    dict: label (int) → list of [x, y, z] coordinates (ordered along vessel)
    """
    pts = graph_mesh.GetPoints()
    labels_arr = graph_mesh.GetCellData().GetArray("labels")

    # Build per-label adjacency from graph edges
    label_adj = {}
    for ci in range(graph_mesh.GetNumberOfCells()):
        label = int(labels_arr.GetValue(ci))
        cell = graph_mesh.GetCell(ci)
        pt_ids = cell.GetPointIds()
        adj = label_adj.setdefault(label, {})
        for j in range(pt_ids.GetNumberOfIds() - 1):
            p0, p1 = pt_ids.GetId(j), pt_ids.GetId(j + 1)
            adj.setdefault(p0, set()).add(p1)
            adj.setdefault(p1, set()).add(p0)

    paths = {}
    for label, adj in label_adj.items():
        # Start from a degree-1 endpoint; fall back to any node for loops
        endpoints = [p for p, nbrs in adj.items() if len(nbrs) == 1]
        cur = endpoints[0] if endpoints else next(iter(adj))

        ordered = [cur]
        visited = {cur}
        prev = None
        while True:
            nbrs = adj.get(cur, set()) - ({prev} if prev is not None else set())
            unvisited = nbrs - visited
            if not unvisited:
                break
            nxt = next(iter(unvisited))
            ordered.append(nxt)
            visited.add(nxt)
            prev, cur = cur, nxt

        paths[label] = [list(pts.GetPoint(pid)) for pid in ordered]

    return paths


def compute_tangents_and_rotations(coords):
    """
    Compute unit tangent and rotation (normal) vectors for each path point.

    Tangent  : central difference (forward/backward at endpoints).
    Rotation : vector perpendicular to tangent, via Gram-Schmidt with [0,1,0].
    """
    pts = np.array(coords)
    n = len(pts)
    tangents = []

    for i in range(n):
        if i == 0:
            t = pts[1] - pts[0]
        elif i == n - 1:
            t = pts[-1] - pts[-2]
        else:
            t = pts[i + 1] - pts[i - 1]
        norm = np.linalg.norm(t)
        tangents.append(t / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0]))

    ref = np.array([0.0, 1.0, 0.0])
    rotations = []
    for t in tangents:
        r = ref - np.dot(ref, t) * t
        norm = np.linalg.norm(r)
        if norm < 1e-10:
            ref = np.array([1.0, 0.0, 0.0])
            r = ref - np.dot(ref, t) * t
            norm = np.linalg.norm(r)
        rotations.append(r / norm if norm > 1e-10 else np.array([0.0, 1.0, 0.0]))

    return tangents, rotations


# ═══════════════════════════════════════════════════════════════════════════════
# Step 5 — Write SimVascular file formats
# ═══════════════════════════════════════════════════════════════════════════════

def write_proj_file(proj_dir):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<simvascular_project version="1.0"/>']
    write_text(lines, os.path.join(proj_dir, "simvascular.proj"))


def write_image_info(img_dir, modality):
    """
    Write image_information.xml.
    Points to no image file — the GUI will show an empty Images slot,
    which is fine when working purely from the surface mesh.
    """
    ts = int(time.time())
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<ImageObjectInformation creation_time="{ts}" modification_time="{ts}" version="1.0">',
        '    <timestep id="0">',
        '        <created_with_simvascular_version>2025.12</created_with_simvascular_version>',
        '        <path></path>',
        '        <image_file_name></image_file_name>',
        '        <image_header_file_name></image_header_file_name>',
        '        <image_name></image_name>',
        '        <data_is_local_copy>false</data_is_local_copy>',
        '        <scale_factor>1.0</scale_factor>',
        '    </timestep>',
        '</ImageObjectInformation>',
    ]
    write_text(lines, os.path.join(img_dir, "image_information.xml"))


def write_model_mdl(models_dir, model_name, face_info):
    """
    Write the SimVascular model descriptor (.mdl).

    Format quirk: SV uses two consecutive XML root elements in this file
    (<format/> then <model>), so we write it as plain text.
    """
    cap_count = 0
    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<format version="1.0" />',
        '<model type="PolyData">',
        '    <timestep id="0">',
        '        <model_element type="PolyData" num_sampling="0">',
        '            <segmentations>',
    ]

    # List cap face names as "segmentations" (SV convention for imported models)
    for fi in face_info:
        if fi["type"] == "cap":
            lines.append(f'                <seg name="{fi["name"]}" />')

    lines += ['            </segmentations>', '            <faces>']

    for fi in face_info:
        if fi["type"] == "wall":
            c1, c2, c3 = WALL_COLOR
        else:
            c1, c2, c3 = CAP_COLORS[cap_count % len(CAP_COLORS)]
            cap_count += 1
        lines.append(
            f'                <face id="{fi["id"]}" name="{fi["name"]}" '
            f'type="{fi["type"]}" visible="true" opacity="1" '
            f'color1="{c1}" color2="{c2}" color3="{c3}" />'
        )

    lines += [
        '            </faces>',
        '            <blend_radii />',
        '            <blend_param blend_iters="2" sub_blend_iters="3" '
        'cstr_smooth_iters="2" lap_smooth_iters="50" '
        'subdivision_iters="1" decimation="0.01" />',
        '        </model_element>',
        '    </timestep>',
        '</model>',
    ]

    write_text(lines, os.path.join(models_dir, f"{model_name}.mdl"))


def write_path_file(paths_dir, label, coords, path_id):
    """
    Write a SimVascular path file (.pth) for one vessel segment.

    The graph points are used as both control points and path_points.
    SV will interpolate/resample them internally.
    """
    name = LABEL_MAP.get(label, f"seg_{label}")
    pts = np.array(coords)
    tangents, rotations = compute_tangents_and_rotations(coords)

    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1) if len(pts) > 1 else [1.0]
    spacing = float(np.mean(dists))

    lines = [
        '<?xml version="1.0" encoding="UTF-8" ?>',
        '<format version="1.0" />',
        f'<path id="{path_id}" method="2" calculation_number="0" '
        f'spacing="{spacing:.6f}" reslice_size="5">',
        '    <timestep id="0">',
        f'        <path_element id="0" method="2" calculation_number="0" '
        f'spacing="{spacing:.6f}">',
        '            <control_points>',
    ]

    for i, pt in enumerate(coords):
        lines.append(f'                <point id="{i}" '
                     f'x="{pt[0]:.6f}" y="{pt[1]:.6f}" z="{pt[2]:.6f}" />')

    lines += ['            </control_points>', '            <path_points>']

    for i, (pt, t, r) in enumerate(zip(coords, tangents, rotations)):
        lines += [
            f'            <path_point id="{i}">',
            f'                <pos x="{pt[0]:.6f}" y="{pt[1]:.6f}" z="{pt[2]:.6f}" />',
            f'                <tangent x="{t[0]:.9f}" y="{t[1]:.9f}" z="{t[2]:.9f}" />',
            f'                <rotation x="{r[0]:.9f}" y="{r[1]:.9f}" z="{r[2]:.9f}" />',
            '            </path_point>',
        ]

    lines += [
        '            </path_points>',
        '        </path_element>',
        '    </timestep>',
        '</path>',
    ]

    write_text(lines, os.path.join(paths_dir, f"{name}.pth"))
    return name


def write_sim_job(sims_dir, model_name, mesh_name, face_info, features):
    """
    Write a SimVascular simulation job file (.sjb).

    Boundary conditions:
    - Inlets  → Prescribed Velocities (parabolic, steady)
    - Outlets → Resistance (simple Windkessel, value from DEFAULT_RESISTANCE)

    All values are estimates — review and adjust in the GUI before solving.
    """
    job_name = f"{model_name}_job"

    # Classify caps into inlets / outlets based on name keyword
    inlets, outlets = [], []
    for fi in face_info:
        if fi["type"] != "cap":
            continue
        n = fi["name"]   # e.g. "cap_seg4_ica_start", "cap_seg2_r_pca_end"
        if any(kw in n for kw in INLET_KEYWORDS):
            # Map seg ID to artery name for flow lookup
            n_up = n.upper()
            if "BA" in n_up:
                artery_key = "BA"
            elif "SEG4" in n_up or ("ICA" in n_up and "SEG4" in n_up):
                artery_key = "R-ICA"
            elif "SEG6" in n_up or ("ICA" in n_up and "SEG6" in n_up):
                artery_key = "L-ICA"
            else:
                artery_key = "UNKNOWN"
            inlets.append((fi["name"], artery_key))
        else:
            outlets.append((fi["name"],))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<mitk_job model_name="{model_name}" mesh_name="{mesh_name}" '
        f'status="No Data Files" version="1.0">',
        '    <job>',
        '        <basic_props>',
        f'            <prop key="Fluid Density" value="{FLUID_DENSITY}"/>',
        f'            <prop key="Fluid Viscosity" value="{FLUID_VISCOSITY}"/>',
        '            <prop key="Initial Pressure" value="0"/>',
        '            <prop key="Initial Velocities" value="0.0001 0.0001 0.0001"/>',
        '        </basic_props>',
        '        <cap_props>',
    ]

    # Inlet BCs — parabolic prescribed velocity (steady)
    for cap_name, artery_key in inlets:
        flow = INFLOW_MM3_S.get(artery_key, DEFAULT_INFLOW)
        lines += [
            f'            <cap name="{cap_name}">',
            '                <prop key="BC Type" value="Prescribed Velocities"/>',
            '                <prop key="Analytic Shape" value="parabolic"/>',
            '                <prop key="Flip Normal" value="False"/>',
            f'                <prop key="Flow Rate" value="2 2\n0.0 {flow:.1f}\n1.0 {flow:.1f}"/>',
            '                <prop key="Fourier Modes" value="1"/>',
            '                <prop key="Period" value="1.0"/>',
            '                <prop key="Point Number" value="2"/>',
            f'            </cap>',
        ]

    # Outlet BCs — resistance
    for (cap_name,) in outlets:
        lines += [
            f'            <cap name="{cap_name}">',
            '                <prop key="BC Type" value="Resistance"/>',
            '                <prop key="Pressure" value="0"/>',
            f'                <prop key="Values" value="{DEFAULT_RESISTANCE}"/>',
            '            </cap>',
        ]

    lines += [
        '        </cap_props>',
        '        <wall_props>',
        '            <prop key="Type" value="rigid"/>',
        '        </wall_props>',
        '        <var_props/>',
        '        <solver_props>',
        '            <prop key="Backflow stabilization coefficient" value="0.2"/>',
        '            <prop key="Number of Timesteps" value="100"/>',
        '            <prop key="Time Step Size" value="0.001"/>',
        '            <prop key="Max iterations" value="10"/>',
        '            <prop key="Min iterations" value="3"/>',
        '            <prop key="Tolerance" value="1e-4"/>',
        '            <prop key="Absolute tolerance" value="1e-4"/>',
        '            <prop key="Solver" value="NS"/>',
        '            <prop key="Krylov space dimension" value="200"/>',
        '            <prop key="NS CG max iterations" value="300"/>',
        '            <prop key="NS CG tolerance" value="1e-3"/>',
        '            <prop key="NS GM max iterations" value="10"/>',
        '            <prop key="NS GM tolerance" value="1e-3"/>',
        '            <prop key="Spectral radius of infinite time step" value="0.5"/>',
        '            <prop key="Save results to VTK format" value="true"/>',
        '            <prop key="Increment in saving VTK files" value="10"/>',
        '            <prop key="Increment in saving restart files" value="10"/>',
        '            <prop key="Start saving after time step" value="1"/>',
        '        </solver_props>',
        '        <run_props/>',
        '    </job>',
        '</mitk_job>',
    ]

    write_text(lines, os.path.join(sims_dir, f"{job_name}.sjb"))


# ═══════════════════════════════════════════════════════════════════════════════
# Main pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a SimVascular project from CoW centerline data."
    )
    parser.add_argument(
        "--modality", choices=["ct", "mr"], default="ct",
        help="Image modality (ct or mr). Both come from patient p025. Default: ct"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    mod = args.modality

    # ── Resolve file paths ─────────────────────────────────────────────────────
    mesh_file    = os.path.join(DATA_DIR, f"mesh_{mod}_025.vtp")
    graph_file   = os.path.join(DATA_DIR, f"graph_{mod}_025.vtp")
    node_file    = os.path.join(DATA_DIR, f"node_{mod}_025.json")
    feat_file    = os.path.join(DATA_DIR, f"features_{mod}_025.json")
    # MR variant file has a different naming convention (variants vs variant)
    variant_file = os.path.join(DATA_DIR,
                                f"variants_{mod}_025.json"
                                if mod == "mr"
                                else f"variant_{mod}_025.json")

    model_name = f"{PATIENT_ID}_{mod}"
    mesh_name  = f"{PATIENT_ID}_{mod}_mesh"
    proj_dir   = os.path.join(OUTPUT_BASE, model_name)

    print(f"\n{'═'*60}")
    print(f"  CoW → SimVascular  |  patient: {PATIENT_ID}  |  modality: {mod.upper()}")
    print(f"{'═'*60}")
    print(f"\n  Data dir   : {DATA_DIR}")
    print(f"  Project dir: {proj_dir}\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    print("── 1. Loading data ───────────────────────────────────────────────────")
    for f in [mesh_file, graph_file, node_file, feat_file]:
        if not os.path.exists(f):
            print(f"  ERROR: file not found: {f}")
            sys.exit(1)

    mesh  = read_vtp(mesh_file)
    graph = read_vtp(graph_file)

    with open(node_file)    as f: nodes    = json.load(f)
    with open(feat_file)    as f: features = json.load(f)
    with open(variant_file) as f: variants = json.load(f)

    print(f"  Surface mesh  : {mesh.GetNumberOfPoints():,} pts, "
          f"{mesh.GetNumberOfCells():,} triangles")
    print(f"  Centerline    : {graph.GetNumberOfPoints():,} pts, "
          f"{graph.GetNumberOfCells():,} edges")

    # Print anatomical variant summary
    print("\n  CoW variant (which segments are present):")
    for region, segs in variants.items():
        present = [k for k, v in segs.items() if v]
        absent  = [k for k, v in segs.items() if not v]
        print(f"    {region:10s} → present: {present}   absent: {absent}")

    # ── Collect degree-1 nodes — unique keys to avoid collision ──────────────
    print("\n── 2. Identifying inlets and outlets ─────────────────────────────────")
    degree1_nodes = collect_degree1_nodes(nodes)

    for key, info in sorted(degree1_nodes.items()):
        kind = "INLET " if any(k in key for k in INLET_KEYWORDS) else "OUTLET"
        print(f"  [{kind}] {key:<35s}  {info['coords'].round(1)}")

    # ── Compute outward normals at each endpoint for cutting planes ───────────
    print("\n── 3. Computing cutting planes at vessel endpoints ───────────────────")
    endpoint_normals = compute_endpoint_normals(degree1_nodes, graph)
    for key, (ep_xyz, outward) in sorted(endpoint_normals.items()):
        print(f"  {key:<35s}  outward={outward.round(3)}")

    # ── Determine per-endpoint sphere radii from morphometrics ────────────────
    print("\n── 3b. Per-endpoint dome-removal sphere radii ────────────────────────")
    sphere_radii = get_endpoint_sphere_radii(degree1_nodes, features)
    for key, r in sorted(sphere_radii.items()):
        print(f"  {key:<35s}  sphere_r={r:.2f} mm")

    # ── Cut the closed mesh open at each endpoint ─────────────────────────────
    print("\n── 4. Cutting closed mesh at vessel terminations ─────────────────────")
    print(f"  Input : {mesh.GetNumberOfPoints():,} pts (closed surface)")
    open_mesh = cut_mesh_at_endpoints(mesh, endpoint_normals, sphere_radii)
    print(f"  Output: {open_mesh.GetNumberOfPoints():,} pts (open surface)")

    # Eliminate isolated dome-cut fragments before loop-finding.
    # Near bifurcations the sphere can leave small disconnected patches;
    # keeping only the largest connected component removes them cleanly.
    conn_open = vtk.vtkPolyDataConnectivityFilter()
    conn_open.SetInputData(open_mesh)
    conn_open.SetExtractionModeToAllRegions()
    conn_open.Update()
    n_open_regions = conn_open.GetNumberOfExtractedRegions()
    if n_open_regions > 1:
        print(f"  NOTE: {n_open_regions} regions in open mesh — "
              f"keeping largest (discarding bifurcation dome fragments)")
        conn_open.SetExtractionModeToLargestRegion()
        conn_open.Update()
        open_mesh = conn_open.GetOutput()
        # Re-clean after extraction to remove unreferenced points
        cl2 = vtk.vtkCleanPolyData()
        cl2.SetInputData(open_mesh)
        cl2.SetTolerance(1e-6)
        cl2.Update()
        open_mesh = cl2.GetOutput()
    else:
        print(f"  Connectivity: {n_open_regions} region (OK)")

    # ── Find boundary loops of the now-open mesh ──────────────────────────────
    print("\n── 5. Finding open boundary loops ────────────────────────────────────")
    loops = find_boundary_loops(open_mesh)
    print(f"  Found {len(loops)} boundary loop(s)")
    if not loops:
        print("  WARNING: no loops found — verify that endpoint coords are inside")
        print("           the mesh bounds and increase cut_offset_mm if needed.")

    loop_names = (match_loops_to_nodes(open_mesh, loops, endpoint_normals, degree1_nodes)
                  if loops else ([], []))
    loops, loop_names = loop_names  # unpack (deduplicated loops, names)

    # ── Cap the open boundaries with fan-triangulated patches ─────────────────
    print("\n── 6. Capping open boundaries (fan triangulation) ────────────────────")
    capped_mesh, face_info = add_triangulated_caps(open_mesh, loops, loop_names)
    print(f"  Capped mesh : {capped_mesh.GetNumberOfCells():,} cells total "
          f"(1 wall + {len(loops)} cap{'s' if len(loops) != 1 else ''})")

    # Connectivity sanity-check — TetGen needs exactly 1 region
    conn_filter = vtk.vtkPolyDataConnectivityFilter()
    conn_filter.SetInputData(capped_mesh)
    conn_filter.SetExtractionModeToAllRegions()
    conn_filter.Update()
    n_regions = conn_filter.GetNumberOfExtractedRegions()
    if n_regions == 1:
        print(f"  Connectivity: {n_regions} region (OK)")
    else:
        print(f"  WARNING: {n_regions} disconnected regions — keeping largest region only")
        conn_filter.SetExtractionModeToLargestRegion()
        conn_filter.Update()
        # Re-attach ModelFaceID from capped_mesh by transferring cell data
        capped_mesh = conn_filter.GetOutput()
        if not capped_mesh.GetCellData().GetArray("ModelFaceID"):
            print("  WARNING: ModelFaceID array lost after region extraction — "
                  "reduce SPHERE_RADIUS_MAX_MM and re-run")

    print("  Face map:")
    for fi in face_info:
        print(f"    id={fi['id']:2d}  type={fi['type']:<4s}  name={fi['name']}")

    # ── Create project directory structure ────────────────────────────────────
    print("\n── 7. Creating project structure ─────────────────────────────────────")
    make_dirs(proj_dir)
    write_proj_file(proj_dir)
    write_image_info(os.path.join(proj_dir, "Images"), mod)

    # ── Write model files ──────────────────────────────────────────────────────
    print("\n── 8. Writing model ──────────────────────────────────────────────────")
    models_dir = os.path.join(proj_dir, "Models")
    write_vtp(capped_mesh, os.path.join(models_dir, f"{model_name}.vtp"))
    write_model_mdl(models_dir, model_name, face_info)

    # ── Extract and write centerline paths ────────────────────────────────────
    print("\n── 9. Writing centerline paths ───────────────────────────────────────")
    paths = extract_paths_from_graph(graph)
    paths_dir = os.path.join(proj_dir, "Paths")
    for path_id, (label, coords) in enumerate(sorted(paths.items()), start=1):
        seg_name = write_path_file(paths_dir, label, coords, path_id)
        print(f"    {seg_name:<10s} ({len(coords)} pts)")

    # ── Write simulation job ───────────────────────────────────────────────────
    print("\n── 10. Writing simulation job ────────────────────────────────────────")
    sims_dir = os.path.join(proj_dir, "Simulations")
    write_sim_job(sims_dir, model_name, mesh_name, face_info, features)

    # ── Final instructions ─────────────────────────────────────────────────────
    inlets_list  = [fi["name"] for fi in face_info
                    if fi["type"] == "cap" and any(k in fi["name"] for k in INLET_KEYWORDS)]
    outlets_list = [fi["name"] for fi in face_info
                    if fi["type"] == "cap" and not any(k in fi["name"] for k in INLET_KEYWORDS)]

    print(f"\n{'═'*60}")
    print("  PROJECT READY")
    print(f"{'═'*60}")
    print(f"""
  Project folder
  ──────────────
  {proj_dir}

  Inlets  ({len(inlets_list)}): {', '.join(inlets_list) or '(none — check loop matching)'}
  Outlets ({len(outlets_list)}): {', '.join(outlets_list) or '(none — check loop matching)'}

  How to open in SimVascular GUI
  ──────────────────────────────
  1.  Launch SimVascular
  2.  File > Open Project
      → select: {proj_dir}

  3.  MODELS tab
      • The model '{model_name}' will appear in the tree
      • Right-click > Show Model to visualise
      • Verify face colours: wall (purple), caps (coloured)
      • If any cap is wrong: right-click face > Change Type

  4.  MESH tab (TetGen)
      • Right-click Meshes > New Mesh
      • Name it: {mesh_name}
      • Set Global Max Edge Size ≈ 0.3–0.5 mm for CoW
      • Click 'Run Mesher'

  5.  SIMULATIONS tab
      • Right-click Simulations > Open Job → select {model_name}_job.sjb
      • Review boundary conditions:
          Inlets  → Prescribed Velocities (steady parabolic, adjust flow rate)
          Outlets → Resistance (adjust value per vessel size)
      • Fluid: density={FLUID_DENSITY} g/cm³, viscosity={FLUID_VISCOSITY} Poise
      • Click 'Create Data Files' then 'Run Simulation'

  NOTE: The boundary condition values in the .sjb are estimates from
  published CoW literature. You must review and calibrate them before
  running a simulation you intend to publish or use clinically.
""")


if __name__ == "__main__":
    main()
