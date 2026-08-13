"""Export the growth region (the volume between the lower and upper limit heightfields)
as a Wavefront .obj mesh, so a renderer can show *where atoms are allowed to grow* next to
the structure. The mesh is a closed prism: the (Fourier-roughened) top surface, the flat
bottom, and the four side walls. Downsampled by ``stride`` -- the limits grid is 250x250 but
the Fourier surface has only ~6 modes, so a coarser grid renders identically."""
import numpy as np


def write_growth_region_mesh(upper_lim, lower_lim, dx, dy, path, stride: int = 3) -> None:
    U = np.asarray(upper_lim)[::stride, ::stride]
    L = np.asarray(lower_lim)[::stride, ::stride]
    nx, ny = U.shape
    xs = np.arange(nx) * dx * stride
    ys = np.arange(ny) * dy * stride

    verts = []
    for i in range(nx):                     # top vertices first, then bottom
        for j in range(ny):
            verts.append((xs[i], ys[j], U[i, j]))
    for i in range(nx):
        for j in range(ny):
            verts.append((xs[i], ys[j], L[i, j]))

    def t(i, j):  # 1-indexed obj vertex id, top surface
        return i * ny + j + 1

    def b(i, j):  # bottom surface
        return nx * ny + i * ny + j + 1

    faces = []
    for i in range(nx - 1):                 # top surface
        for j in range(ny - 1):
            faces += [(t(i, j), t(i + 1, j), t(i + 1, j + 1)),
                      (t(i, j), t(i + 1, j + 1), t(i, j + 1))]
    for i in range(nx - 1):                 # bottom surface (reversed winding)
        for j in range(ny - 1):
            faces += [(b(i, j), b(i + 1, j + 1), b(i + 1, j)),
                      (b(i, j), b(i, j + 1), b(i + 1, j + 1))]

    # side walls around the perimeter (top edge -> bottom edge)
    perim = ([(0, j) for j in range(ny)] + [(i, ny - 1) for i in range(1, nx)] +
             [(nx - 1, j) for j in range(ny - 2, -1, -1)] + [(i, 0) for i in range(nx - 2, 0, -1)])
    perim.append(perim[0])
    for k in range(len(perim) - 1):
        (i0, j0), (i1, j1) = perim[k], perim[k + 1]
        faces += [(t(i0, j0), t(i1, j1), b(i1, j1)),
                  (t(i0, j0), b(i1, j1), b(i0, j0))]

    with open(str(path), "w") as f:
        f.write("# growth region (allowed area to grow): rough top + flat bottom + walls\n")
        for vx, vy, vz in verts:
            f.write(f"v {vx:.4f} {vy:.4f} {vz:.4f}\n")
        for a, bb, c in faces:
            f.write(f"f {a} {bb} {c}\n")
