#!/usr/bin/env python3
"""Serve sliding-window results in Neuroglancer (tissue + probability + binary).

Loads the three .npy outputs memory-mapped (no full load into RAM), so it works
on the big regions. Prints a URL; keep the process running while you view.

Install once:   pip install neuroglancer

Run (on the cluster where the outputs live, or on your laptop after copying):
    python view_neuroglancer.py --prefix /orcd/.../sliding_infer_out_slice037/region1

Then open the printed URL in a browser. If you ran it on the cluster, first make
an SSH tunnel from your laptop:
    ssh -L 9999:localhost:9999 sparsh@<login-host>
and replace the host in the URL with localhost:9999 (use --port 9999 below).
"""
import argparse
import numpy as np
import neuroglancer

ap = argparse.ArgumentParser()
ap.add_argument("--prefix", required=True, help="e.g. .../region1  (expects _image/_probability/_binary .npy)")
ap.add_argument("--port", type=int, default=9999)
ap.add_argument("--bind", default="127.0.0.1", help="127.0.0.1 for tunnel; 0.0.0.0 to expose")
ap.add_argument("--voxel_um", nargs=3, type=float, default=[1.434, 1.16, 1.0],
                metavar=("Z", "Y", "X"), help="slice037/049 level-0 voxel size (µm)")
args = ap.parse_args()

img = np.load(f"{args.prefix}_image.npy", mmap_mode="r")
prob = np.load(f"{args.prefix}_probability.npy", mmap_mode="r")
binr = np.load(f"{args.prefix}_binary.npy", mmap_mode="r")
print(f"image={img.shape} prob={prob.shape} binary={binr.shape}")

neuroglancer.set_server_bind_address(args.bind, bind_port=args.port)
viewer = neuroglancer.Viewer()
dims = neuroglancer.CoordinateSpace(names=["z", "y", "x"], units="um", scales=args.voxel_um)


def local(a):
    return neuroglancer.LocalVolume(a, dimensions=dims)


with viewer.txn() as s:
    s.dimensions = dims
    # tissue — grayscale background
    s.layers["tissue"] = neuroglancer.ImageLayer(
        source=local(img),
        shader="#uicontrol invlerp v(range=[0,1])\nvoid main(){emitGrayscale(v());}",
    )
    # probability — red heatmap over the tissue
    s.layers["probability"] = neuroglancer.ImageLayer(
        source=local(prob),
        shader="#uicontrol invlerp v(range=[0,1])\nvoid main(){emitRGB(vec3(v(),0.0,0.0));}",
        opacity=0.6,
    )
    # binary mask — GREEN (distinct from the red probability layer), with
    # PROPORTIONAL alpha (boosted), not a hard threshold. When zoomed out,
    # neuroglancer serves a downsampled/averaged level where a 1-voxel fiber
    # becomes ~0.1; a hard `if(v()>0.5)` would hide all but the densest fiber
    # (the "only one streamline" bug). alpha = clamp(v()*4) keeps thin fibers
    # visible at every zoom. range=[0,1] is essential (uint8 else /255 -> 0.004).
    s.layers["binary"] = neuroglancer.ImageLayer(
        source=local(binr),
        shader=("#uicontrol invlerp v(range=[0,1])\n"
                "void main(){ float a = clamp(v()*4.0, 0.0, 1.0);"
                " emitRGBA(vec4(0.0,1.0,0.0,a)); }"),
    )
    s.layout = "xy"

print("\nNeuroglancer URL:")
print(viewer)
print("\nKeep this process running while viewing. Ctrl-C to stop.")
try:
    import time
    while True:
        time.sleep(3600)
except KeyboardInterrupt:
    print("stopped")
