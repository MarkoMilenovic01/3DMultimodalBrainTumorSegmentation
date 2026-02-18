from __future__ import annotations
from pathlib import Path
import numpy as np
from skimage import measure
import trimesh

def main():
    pred_path = Path("predictions_val/BraTS20_Validation_042_pred.npy")  # change
    out_dir = Path("recon")
    out_dir.mkdir(parents=True, exist_ok=True)

    pred = np.load(pred_path).astype(np.uint8)  # (D,H,W), labels 0..3

    # Pick which region to reconstruct:
    # whole tumor = anything >0
    tumor = (pred > 0).astype(np.uint8)

    if tumor.sum() == 0:
        print("No tumor predicted in this case.")
        return

    # marching cubes expects (Z,Y,X) or similar; our pred is (D,H,W) = (Z,Y,X)
    verts, faces, normals, values = measure.marching_cubes(tumor, level=0.5)

    mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_normals=normals, process=True)

    out_path = out_dir / (pred_path.stem.replace("_pred", "") + "_tumor.stl")
    mesh.export(out_path)
    print("Saved mesh:", out_path)

if __name__ == "__main__":
    main()
