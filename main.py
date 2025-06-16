from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import Response
import numpy as np, open3d as o3d, io
import open3d.ml.torch as ml3d

app = FastAPI()

# Load the model once at startup
model = ml3d.models.BIMNetPlusPlus(backbone="hepic")
model.load("/app/BIM-Net++_HePIC.pth")
model.eval()

@app.post("/infer")
async def infer(ply_file: UploadFile = File(...)):
    data = await ply_file.read()
    pcd = o3d.io.read_point_cloud(io.BytesIO(data))
    pts = np.asarray(pcd.points, dtype=np.float32)
    labels = model.infer(pts[np.newaxis, ...])[0]
    out = io.BytesIO()
    np.save(out, labels)
    out.seek(0)
    return Response(content=out.read(), media_type="application/octet-stream")
