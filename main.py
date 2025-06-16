# main.py
import io
import numpy as np
import open3d as o3d
import open3d.ml.torch as ml3d
from runpod.serverless import RPApp, RPJob, RPResponse

# Initialize the RPApp
rpapp = RPApp()

# Load the model once at startup
MODEL_PATH = "/app/BIM-Net++_HePIC.pth"
model = ml3d.models.BIMNetPlusPlus(backbone="hepic")
model.load(MODEL_PATH)
model.eval()

@rpapp.handler()
def handler(job: RPJob) -> RPResponse:
    """
    This function will be called whenever you POST a job to /run.
    Expects the PLY file as raw bytes under the 'ply_file' key.
    """
    # 1) Read bytes from the job input
    pcd_bytes = job.get_input_bytes("ply_file")
    
    # 2) Load into an Open3D point cloud
    pcd = o3d.io.read_point_cloud(io.BytesIO(pcd_bytes))
    pts = np.asarray(pcd.points, dtype=np.float32)
    
    # 3) Run inference (BIM-Net++ expects a [1,N,3] tensor)
    labels = model.infer(pts[np.newaxis, ...])[0]
    
    # 4) Return the labels as a list in JSON
    return rpapp.Response(output={"labels": labels.tolist()})

if __name__ == "__main__":
    # This starts the serverless worker process
    rpapp.run()

