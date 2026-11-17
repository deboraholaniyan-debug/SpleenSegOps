
import sys
import tempfile
import shutil
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

# Make modules inside src importable.
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from inference import predict_spleen


app = FastAPI(
    title="SpleenSegOps API",
    description="CPU-based 3D spleen segmentation using SlimFormer3D.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "project": "SpleenSegOps",
        "status": "running",
        "model": "SlimFormer3D",
        "device": "CPU",
    }


@app.post("/predict")
def predict(file: UploadFile = File(...)):
    """Accept a CT scan and return its spleen segmentation mask."""

    filename = (file.filename or "").lower()

    if not filename.endswith((".nii", ".nii.gz")):
        raise HTTPException(
            status_code=400,
            detail="Please upload a NIfTI file (.nii or .nii.gz).",
        )

    temp_dir = tempfile.TemporaryDirectory()
    work_dir = Path(temp_dir.name)

    input_path = work_dir / "input.nii.gz"
    output_path = work_dir / "spleen_mask.nii.gz"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print("Uploaded file size:", input_path.stat().st_size)
        print("Uploaded file header:", input_path.read_bytes()[:4])
        
        predict_spleen(
            input_path=input_path,
            output_path=output_path,
        )

        return FileResponse(
            path=str(output_path),
            media_type="application/gzip",
            filename="spleen_mask.nii.gz",
            background=BackgroundTask(temp_dir.cleanup),
        )

    except HTTPException:
        temp_dir.cleanup()
        raise

    except Exception as exc:
        temp_dir.cleanup()
        raise HTTPException(
            status_code=500,
            detail=f"Segmentation failed: {str(exc)}",
        )

    finally:
        file.file.close()
