from __future__ import annotations

import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from ai_engine import pipeline


PROJECT_ROOT = Path(__file__).resolve().parent.parent
UPLOADS_DIR = PROJECT_ROOT / "data" / "uploads"
RESULTS_DIR = PROJECT_ROOT / "data" / "results"
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# The existing engine uses a module-level output directory and fixed filenames.
# Serialize runs while redirecting that directory to keep results isolated.
_PIPELINE_LOCK = threading.Lock()

app = FastAPI(title="LUCAS Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
app.mount(
    "/api/results",
    StaticFiles(directory=RESULTS_DIR),
    name="results",
)
app.mount(
    "/api/dataset-files",
    StaticFiles(directory=DATA_DIR),
    name="dataset-files",
)
app.mount(
    "/frontend",
    StaticFiles(directory=FRONTEND_DIR, html=True),
    name="frontend",
)


@app.get("/")
def root():
    return RedirectResponse(url="/frontend/")


def _safe_extension(filename: str | None, content_type: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    if suffix in allowed:
        return suffix

    content_types = {
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/bmp": ".bmp",
        "image/tiff": ".tiff",
        "image/webp": ".webp",
    }
    return content_types.get(content_type or "", ".bin")


def _dataset_path(relative_path: str) -> Path:
    candidate = (DATA_DIR / relative_path).resolve()
    try:
        candidate.relative_to(DATA_DIR.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Invalid dataset path.") from exc

    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Dataset image not found.")
    return candidate


def _dataset_entries() -> list[dict[str, Any]]:
    image_suffixes = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
    entries = []
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        if "results" in path.parts or "uploads" in path.parts:
            continue

        relative = path.relative_to(DATA_DIR).as_posix()
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        height = int(image.shape[0]) if image is not None else None
        width = int(image.shape[1]) if image is not None else None
        entries.append(
            {
                "name": path.name,
                "path": relative,
                "type": path.suffix.lower().lstrip("."),
                "width": width,
                "height": height,
                "preview_url": f"/api/dataset-preview/{relative}",
                "file_url": f"/api/dataset-files/{relative}",
                "dataset": path.parent.name,
            }
        )
    return entries


def _dataset_response() -> dict[str, Any]:
    entries = _dataset_entries()
    default_source = "processed/tmc2_apollo16_focused.png"
    default_reference = "processed/lro_apollo16_overlap.png"
    paths = {entry["path"] for entry in entries}
    return {
        "images": entries,
        "source_images": entries,
        "reference_images": entries,
        "default_source": default_source if default_source in paths else None,
        "default_reference": default_reference if default_reference in paths else None,
    }


async def _save_image(upload: UploadFile, path: Path) -> None:
    if not (upload.content_type or "").startswith("image/"):
        raise HTTPException(
            status_code=415,
            detail=f"{upload.filename or 'Uploaded file'} is not an image.",
        )

    content = await upload.read()
    image = cv2.imdecode(
        np.frombuffer(content, dtype=np.uint8),
        cv2.IMREAD_UNCHANGED,
    )
    if image is None:
        raise HTTPException(
            status_code=415,
            detail=f"{upload.filename or 'Uploaded file'} is not a valid image.",
        )

    path.write_bytes(content)


def _result_path_url(path_value: Any, run_id: str) -> str | None:
    if not path_value:
        return None

    path = Path(str(path_value))
    try:
        relative = path.relative_to(RESULTS_DIR)
    except ValueError:
        return None

    return f"/api/results/{run_id}/{relative.name}"


def _response_from_result(result: dict[str, Any], run_id: str) -> dict[str, Any]:
    geometry = result.get("geometry") or {}
    accepted = result.get("status") == "ACCEPTED"
    quality_reasons = result.get("quality_reasons") or []

    return {
        "run_id": run_id,
        "success": accepted,
        "metrics": {
            "matches": result.get("matches"),
            "inliers": result.get("inliers"),
            "inlier_ratio": result.get("inlier_ratio"),
            "rmse": result.get("rmse_px"),
            "median_error": result.get("median_error_px"),
            "max_error": result.get("max_error_px"),
            "spatial_coverage": result.get("spatial_coverage"),
            "spatial_uniformity": result.get("spatial_uniformity"),
        },
        "geometry_model": geometry.get("selected_model"),
        "quality": {
            "success": accepted,
            "reason": result.get("reason"),
            "reasons": quality_reasons,
        },
        "outputs": {
            "registered_image": _result_path_url(
                result.get("registered_image"), run_id
            ),
            "match_visualization": _result_path_url(
                result.get("match_visualization"), run_id
            ),
            "inlier_visualization": _result_path_url(
                result.get("inlier_visualization"), run_id
            ),
        },
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "LUCAS Backend"}


@app.get("/api/dataset")
def dataset() -> dict[str, Any]:
    return _dataset_response()


@app.get("/api/dataset-preview/{relative_path:path}")
def dataset_preview(relative_path: str) -> Response:
    path = _dataset_path(relative_path)
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise HTTPException(status_code=415, detail="Dataset image cannot be previewed.")

    height, width = image.shape[:2]
    scale = min(1.0, 1400 / max(width, height))
    if scale < 1.0:
        image = cv2.resize(
            image,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    if image.ndim == 3 and image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not encoded:
        raise HTTPException(status_code=500, detail="Unable to create image preview.")
    return Response(content=buffer.tobytes(), media_type="image/jpeg")


@app.get("/", include_in_schema=False)
def frontend() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.post("/api/register")
async def register(
    source_image: UploadFile | None = File(default=None),
    reference_image: UploadFile | None = File(default=None),
    source_path: str | None = Form(default=None),
    reference_path: str | None = Form(default=None),
) -> dict[str, Any]:
    if source_path and reference_path:
        source_dataset_path = _dataset_path(source_path)
        reference_dataset_path = _dataset_path(reference_path)
        run_id = uuid.uuid4().hex
        source_path_value = source_dataset_path
        reference_path_value = reference_dataset_path
    elif source_image is None:
        raise HTTPException(status_code=400, detail="source_image is required.")
    elif reference_image is None:
        raise HTTPException(
            status_code=400,
            detail="reference_image is required.",
        )
    else:
        run_id = uuid.uuid4().hex
        source_path_value = UPLOADS_DIR / f"{run_id}_source{_safe_extension(source_image.filename, source_image.content_type)}"
        reference_path_value = UPLOADS_DIR / f"{run_id}_reference{_safe_extension(reference_image.filename, reference_image.content_type)}"
        await _save_image(source_image, source_path_value)
        await _save_image(reference_image, reference_path_value)

    run_results_dir = RESULTS_DIR / run_id
    run_results_dir.mkdir(parents=True, exist_ok=True)

    with _PIPELINE_LOCK:
        previous_results_dir = pipeline.RESULTS_DIR
        pipeline.RESULTS_DIR = run_results_dir
        try:
            result = pipeline.run_lucas(
                source_path=source_path_value,
                reference_path=reference_path_value,
                show_visualization=False,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            raise HTTPException(
                status_code=500,
                detail=f"LUCAS registration failed: {exc}",
            ) from exc
        finally:
            pipeline.RESULTS_DIR = previous_results_dir

    return _response_from_result(result, run_id)