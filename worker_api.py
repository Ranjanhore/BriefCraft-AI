import os
import json
import uuid
import subprocess
from threading import Thread

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Blender Worker API")

BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")
BLENDER_SCRIPT = os.getenv("BLENDER_SCRIPT", "blender_script.py")
WORKER_OUTPUT_DIR = os.getenv("WORKER_OUTPUT_DIR", "/tmp/blender_worker_jobs")

os.makedirs(WORKER_OUTPUT_DIR, exist_ok=True)

jobs = {}


class RenderRequest(BaseModel):
    scene: dict
    render: dict


def run_blender_job(job_id: str, payload: dict):
    try:
        jobs[job_id]["status"] = "running"

        job_dir = os.path.join(WORKER_OUTPUT_DIR, job_id)
        os.makedirs(job_dir, exist_ok=True)

        json_path = os.path.join(job_dir, "scene.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        cmd = [
            BLENDER_PATH,
            "-b",
            "-P",
            BLENDER_SCRIPT,
            "--",
            json_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            jobs[job_id]["status"] = "failed"
            jobs[job_id]["error"] = result.stderr
            return

        manifest_path = os.path.join(payload["render"]["output_dir"], "manifest.json")
        if not os.path.exists(manifest_path):
            manifest_path = os.path.join(job_dir, "manifest.json")

        manifest = None
        if os.path.exists(manifest_path):
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        jobs[job_id]["status"] = "done"
        jobs[job_id]["result"] = manifest or {
            "output_dir": job_dir
        }

    except Exception as e:
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)


@app.get("/")
def root():
    return {"message": "Blender Worker API running"}


@app.post("/render")
def render(payload: RenderRequest):
    job_id = str(uuid.uuid4())

    job_dir = os.path.join(WORKER_OUTPUT_DIR, job_id)
    render_payload = payload.dict()

    # force isolated output dir per job
    render_payload["render"]["output_dir"] = job_dir

    jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "result": None,
        "error": None
    }

    t = Thread(target=run_blender_job, args=(job_id, render_payload))
    t.start()

    return {
        "job_id": job_id,
        "status": "queued"
    }


@app.get("/job/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]
