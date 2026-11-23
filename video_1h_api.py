import asyncio
import json
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from PIL import Image, ImageDraw
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).parent
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
JOBS_FILE = OUTPUTS_DIR / "jobs.json"

ASSETS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

MAX_TOTAL_SECONDS = 3600.0  # 1 hour max per job

app = FastAPI(title="1-Hour Image & Video API", version="1.0.0")
JOBS: Dict[str, dict] = {}


def _load_jobs() -> None:
    global JOBS
    if JOBS_FILE.exists():
        try:
            JOBS = json.loads(JOBS_FILE.read_text(encoding="utf-8"))
        except Exception:
            JOBS = {}


def _save_jobs() -> None:
    JOBS_FILE.write_text(json.dumps(JOBS, indent=2), encoding="utf-8")


_load_jobs()


class SceneSpec(BaseModel):
    image: Optional[str] = Field(None, description="Filename in ./assets, file path, or http(s) URL")
    prompt: Optional[str] = Field(None, max_length=300, description="Text prompt for generated scene card if no image")
    duration_sec: float = Field(10.0, gt=0.5, le=3600.0, description="Scene duration in seconds")
    caption: Optional[str] = Field(None, max_length=160, description="Bottom caption bar text")


class VideoJobRequest(BaseModel):
    title: str = Field("one_hour_render", min_length=1, max_length=60)
    width: int = Field(1280, ge=480, le=1920)
    height: int = Field(720, ge=360, le=1920)
    fps: int = Field(30, ge=15, le=60)
    scenes: List[SceneSpec] = Field(..., min_length=1, max_length=500)


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _generate_image_card(prompt_text: str, out_path: Path, width: int, height: int, idx: int) -> Path:
    palettes = [
        ((15, 23, 42), (59, 130, 246)),
        ((24, 24, 27), (225, 29, 72)),
        ((17, 24, 39), (16, 185, 129)),
        ((30, 27, 75), (168, 85, 247)),
    ]
    bg, accent = palettes[idx % len(palettes)]
    img = Image.new("RGB", (width, height), bg)
    draw = ImageDraw.Draw(img)

    # Subtle accent frame and block
    margin = max(24, min(width, height) // 20)
    draw.rectangle([margin, margin, width - margin, height - margin], outline=accent, width=3)
    draw.rectangle([margin + 24, height // 2 - 42, margin + 36, height // 2 + 42], fill=accent)

    label = (prompt_text or f"Scene {idx + 1}").strip()
    draw.text((margin + 56, height // 2 - 12), label[:90], fill=(248, 250, 252))
    img.save(out_path, format="JPEG", quality=90)
    return out_path


async def _prepare_image(scene: SceneSpec, work_dir: Path, idx: int, width: int, height: int) -> Path:
    if scene.image:
        if scene.image.startswith(("http://", "https://")):
            target = work_dir / f"src_{idx:04d}.jpg"
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(scene.image)
                resp.raise_for_status()
                target.write_bytes(resp.content)
            return target

        p = Path(scene.image)
        if not p.is_absolute():
            candidate = ASSETS_DIR / scene.image
            p = candidate if candidate.exists() else (BASE_DIR / scene.image)
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {scene.image}")
        return p

    target = work_dir / f"gen_{idx:04d}.jpg"
    return await asyncio.to_thread(_generate_image_card, scene.prompt or "", target, width, height, idx)


def _render_scene_mp4(
    image_path: Path,
    out_clip: Path,
    duration: float,
    width: int,
    height: int,
    fps: int,
    caption: Optional[str],
) -> None:
    scale_pad = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,setsar=1"
    )
    if caption:
        safe = _escape_drawtext(caption)
        vf = (
            f"{scale_pad},drawbox=x=0:y=ih-100:w=iw:h=100:color=black@0.55:t=fill,"
            f"drawtext=text='{safe}':fontcolor=white:fontsize=28:x=(w-text_w)/2:y=h-64"
        )
    else:
        vf = scale_pad

    cmd = [
        "ffmpeg",
        "-y",
        "-loop",
        "1",
        "-t",
        f"{duration:.3f}",
        "-i",
        str(image_path),
        "-f",
        "lavfi",
        "-t",
        f"{duration:.3f}",
        "-i",
        "anullsrc=r=44100:cl=stereo",
        "-vf",
        vf,
        "-r",
        str(fps),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-c:a",
        "aac",
        "-ar",
        "44100",
        "-ac",
        "2",
        "-b:a",
        "128k",
        "-shortest",
        str(out_clip),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Scene FFmpeg error: {res.stderr[-500:]}")


def _concat_stream_copy(clips: List[Path], list_file: Path, final_mp4: Path) -> None:
    lines = [f"file '{clip.resolve().as_posix()}'" for clip in clips]
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(list_file),
        "-c",
        "copy",
        str(final_mp4),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"Concat demuxer error: {res.stderr[-500:]}")


async def _run_video_job(job_id: str, req: VideoJobRequest) -> None:
    JOBS[job_id]["status"] = "processing"
    _save_jobs()

    work_dir = OUTPUTS_DIR / f"work_{job_id}"
    work_dir.mkdir(parents=True, exist_ok=True)

    try:
        clips: List[Path] = []
        total_scenes = len(req.scenes)
        for idx, scene in enumerate(req.scenes):
            img_path = await _prepare_image(scene, work_dir, idx, req.width, req.height)
            clip_path = work_dir / f"clip_{idx:04d}.mp4"
            await asyncio.to_thread(
                _render_scene_mp4,
                img_path,
                clip_path,
                scene.duration_sec,
                req.width,
                req.height,
                req.fps,
                scene.caption,
            )
            clips.append(clip_path)
            JOBS[job_id]["progress_pct"] = int(((idx + 1) / total_scenes) * 90)
            _save_jobs()

        safe_title = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in req.title)
        out_name = f"{job_id}_{safe_title}.mp4"
        final_mp4 = OUTPUTS_DIR / out_name
        list_file = work_dir / "clips.txt"

        await asyncio.to_thread(_concat_stream_copy, clips, list_file, final_mp4)

        JOBS[job_id]["status"] = "completed"
        JOBS[job_id]["progress_pct"] = 100
        JOBS[job_id]["file"] = out_name
        JOBS[job_id]["download_url"] = f"/v1/videos/{out_name}"
        _save_jobs()
    except Exception as exc:
        JOBS[job_id]["status"] = "failed"
        JOBS[job_id]["error"] = str(exc)
        _save_jobs()
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "max_duration_sec": MAX_TOTAL_SECONDS,
    }


@app.post("/v1/jobs", status_code=202)
async def create_video_job(req: VideoJobRequest):
    if not shutil.which("ffmpeg"):
        raise HTTPException(status_code=500, detail="FFmpeg not installed or not in PATH")

    total_sec = sum(s.duration_sec for s in req.scenes)
    if total_sec > MAX_TOTAL_SECONDS:
        raise HTTPException(
            status_code=400,
            detail=f"Total duration ({total_sec}s) exceeds 1-hour limit ({MAX_TOTAL_SECONDS}s)",
        )

    job_id = uuid.uuid4().hex[:12]
    JOBS[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress_pct": 0,
        "total_duration_sec": total_sec,
        "file": None,
        "download_url": None,
        "error": None,
    }
    _save_jobs()

    asyncio.create_task(_run_video_job(job_id, req))
    return JOBS[job_id]


@app.get("/v1/jobs/{job_id}")
def get_job_status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]


@app.get("/v1/videos/{filename}")
def download_video(filename: str):
    target = (OUTPUTS_DIR / Path(filename).name).resolve()
    if not target.exists() or target.parent != OUTPUTS_DIR.resolve():
        raise HTTPException(status_code=404, detail="Video file not found")
    return FileResponse(path=target, media_type="video/mp4", filename=target.name)
