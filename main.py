import os

import json

import uuid

import datetime

import subprocess

from typing import Optional, Any, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks

from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

from dotenv import load_dotenv

from openai import OpenAI

from jose import jwt

from passlib.context import CryptContext

import psycopg

from psycopg.rows import dict_row

# =========================

# ENV

# =========================

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

DATABASE_URL = os.getenv("DATABASE_URL")

SECRET_KEY = os.getenv("SECRET_KEY")

BLENDER_PATH = os.getenv("BLENDER_PATH", "blender")

BLENDER_SCRIPT = os.getenv("BLENDER_SCRIPT", "blender_script.py")

RENDER_OUTPUT_DIR = os.getenv("RENDER_OUTPUT_DIR", "/tmp/ai_creative_renders")

os.makedirs(RENDER_OUTPUT_DIR, exist_ok=True)

# =========================

# APP

# =========================

app = FastAPI(title="AI Creative Studio API")

app.add_middleware(

    CORSMiddleware,

    allow_origins=["*"],

    allow_credentials=True,

    allow_methods=["*"],

    allow_headers=["*"],

)

# =========================

# STARTUP

# =========================

@app.on_event("startup")

def startup() -> None:

    print("=== ENV CHECK ===")

    print("OPENAI:", bool(OPENAI_API_KEY))

    print("DB:", bool(DATABASE_URL))

    print("SECRET:", bool(SECRET_KEY))

# =========================

# CLIENTS

# =========================

client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

conn = None

if DATABASE_URL:

    try:

        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)

        conn.autocommit = True

        print("DB connected")

    except Exception as e:

        print("DB connection failed:", e)

        conn = None

# =========================

# HELPERS

# =========================

def get_cursor():

    if not conn:

        raise HTTPException(status_code=500, detail="DB not connected")

    return conn.cursor()

def safe_json(value: Any):

    if value is None:

        return None

    if isinstance(value, (dict, list)):

        return value

    if isinstance(value, str):

        try:

            return json.loads(value)

        except Exception:

            return value

    return value

# =========================

# AUTH

# =========================

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:

    return pwd.hash(password)

def verify_password(password: str, hashed: str) -> bool:

    return pwd.verify(password, hashed)

def create_token(user_id: str) -> str:

    if not SECRET_KEY:

        raise HTTPException(status_code=500, detail="SECRET_KEY missing")

    payload = {

        "user_id": user_id,

        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24),

    }

    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_user_from_token(token: str) -> str:

    if not SECRET_KEY:

        raise HTTPException(status_code=500, detail="SECRET_KEY missing")

    try:

        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])

        return payload["user_id"]

    except Exception:

        raise HTTPException(status_code=401, detail="Invalid token")

# =========================

# MODELS

# =========================

class UserInput(BaseModel):

    email: str

    password: str

    full_name: Optional[str] = None

class RunInput(BaseModel):

    text: str

    project_id: Optional[str] = None

    name: Optional[str] = None

    event_type: Optional[str] = None

class SelectConceptInput(BaseModel):

    project_id: str

    index: int

class Generate3DInput(BaseModel):

    project_id: str

    width: int = 1920

    height: int = 1080

class CommentInput(BaseModel):

    project_id: str

    section: str

    comment_text: str

class UpdateProjectInput(BaseModel):

    project_id: str

    field: str

    value: Any

# =========================

# ROOT

# =========================

@app.get("/")

def root():

    return {"message": "AI Creative Studio API is running"}

# =========================

# LLM

# =========================

def llm(prompt: str, temperature: float = 0.5) -> str:

    if not client:

        return "OpenAI not configured"

    response = client.chat.completions.create(

        model="gpt-4.1",

        messages=[

            {

                "role": "system",

                "content": (

                    "You are an expert AI Creative Studio planner for exhibitions, "

                    "activations, road shows, concerts, government events, and stage productions. "

                    "Be precise, structured, and production-focused."

                ),

            },

            {"role": "user", "content": prompt},

        ],

        temperature=temperature,

    )

    return response.choices[0].message.content.strip()

def llm_json(prompt: str) -> Any:

    raw = llm(prompt)

    try:

        return json.loads(raw)

    except Exception:

        cleaned = raw.strip().removeprefix("```json").removesuffix("```").strip()

        try:

            return json.loads(cleaned)

        except Exception:

            return {"raw": raw}

# =========================

# DB: USERS

# =========================

def create_user(email: str, password: str, full_name: Optional[str] = None) -> str:

    cur = get_cursor()

    user_id = str(uuid.uuid4())

    cur.execute(

        """

        insert into public.users (id, email, password, full_name)

        values (%s, %s, %s, %s)

        returning id

        """,

        (user_id, email, hash_password(password), full_name),

    )

    row = cur.fetchone()

    return str(row["id"])

def get_user_by_email(email: str):

    cur = get_cursor()

    cur.execute(

        """

        select id, email, password, full_name, role, is_active

        from public.users

        where email = %s

        """,

        (email,),

    )

    return cur.fetchone()

# =========================

# DB: PROJECTS

# =========================

def create_project(user_id: str, name: Optional[str] = None, event_type: Optional[str] = None) -> str:

    cur = get_cursor()

    project_id = str(uuid.uuid4())

    cur.execute(

        """

        insert into public.projects (id, user_id, name, event_type, status)

        values (%s, %s, %s, %s, %s)

        returning id

        """,

        (project_id, user_id, name or "Untitled Project", event_type, "draft"),

    )

    row = cur.fetchone()

    return str(row["id"])

def get_project(project_id: str) -> Optional[Dict[str, Any]]:

    cur = get_cursor()

    cur.execute(

        """

        select *

        from public.projects

        where id = %s

        """,

        (project_id,),

    )

    row = cur.fetchone()

    if not row:

        return None

    project = dict(row)

    for key in [

        "concepts",

        "selected",

        "images",

        "render3d",

        "scene_json",

        "deliverables",

        "dimensions",

        "brand_data",

        "presentation_data",

    ]:

        project[key] = safe_json(project.get(key))

    return project

def update_project_field(project_id: str, field: str, value: Any) -> None:

    allowed = {

        "name",

        "event_type",

        "status",

        "brief",

        "analysis",

        "concepts",

        "selected",

        "moodboard",

        "images",

        "render3d",

        "cad",

        "scene_json",

        "deliverables",

        "dimensions",

        "brand_data",

        "presentation_data",

    }

    if field not in allowed:

        raise HTTPException(status_code=400, detail="Invalid field")

    db_value = json.dumps(value) if isinstance(value, (dict, list)) else value

    cur = get_cursor()

    cur.execute(

        f"update public.projects set {field} = %s where id = %s",

        (db_value, project_id),

    )

def snapshot_project_version(project_id: str, user_id: str, note: str = "") -> None:

    project = get_project(project_id)

    if not project:

        return

    cur = get_cursor()

    cur.execute(

        """

        select coalesce(max(version_no), 0) + 1 as next_version

        from public.project_versions

        where project_id = %s

        """,

        (project_id,),

    )

    next_version = int(cur.fetchone()["next_version"])

    cur.execute(

        """

        insert into public.project_versions (id, project_id, user_id, version_no, snapshot, note)

        values (%s, %s, %s, %s, %s, %s)

        """,

        (

            str(uuid.uuid4()),

            project_id,

            user_id,

            next_version,

            json.dumps(project),

            note,

        ),

    )

# =========================

# DB: COMMENTS

# =========================

def add_comment(project_id: str, user_id: str, section: str, comment_text: str) -> str:

    cur = get_cursor()

    comment_id = str(uuid.uuid4())

    cur.execute(

        """

        insert into public.project_comments (id, project_id, user_id, section, comment_text, status)

        values (%s, %s, %s, %s, %s, %s)

        returning id

        """,

        (comment_id, project_id, user_id, section, comment_text, "open"),

    )

    row = cur.fetchone()

    return str(row["id"])

def get_comments(project_id: str):

    cur = get_cursor()

    cur.execute(

        """

        select *

        from public.project_comments

        where project_id = %s

        order by created_at desc

        """,

        (project_id,),

    )

    return [dict(r) for r in cur.fetchall()]

# =========================

# DB: RENDER JOBS

# =========================

def create_render_job(project_id: str, user_id: str, job_type: str, input_json: Dict[str, Any]) -> str:

    cur = get_cursor()

    job_id = str(uuid.uuid4())

    cur.execute(

        """

        insert into public.render_jobs (

            id, project_id, user_id, job_type, status, input_json

        )

        values (%s, %s, %s, %s, %s, %s)

        returning id

        """,

        (

            job_id,

            project_id,

            user_id,

            job_type,

            "queued",

            json.dumps(input_json),

        ),

    )

    row = cur.fetchone()

    return str(row["id"])

def update_render_job(job_id: str, status: str, output_json: Any = None, error_text: Optional[str] = None):

    cur = get_cursor()

    started_at = None

    finished_at = None

    if status == "running":

        started_at = datetime.datetime.utcnow()

    if status in {"done", "failed"}:

        finished_at = datetime.datetime.utcnow()

    cur.execute(

        """

        update public.render_jobs

        set

            status = %s,

            output_json = coalesce(%s, output_json),

            error_text = coalesce(%s, error_text),

            started_at = coalesce(%s, started_at),

            finished_at = coalesce(%s, finished_at)

        where id = %s

        """,

        (

            status,

            json.dumps(output_json) if output_json is not None else None,

            error_text,

            started_at,

            finished_at,

            job_id,

        ),

    )

def get_render_job(job_id: str):

    cur = get_cursor()

    cur.execute(

        """

        select *

        from public.render_jobs

        where id = %s

        """,

        (job_id,),

    )

    row = cur.fetchone()

    if not row:

        return None

    result = dict(row)

    result["input_json"] = safe_json(result.get("input_json"))

    result["output_json"] = safe_json(result.get("output_json"))

    return result

# =========================

# DB: FILES

# =========================

def add_project_file(

    project_id: str,

    user_id: Optional[str],

    file_type: str,

    file_name: str,

    file_url: str,

    meta: Optional[dict] = None,

):

    cur = get_cursor()

    cur.execute(

        """

        insert into public.project_files (id, project_id, user_id, file_type, file_name, file_url, meta)

        values (%s, %s, %s, %s, %s, %s, %s)

        """,

        (

            str(uuid.uuid4()),

            project_id,

            user_id,

            file_type,

            file_name,

            file_url,

            json.dumps(meta or {}),

        ),

    )

# =========================

# AI AGENTS

# =========================

def analyze_brief(brief: str) -> str:

    return llm(

        f"""

Analyze this creative brief for production planning.

Need:

1. event objective

2. audience profile

3. event type

4. key deliverables

5. branding requirements

6. probable dimensions if missing

7. missing clarifications

8. production concerns

Brief:

{brief}

"""

    )

def generate_concepts(analysis: str) -> Any:

    result = llm_json(

        f"""

Return exactly 3 strong concepts as a JSON array.

Each object must contain:

- name

- summary

- style

- materials

- colors

- lighting

- stage_elements

- camera_style

Analysis:

{analysis}

"""

    )

    return result if isinstance(result, list) else []

def generate_moodboard(selected_concept: Any) -> str:

    return llm(

        f"""

Create a polished moodboard and visual direction.

Include:

- material palette

- color palette

- lighting language

- finish recommendations

- branding placement logic

- premium visual notes

Concept:

{json.dumps(selected_concept, indent=2)}

"""

    )

def generate_concept_images(selected_concept: Any) -> Any:

    if not client:

        return []

    prompt = f"""

High-end event stage or exhibition concept in 16:9 ratio.

Use this concept:

{json.dumps(selected_concept, indent=2)}

Requirements:

- cinematic

- premium

- realistic materials

- polished design presentation

- 16:9 framing

"""

    urls = []

    for _ in range(3):

        img = client.images.generate(

            model="gpt-image-1",

            prompt=prompt,

            size="1536x1024",

        )

        urls.append(img.data[0].url)

    return urls

def generate_scene_json(selected_concept: Any, brief: str) -> Dict[str, Any]:

    result = llm_json(

        f"""

Create structured JSON for Blender scene generation.

Return valid JSON only.

Requirements:

- use feet

- render ratio 16:9

- stage dimensions

- led wall dimensions

- truss height

- audience rows and cols

- primary and secondary brand colors

- camera target

Brief:

{brief}

Concept:

{json.dumps(selected_concept, indent=2)}

"""

    )

    if not isinstance(result, dict):

        result = {}

    result.setdefault("units", "feet")

    result.setdefault("stage", {"width": 60, "depth": 24, "height": 4})

    result.setdefault("led_wall", {"type": "curved", "width": 40, "height": 12})

    result.setdefault("truss", {"height": 18})

    result.setdefault("audience", {"rows": 10, "cols": 20})

    result.setdefault("colors", {"primary": "#1A5DFF", "secondary": "#A855F7"})

    result.setdefault("lighting", {"style": "futuristic"})

    result.setdefault("branding", {"notes": "premium branded stage"})

    result.setdefault("camera_target", [0, 0, 6])

    result.setdefault("render", {"ratio": "16:9"})

    return result

# =========================

# BLENDER EXECUTION

# =========================

def run_blender_multi_angle(

    scene_json: Dict[str, Any],

    job_id: str,

    width: int,

    height: int,

    project_id: str,

    user_id: str,

):

    try:

        update_render_job(job_id, "running")

        job_dir = os.path.join(RENDER_OUTPUT_DIR, job_id)

        os.makedirs(job_dir, exist_ok=True)

        scene_json_path = os.path.join(job_dir, "scene.json")

        with open(scene_json_path, "w", encoding="utf-8") as f:

            json.dump(

                {

                    "scene": scene_json,

                    "render": {

                        "width": width,

                        "height": height,

                        "output_dir": job_dir,

                    },

                },

                f,

                indent=2,

            )

        cmd = [

            BLENDER_PATH,

            "-b",

            "-P",

            BLENDER_SCRIPT,

            "--",

            scene_json_path,

        ]

        completed = subprocess.run(cmd, capture_output=True, text=True)

        if completed.returncode != 0:

            update_render_job(job_id, "failed", error_text=completed.stderr[:2000])

            return

        render_map = {

            "front_wide": os.path.join(job_dir, "front_wide.png"),

            "front_center": os.path.join(job_dir, "front_center.png"),

            "left_perspective": os.path.join(job_dir, "left_perspective.png"),

            "right_perspective": os.path.join(job_dir, "right_perspective.png"),

            "top_plan": os.path.join(job_dir, "top_plan.png"),

            "audience_view": os.path.join(job_dir, "audience_view.png"),

            "glb": os.path.join(job_dir, "scene.glb"),

            "manifest": os.path.join(job_dir, "manifest.json"),

        }

        update_render_job(job_id, "done", output_json=render_map)

        update_project_field(project_id, "render3d", render_map)

        snapshot_project_version(project_id, user_id, note="Multi-angle render batch completed")

    except Exception as e:

        update_render_job(job_id, "failed", error_text=str(e))

# =========================

# AUTH ROUTES

# =========================

@app.post("/signup")

def signup(payload: UserInput):

    existing = get_user_by_email(payload.email)

    if existing:

        raise HTTPException(status_code=400, detail="User exists")

    user_id = create_user(payload.email, payload.password, payload.full_name)

    return {"message": "User created", "user_id": user_id}

@app.post("/login")

def login(payload: UserInput):

    user = get_user_by_email(payload.email)

    if not user:

        raise HTTPException(status_code=400, detail="User not found")

    if not verify_password(payload.password, user["password"]):

        raise HTTPException(status_code=400, detail="Wrong password")

    token = create_token(str(user["id"]))

    return {"token": token, "user_id": str(user["id"])}

# =========================

# PROJECT ROUTES

# =========================

@app.get("/project/{project_id}")

def project_detail(project_id: str, token: str):

    get_user_from_token(token)

    project = get_project(project_id)

    if not project:

        raise HTTPException(status_code=404, detail="Project not found")

    return project

@app.post("/project/update")

def project_update(payload: UpdateProjectInput, token: str):

    user_id = get_user_from_token(token)

    update_project_field(payload.project_id, payload.field, payload.value)

    snapshot_project_version(payload.project_id, user_id, note=f"Updated field: {payload.field}")

    return {"message": "Project updated"}

# =========================

# COMMENTS

# =========================

@app.post("/comment")

def create_comment(payload: CommentInput, token: str):

    user_id = get_user_from_token(token)

    comment_id = add_comment(payload.project_id, user_id, payload.section, payload.comment_text)

    return {"message": "Comment added", "comment_id": comment_id}

@app.get("/comments/{project_id}")

def list_comments(project_id: str, token: str):

    get_user_from_token(token)

    return {"comments": get_comments(project_id)}

# =========================

# MAIN PIPELINE

# =========================

@app.post("/run")

def run_pipeline(payload: RunInput, token: str):

    user_id = get_user_from_token(token)

    project_id = payload.project_id or create_project(user_id, payload.name, payload.event_type)

    project = get_project(project_id)

    if not project:

        raise HTTPException(status_code=404, detail="Project not found")

    if not project["brief"]:

        update_project_field(project_id, "brief", payload.text)

        update_project_field(project_id, "status", "brief_received")

        snapshot_project_version(project_id, user_id, note="Brief saved")

        return {"stage": "brief_saved", "project_id": project_id}

    if not project["analysis"]:

        analysis = analyze_brief(project["brief"])

        update_project_field(project_id, "analysis", analysis)

        update_project_field(project_id, "status", "analysis_ready")

        snapshot_project_version(project_id, user_id, note="Analysis generated")

        return {"stage": "analysis_ready", "project_id": project_id, "analysis": analysis}

    if not project["concepts"]:

        concepts = generate_concepts(project["analysis"])

        update_project_field(project_id, "concepts", concepts)

        update_project_field(project_id, "status", "concepts_ready")

        snapshot_project_version(project_id, user_id, note="Concept options generated")

        return {"stage": "concepts_ready", "project_id": project_id, "concepts": concepts}

    if not project["selected"]:

        return {

            "stage": "awaiting_concept_selection",

            "project_id": project_id,

            "options": project["concepts"],

        }

    if not project["moodboard"]:

        moodboard = generate_moodboard(project["selected"])

        update_project_field(project_id, "moodboard", moodboard)

        update_project_field(project_id, "status", "moodboard_ready")

        snapshot_project_version(project_id, user_id, note="Moodboard generated")

        return {"stage": "moodboard_ready", "project_id": project_id, "moodboard": moodboard}

    if not project["images"]:

        images = generate_concept_images(project["selected"])

        update_project_field(project_id, "images", images)

        update_project_field(project_id, "status", "concept_images_ready")

        snapshot_project_version(project_id, user_id, note="Concept images generated")

        return {"stage": "concept_images_ready", "project_id": project_id, "images": images}

    return {

        "stage": "ready_for_multi_angle_3d",

        "project_id": project_id,

        "message": "Concept approved. Ready for multi-angle render.",

    }

# =========================

# SELECT CONCEPT

# =========================

@app.post("/select")

def select_concept(payload: SelectConceptInput, token: str):

    user_id = get_user_from_token(token)

    project = get_project(payload.project_id)

    if not project:

        raise HTTPException(status_code=404, detail="Project not found")

    concepts = project["concepts"]

    if not isinstance(concepts, list) or payload.index < 0 or payload.index >= len(concepts):

        raise HTTPException(status_code=400, detail="Invalid concept index")

    selected = concepts[payload.index]

    update_project_field(payload.project_id, "selected", selected)

    update_project_field(payload.project_id, "status", "concept_selected")

    snapshot_project_version(payload.project_id, user_id, note=f"Concept {payload.index} selected")

    return {"message": "Concept selected", "selected": selected}

# =========================

# MULTI-ANGLE RENDER

# =========================

@app.post("/generate-multi-angle")

def generate_multi_angle(payload: Generate3DInput, token: str, background_tasks: BackgroundTasks):

    user_id = get_user_from_token(token)

    project = get_project(payload.project_id)

    if not project:

        raise HTTPException(status_code=404, detail="Project not found")

    if not project["selected"]:

        raise HTTPException(status_code=400, detail="Select concept first")

    scene_json = generate_scene_json(project["selected"], project["brief"] or "")

    update_project_field(payload.project_id, "scene_json", scene_json)

    job_id = create_render_job(

        project_id=payload.project_id,

        user_id=user_id,

        job_type="multi_angle_render",

        input_json={

            "scene_json": scene_json,

            "width": payload.width,

            "height": payload.height,

        },

    )

    background_tasks.add_task(

        run_blender_multi_angle,

        scene_json,

        job_id,

        payload.width,

        payload.height,

        payload.project_id,

        user_id,

    )

    return {

        "message": "Multi-angle render queued",

        "job_id": job_id,

        "project_id": payload.project_id,

    }

@app.get("/job/{job_id}")

def job_status(job_id: str, token: str):

    get_user_from_token(token)

    job = get_render_job(job_id)

    if not job:

        raise HTTPException(status_code=404, detail="Job not found")

    return job