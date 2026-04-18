from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import OpenAI
from dotenv import load_dotenv
from jose import jwt
from passlib.context import CryptContext
import psycopg2
import os, json, uuid, datetime

# -------------------------
# LOAD ENV
# -------------------------
load_dotenv()

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
DB_URL = os.getenv("DATABASE_URL")
SECRET_KEY = os.getenv("SECRET_KEY")

print("App starting...")
print("OPENAI:", bool(OPENAI_KEY))
print("DB:", bool(DB_URL))
print("SECRET:", bool(SECRET_KEY))

# -------------------------
# APP INIT
# -------------------------
app = FastAPI()

# -------------------------
# SAFE CLIENT INIT
# -------------------------
client = None
if OPENAI_KEY:
    client = OpenAI(api_key=OPENAI_KEY)

# -------------------------
# SAFE DB CONNECTION
# -------------------------
conn = None
if DB_URL:
    try:
        conn = psycopg2.connect(DB_URL, sslmode='require')
        conn.autocommit = True
        print("DB connected")
    except Exception as e:
        print("DB connection failed:", e)

# -------------------------
# AUTH
# -------------------------
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password):
    return pwd_context.hash(password)

def verify_password(password, hashed):
    return pwd_context.verify(password, hashed)

def create_token(user_id):
    if not SECRET_KEY:
        raise HTTPException(500, "SECRET_KEY missing")

    payload = {
        "user_id": user_id,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def get_user_from_token(token):
    if not SECRET_KEY:
        raise HTTPException(500, "SECRET_KEY missing")

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        return payload["user_id"]
    except:
        raise HTTPException(401, "Invalid token")

# -------------------------
# MODELS
# -------------------------
class User(BaseModel):
    email: str
    password: str

class Input(BaseModel):
    text: str
    project_id: str = None

class SelectConcept(BaseModel):
    index: int
    project_id: str

# -------------------------
# AUTH ROUTES
# -------------------------
@app.post("/signup")
def signup(user: User):
    if not conn:
        raise HTTPException(500, "DB not connected")

    cur = conn.cursor()
    hashed = hash_password(user.password)

    try:
        cur.execute(
            "INSERT INTO users (id, email, password) VALUES (%s,%s,%s)",
            (str(uuid.uuid4()), user.email, hashed)
        )
        return {"message": "User created"}
    except Exception as e:
        print(e)
        raise HTTPException(400, "User exists")

@app.post("/login")
def login(user: User):
    if not conn:
        raise HTTPException(500, "DB not connected")

    cur = conn.cursor()
    cur.execute("SELECT id, password FROM users WHERE email=%s", (user.email,))
    data = cur.fetchone()

    if not data:
        raise HTTPException(400, "User not found")

    user_id, hashed = data

    if not verify_password(user.password, hashed):
        raise HTTPException(400, "Wrong password")

    token = create_token(user_id)
    return {"token": token}

# -------------------------
# LLM
# -------------------------
def llm(prompt):
    if not client:
        return "OpenAI not configured"

    res = client.chat.completions.create(
        model="gpt-5.3",
        messages=[
            {"role": "system", "content": "Creative studio AI"},
            {"role": "user", "content": prompt}
        ],
        temperature=0.6
    )
    return res.choices[0].message.content.strip()

# -------------------------
# DB HELPERS
# -------------------------
def create_project(user_id):
    if not conn:
        raise HTTPException(500, "DB not connected")

    project_id = str(uuid.uuid4())
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO projects (id, user_id) VALUES (%s,%s)",
        (project_id, user_id)
    )
    return project_id

def get_project(project_id):
    if not conn:
        raise HTTPException(500, "DB not connected")

    cur = conn.cursor()
    cur.execute("SELECT * FROM projects WHERE id=%s", (project_id,))
    row = cur.fetchone()

    if not row:
        return None

    return {
        "id": row[0],
        "user_id": row[1],
        "brief": row[2],
        "analysis": row[3],
        "concepts": row[4],
        "selected": row[5],
        "moodboard": row[6],
        "images": row[7],
        "render3d": row[8],
        "cad": row[9]
    }

def update_project(project_id, field, value):
    if not conn:
        raise HTTPException(500, "DB not connected")

    cur = conn.cursor()
    cur.execute(
        f"UPDATE projects SET {field}=%s WHERE id=%s",
        (json.dumps(value) if isinstance(value,(list,dict)) else value, project_id)
    )

# -------------------------
# AI AGENTS
# -------------------------
def analysis_agent(b): return llm(f"Analyze:\n{b}")
def concept_agent(a):
    try:
        return json.loads(llm(f"Return 3 concepts JSON:\n{a}"))
    except:
        return []
def moodboard_agent(c): return llm(f"Moodboard:\n{c}")
def render3d_agent(c): return llm(f"3D:\n{c}")
def cad_agent(c): return llm(f"CAD:\n{c}")

def image_agent(c):
    if not client:
        return []

    imgs=[]
    for _ in range(2):
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"{c}, realistic render",
            size="1024x1024"
        )
        imgs.append(img.data[0].url)
    return imgs

# -------------------------
# MAIN ORCHESTRATOR
# -------------------------
@app.post("/run")
def run(data: Input, token: str):

    user_id = get_user_from_token(token)

    project_id = data.project_id or create_project(user_id)
    project = get_project(project_id)

    if not project["brief"]:
        update_project(project_id, "brief", data.text)
        return {"stage":"brief","project_id":project_id}

    if not project["analysis"]:
        r = analysis_agent(project["brief"])
        update_project(project_id,"analysis",r)
        return {"stage":"analysis","project_id":project_id}

    if not project["concepts"]:
        r = concept_agent(project["analysis"])
        update_project(project_id,"concepts",r)
        return {"stage":"concepts","data":r,"project_id":project_id}

    if not project["selected"]:
        return {"stage":"select","options":project["concepts"],"project_id":project_id}

    if not project["moodboard"]:
        r = moodboard_agent(project["selected"])
        update_project(project_id,"moodboard",r)
        return {"stage":"moodboard","project_id":project_id}

    if not project["images"]:
        r = image_agent(project["selected"])
        update_project(project_id,"images",r)
        return {"stage":"images","data":r,"project_id":project_id}

    if not project["render3d"]:
        r = render3d_agent(project["selected"])
        update_project(project_id,"render3d",r)
        return {"stage":"3d","project_id":project_id}

    if not project["cad"]:
        r = cad_agent(project["selected"])
        update_project(project_id,"cad",r)
        return {"stage":"cad","project_id":project_id}

    return {"stage":"complete","project_id":project_id}

# -------------------------
# SELECT
# -------------------------
@app.post("/select")
def select(req: SelectConcept, token: str):
    get_user_from_token(token)
    project = get_project(req.project_id)
    selected = project["concepts"][req.index]
    update_project(req.project_id,"selected",selected)
    return {"message":"selected"}
