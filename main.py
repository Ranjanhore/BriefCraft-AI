# =============================================================================
# GurukulAI Brain  —  Production Backend
# Perfect Indian Teacher: Intro → Chapter Intro → Teaching → Quiz → Done
# =============================================================================
import os
import re
import uuid
from typing import Any, Dict, List, Optional
import requests
import tempfile
import fitz

# ── FastAPI ────────────────────────────────────────────────────────────────
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── OpenAI  (pre-init all vars — safe on any Python version) ──────────────
_OpenAI = None          # type: ignore
_openai_client = None
try:
    from openai import OpenAI as _OpenAI
except Exception as _e:
    print(f"[WARN] openai import failed: {_e}")

# ── Supabase  (pre-init all vars — safe on any Python version) ────────────
_SUPABASE_OK = False
_sb_create = None       # type: ignore
try:
    from supabase import create_client as _sb_create
    _SUPABASE_OK = True
except Exception as _e:
    print(f"[WARN] supabase import failed: {_e}")

# =============================================================================
# ENV CONFIG
# =============================================================================
AUDIO_DIR     = os.getenv("AUDIO_DIR", "audio_files")
RENDER_DOMAIN = os.getenv("RENDER_EXTERNAL_HOSTNAME", "localhost:8000")
OPENAI_KEY    = os.getenv("OPENAI_API_KEY", "")
SB_URL = "https://zvfebuoasoomeanevcjz.supabase.co"
SB_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Inp2ZmVidW9hc29vbWVhbmV2Y2p6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzE3NjA5ODcsImV4cCI6MjA4NzMzNjk4N30.5w14-FT1yCMod1OtSMOK8Ibj3pJEdLeqOJ1aruEsoJU"

os.makedirs(AUDIO_DIR, exist_ok=True)

# Initialise OpenAI client safely
def _get_openai():
    global _openai_client
    if _openai_client is not None:
        return _openai_client
    if _OpenAI and OPENAI_KEY:
        try:
            _openai_client = _OpenAI(api_key=OPENAI_KEY)
        except Exception as e:
            print(f"[WARN] OpenAI init: {e}")
    return _openai_client

# Initialise Supabase client safely
_sb_client = None
def _get_sb():
    global _sb_client
    if _sb_client is not None:
        return _sb_client
    if _SUPABASE_OK and _sb_create and SB_URL and SB_KEY:
        try:
            _sb_client = _sb_create(SB_URL, SB_KEY)
            print("[DB] Supabase connected")
        except Exception as e:
            print(f"[WARN] Supabase init: {e}")
    return _sb_client

# =============================================================================
# APP  (middleware BEFORE mount to avoid Starlette ordering issues)
# =============================================================================
app = FastAPI(title="GurukulAI Brain", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")

# =============================================================================
# SESSION STORE
# =============================================================================
SESSIONS: Dict[str, Dict[str, Any]] = {}

# =============================================================================
# PYDANTIC MODELS
# =============================================================================
class StartSession(BaseModel):
    session_id:          Optional[str] = None
    board:               Optional[str] = None
    class_level:         Optional[str] = None
    subject:             Optional[str] = None
    book_name:           Optional[str] = None
    chapter_name:        Optional[str] = None
    chapter_title:       Optional[str] = None
    chapter_code:        Optional[str] = None
    syllabus_chapter_id: Optional[str] = None
    language:            Optional[str] = None

class StudentReply(BaseModel):
    session_id:          str
    message:             str
    board:               Optional[str] = None
    class_level:         Optional[str] = None
    subject:             Optional[str] = None
    book_name:           Optional[str] = None
    chapter_name:        Optional[str] = None
    chapter_title:       Optional[str] = None
    chapter_code:        Optional[str] = None
    syllabus_chapter_id: Optional[str] = None

class TTSRequest(BaseModel):
    text:             str
    voice:            Optional[str] = None
    model:            Optional[str] = None
    instructions:     Optional[str] = None
    subject:          Optional[str] = None
    teacher_gender:   Optional[str] = "female"
    preferred_language: Optional[str] = None

class HomeworkRequest(BaseModel):
    session_id: str
    question:   str
    subject:    Optional[str] = None
    board:      Optional[str] = None
    class_level: Optional[str] = None
    chapter_name: Optional[str] = None
    chapter_title: Optional[str] = None
    chapter_code: Optional[str] = None
    syllabus_chapter_id: Optional[str] = None
    book_name: Optional[str] = None

# =============================================================================
# TEXT UTILITIES
# =============================================================================
_EMOJI = ["😊","🙂","😂","😄","👍","😃","🎉","✅","❌","🎓","📚","💡","•","→","►"]
_PLACEHOLDER_FRAGMENTS = [
    "अब इस अंश को सरल भाषा में समझते हैं",
    "इस अध्याय का विषय है",
    "suno dhyan se",
]

def clean_text(text: str) -> str:
    """Strip markdown, emoji, extra whitespace for TTS."""
    if not text:
        return ""
    for ch in _EMOJI + ["*","_","#","`","~","**","__"]:
        text = text.replace(ch, "")
    text = re.sub(r"\n{2,}", ". ", text)
    text = re.sub(r"\n", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def is_placeholder(text: Optional[str]) -> bool:
    if not text or len(text.strip()) < 25:
        return True
    t = text.lower()
    return any(p.lower() in t for p in _PLACEHOLDER_FRAGMENTS)

def is_krutidev(text: Optional[str]) -> bool:
    """Detect Krutidev legacy font encoding (ASCII-faked Hindi)."""
    if not text or len(text) < 15:
        return True
    dev = sum(1 for c in text if "\u0900" <= c <= "\u097f")
    asc = sum(1 for c in text if c.isascii() and c.isalpha())
    return asc > 8 and dev < 4

# =============================================================================
# DATABASE HELPERS  (all return empty gracefully when DB unavailable)
# =============================================================================
def db_chapter_meta(chapter_id: str) -> dict:
    sb = _get_sb()
    if not sb or not chapter_id:
        return {}
    try:
        r = (sb.table("syllabus_chapters")
               .select("id,chapter_title,chapter_code,book_name,author,board,class_level,subject,language")
               .eq("id", chapter_id).limit(1).execute())
        return (r.data or [{}])[0]
    except Exception as e:
        print(f"[DB] chapter_meta: {e}"); return {}

def db_teacher(subject: str, board: str, class_level: str) -> dict:
    sb = _get_sb()
    if not sb:
        return {}
    try:
        # Try subject-specific teacher first via teacher_subject_map
        r = (sb.table("teacher_subject_map")
               .select("teacher_id, teacher_profiles(teacher_code,teacher_name,gender,base_language,accent_style,persona_prompt,teaching_pattern,voice_provider,voice_id)")
               .eq("subject", subject).eq("board", board).eq("active", True)
               .limit(1).execute())
        rows = r.data or []
        if rows and rows[0].get("teacher_profiles"):
            return rows[0]["teacher_profiles"]
        # Fallback: any active teacher
        r2 = (sb.table("teacher_profiles")
                .select("teacher_code,teacher_name,gender,base_language,accent_style,persona_prompt,teaching_pattern,voice_provider,voice_id")
                .eq("active", True).limit(1).execute())
        return (r2.data or [{}])[0]
    except Exception as e:
        print(f"[DB] teacher: {e}"); return {}

def db_live_chunks(chapter_id: str) -> List[dict]:
    sb = _get_sb()
    if not sb or not chapter_id:
        return []
    try:
        r = (sb.table("live_teaching_chunks")
               .select("id,live_order,chunk_kind,stage_type,cleaned_text,read_text,explain_text,ask_text,practice_text,recap_text,keywords")
               .eq("chapter_id", chapter_id).eq("is_active", True)
               .order("live_order").limit(300).execute())
        return r.data or []
    except Exception as e:
        print(f"[DB] live_chunks: {e}"); return []

def db_quiz_questions(chapter_id: str) -> List[dict]:
    sb = _get_sb()
    if not sb or not chapter_id:
        return []
    try:
        # Try chapter_quiz_questions via chapter_parts
        r = (sb.table("chapter_quiz_questions")
               .select("question_text,options,correct_answer,explanation,question_type")
               .eq("is_active", True).limit(8).execute())
        return r.data or []
    except Exception as e:
        print(f"[DB] quiz: {e}"); return []

# =============================================================================
# TTS  — Human Indian Teacher Voice
# =============================================================================
# OpenAI voice personalities:
# nova   → warm, clear, female  ← best for Indian English teacher
# shimmer → soft, gentle, female
# alloy  → neutral, balanced
# echo   → male, warm
# onyx   → male, deep
# fable  → male, storytelling

_TTS_VOICE_MAP = {
    "female": "nova",
    "male":   "echo",
    "nova": "nova", "shimmer": "shimmer", "alloy": "alloy",
    "echo": "echo", "onyx": "onyx", "fable": "fable",
    "ash": "echo", "coral": "shimmer", "sage": "alloy",
}

def _pick_voice(session: dict) -> str:
    teacher = session.get("teacher", {})
    vid = teacher.get("voice_id", "")
    if vid and vid in _TTS_VOICE_MAP:
        return _TTS_VOICE_MAP[vid]
    gender = (teacher.get("gender", "") or "female").lower()
    return _TTS_VOICE_MAP.get(gender, "nova")

def _tts_instructions(lang: str) -> str:
    """Build voice instructions for perfect Indian accent based on student language."""
    lang_low = (lang or "").lower()

    # Regional language pronunciation guidance
    if "bengali" in lang_low or "bangla" in lang_low:
        lang_note = (
            "Pronounce Bengali words with authentic softness — 'bhaalo', 'tumi', 'keno', 'shikha' "
            "should sound natural, not anglicised. Switch warmly between Bengali and Hindi."
        )
    elif "hindi" in lang_low or "hinglish" in lang_low:
        lang_note = (
            "Pronounce Hindi words with full authentic Devanagari sounds. "
            "'maa' not 'mah', 'padhna' not 'paadna', 'pyaar' with proper retroflex. "
            "Hindi should sound like a native Hindi speaker, not translated English."
        )
    elif "tamil" in lang_low:
        lang_note = "Mix Tamil words naturally. Pronounce Tamil with authentic retroflex sounds."
    elif "telugu" in lang_low:
        lang_note = "Mix Telugu words naturally with authentic pronunciation."
    elif "marathi" in lang_low:
        lang_note = "Mix Marathi naturally. Pronounce Marathi words authentically."
    elif "gujarati" in lang_low:
        lang_note = "Mix Gujarati naturally with authentic pronunciation."
    elif "kannada" in lang_low:
        lang_note = "Mix Kannada naturally with authentic pronunciation."
    elif "malayalam" in lang_low:
        lang_note = "Mix Malayalam naturally with authentic pronunciation."
    elif "punjabi" in lang_low:
        lang_note = "Mix Punjabi naturally. Pronounce Punjabi with authentic sounds."
    else:
        lang_note = (
            "Use natural Indian English pronunciation — warm Indian accent, "
            "not American or British."
        )

    return f"""You are Priya Ma'am — a warm, energetic, experienced Indian school teacher.

ACCENT AND VOICE:
- Speak with a NATURAL INDIAN ACCENT — NOT American, NOT British, NOT robotic
- Volume: speak LOUDLY and CLEARLY like a classroom teacher — not soft or mumbled
- Pace: slightly slower than conversation, pause between key ideas for emphasis
- Energy: HIGH and GENUINE — excited about teaching, proud of students
- Warmth: every sentence carries genuine care for the student

LANGUAGE PRONUNCIATION:
- {lang_note}
- English technical words: pronounce with Indian English (not American accent)
- Names: Ranjan, Priya, Malhar — authentic Indian pronunciation always
- NEVER anglicise Indian words — keep them authentic

EMOTIONAL EXPRESSION:
- "Arey waah!" — genuinely excited, voice lifts up
- "Ekdum sahi!" — proud and celebratory tone
- "Yaad rakhna..." — slower, serious, emphasised
- Questions: voice rises naturally at the end
- Corrections: warm and encouraging, never harsh
- Build suspense before revealing answers — slight pause then reveal"""


def generate_audio(text: str, voice: str = "nova", lang: str = "Hinglish") -> Optional[str]:
    """Generate TTS audio with Indian accent instructions. Returns public URL or None."""
    oai = _get_openai()
    if not oai:
        return None
    clean = clean_text(text)
    if not clean or len(clean) < 3:
        return None
    try:
        # Try gpt-4o-mini-tts first (supports instructions for accent control)
        try:
            resp = oai.audio.speech.create(
                model="gpt-4o-mini-tts",
                voice=voice,
                input=clean[:4000],
                instructions=_tts_instructions(lang),
            )
        except Exception:
            # Fallback to tts-1-hd if gpt-4o-mini-tts unavailable
            resp = oai.audio.speech.create(
                model="tts-1-hd",
                voice=voice,
                input=clean[:4000],
                speed=0.92,
            )
        fname = f"{uuid.uuid4()}.mp3"
        fpath = os.path.join(AUDIO_DIR, fname)
        resp.stream_to_file(fpath)
        return f"https://{RENDER_DOMAIN}/audio/{fname}"
    except Exception as e:
        print(f"[TTS] error: {e}")
        return None

# =============================================================================
# GPT  — Brain of the teacher
# =============================================================================
def _gpt(messages: List[dict], max_tokens: int = 400, temp: float = 0.85) -> str:
    oai = _get_openai()
    if not oai:
        return "Chalo aage badhte hain!"
    try:
        r = oai.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temp,
        )
        return r.choices[0].message.content.strip()
    except Exception as e:
        print(f"[GPT] error: {e}")
        return "Thoda technical issue hai, lekin chalo aage badhte hain!"

# =============================================================================
# TEACHER PERSONA  — The heart of GurukulAI
# =============================================================================
def _lang_rules(lang: str) -> str:
    """Return strict language mixing rules based on student preference."""
    l = (lang or "").lower()

    if "bengali" in l or "bangla" in l:
        return """LANGUAGE — BENGALI + HINDI MIX (MANDATORY):
You MUST speak in Bengali + Hindi mix throughout. No pure English unless it's a technical term.
Bengali words to use naturally: "bhalo" (good), "keno" (why), "ki" (what), "tumi" (you),
"amra" (we), "bujhecho?" (understood?), "darun!" (excellent!), "thik ache" (okay),
"shono" (listen), "dekho" (look/see), "janao" (tell me), "boro" (big), "choto" (small).
Celebrate: "Darun! Khub bhalo!" | Questions: "Tumi ki bujhecho?" | Encouragement: "Cheshta koro!"
Mix example: "Dekho, jemon amar ma amader care koren — sheikhanei 'Maa' ar 'Matrubhumi' eki."
NEVER speak in only English. Bengali first, Hindi support, English only for terms."""

    elif "hindi" in l or "hinglish" in l:
        return """LANGUAGE — HINDI + ENGLISH MIX (HINGLISH):
Speak natural Hinglish — Hindi base with English terms mixed.
Use full Hindi sentences: "Dekho beta, yeh concept bahut important hai."
Hindi celebrations: "Ekdum sahi!", "Wah wah!", "Bahut achha!"
Questions: "Samajh aaya?", "Batao, kya socha?", "Koi sawaal?"
English only for technical terms — not everyday words."""

    elif "tamil" in l:
        return """LANGUAGE — TAMIL + HINDI + ENGLISH MIX:
Use Tamil words: "nalla" (good), "enna" (what), "aamaa" (yes), "purinjucha?" (understood?),
"super!" (great), "romba nalla" (very good), "paaru" (look/see).
Mix: Tamil + Hindi + English technical terms naturally."""

    elif "telugu" in l:
        return """LANGUAGE — TELUGU + HINDI + ENGLISH MIX:
Use Telugu words: "bagundi" (good), "enti" (what), "cheppandi" (tell me), "artham ayinda?" (understood?),
"chala bagundi!" (very good!), "chudandi" (look/see).
Mix: Telugu + Hindi + English technical terms naturally."""

    elif "marathi" in l:
        return """LANGUAGE — MARATHI + HINDI + ENGLISH MIX:
Use Marathi words: "chan" (good), "kay" (what), "sangaa" (tell me), "samajle?" (understood?),
"shabas!" (bravo!), "khup chan!" (very good!), "bagha" (look/see).
Mix: Marathi + Hindi + English technical terms naturally."""

    elif "gujarati" in l:
        return """LANGUAGE — GUJARATI + HINDI + ENGLISH MIX:
Use Gujarati words: "saras" (good), "shu" (what), "samjayu?" (understood?),
"shabash!" (bravo!), "khub saras!" (very good!), "juo" (look/see).
Mix: Gujarati + Hindi + English technical terms naturally."""

    elif "kannada" in l:
        return """LANGUAGE — KANNADA + HINDI + ENGLISH MIX:
Use Kannada words: "chennagide" (good), "yenu" (what), "arthavaytu?" (understood?),
"shabash!" (bravo!), "tumba chennagide!" (very good!), "nodi" (look/see).
Mix: Kannada + Hindi + English technical terms naturally."""

    elif "malayalam" in l:
        return """LANGUAGE — MALAYALAM + HINDI + ENGLISH MIX:
Use Malayalam words: "nannayittundu" (good), "enthanu" (what), "manasilaayo?" (understood?),
"kalaakki!" (excellent!), "valare nannayittundu!" (very good!).
Mix: Malayalam + Hindi + English technical terms naturally."""

    else:
        return """LANGUAGE — INDIAN ENGLISH + HINDI MIX:
Speak warm Indian English with natural Hindi phrases mixed in.
Hindi phrases: "Dekho", "Samajh aaya?", "Bilkul sahi!", "Bahut achha!", "Koi doubt?"
Keep it conversational and Indian — not formal British/American English."""


def _system_prompt(session: dict) -> str:
    st   = session["student"]
    meta = session["meta"]
    tea  = session.get("teacher", {})

    name    = st.get("name", "beta")
    lang    = st.get("language", "Hinglish")
    subject = meta.get("subject", "")
    chapter = meta.get("chapter_title", "")
    book    = meta.get("book_name", "")
    board   = meta.get("board", "CBSE")
    cls     = meta.get("class_level", "")
    author  = meta.get("author", "")

    t_name    = tea.get("teacher_name", "Priya Ma'am")
    t_pattern = tea.get("teaching_pattern") or "Warm, patient, concept-first."
    t_persona = tea.get("persona_prompt") or "Speak naturally like a warm Indian school teacher."

    lang_rules = _lang_rules(lang)

    return f"""You are {t_name}, a brilliant and beloved Indian school teacher with 20+ years of experience.

━━━ STUDENT CONTEXT ━━━
Name: {name}  |  Class: {cls}  |  Board: {board}
Subject: {subject}  |  Book: {book}{f'  |  Author: {author}' if author else ''}
Chapter: {chapter}

━━━ {lang_rules}

━━━ YOUR PERSONALITY ━━━
{t_persona}
Teaching Style: {t_pattern}

━━━ HOW YOU TEACH ━━━
• SHORT sentences — max 2 lines per thought (this is SPOKEN audio, not text)
• Vary ENERGY: excited for new concepts, suspenseful before answers, warm always
• REAL Indian examples: chai, cricket, roti, rickshaw, mango, festivals, Bollywood
• "Yaad rakhna..." before KEY exam points (slow down, emphasise)
• After every explanation: check with "Samajh aaya?" or language equivalent
• Build SUSPENSE: "Toh batao... kya socha?" (pause effect)
• Playfully strict: "Dhyan se! Yeh exam mein zaroor aayega!"

━━━ EMOTIONAL REACTIONS ━━━
Correct answer  → CELEBRATE loudly: "AREY WAAH! Ekdum sahi! Mujhe pata tha!"
Almost right    → "Bahut close hai! Aur thoda socho..."
Wrong answer    → Warm: "Achha try! Sahi answer hai actually... [explain with analogy]"
Doubt asked     → "Bahut achha sawaal {name}! [answer enthusiastically with example]"
Quiet/shy       → "Koi baat nahi, aaram se. Main yahan hoon."

━━━ STRICT SPEECH RULES ━━━
1. NO bullet points, NO asterisks, NO markdown — PURE SPOKEN WORDS ONLY
2. NO robotic monotone — every sentence has ENERGY and EMOTION
3. Keep under 100 words per response — short bursts work better in audio
4. ALWAYS end with either a question OR a clear "Chalo aage badhte hain!"
5. Use {name}'s name at least once per response — personal connection matters"""

def teacher_say(session: dict, instruction: str, max_tokens: int = 350) -> str:
    """Ask GPT to generate teacher speech, maintaining conversation history."""
    history = session["teaching"].get("history", [])
    msgs = [{"role": "system", "content": _system_prompt(session)}]
    msgs += history[-14:]          # last 7 exchanges = 14 messages
    msgs.append({"role": "user", "content": instruction})
    reply = _gpt(msgs, max_tokens=max_tokens)
    # Update history
    session["teaching"].setdefault("history", [])
    session["teaching"]["history"].append({"role": "user",      "content": instruction})
    session["teaching"]["history"].append({"role": "assistant", "content": reply})
    return reply

# =============================================================================
# RESPONSE BUILDER
# =============================================================================
def make_resp(
    text: str,
    phase: str,
    next_step: str,
    session: dict,
    session_id: Optional[str] = None,
    should_listen: bool = False,
    extra_meta: Optional[dict] = None,
) -> dict:
    voice    = _pick_voice(session)
    lang     = session.get("student", {}).get("language", "Hinglish")
    tea      = session.get("teacher", {})
    meta     = session.get("meta", {})
    teaching = session.get("teaching", {})
    audio_url = generate_audio(text, voice=voice, lang=lang)

    return {
        "ok":                True,
        "session_id":        session_id,
        "text":              text,
        "tts_text":          clean_text(text),
        "audio_url":         audio_url,
        "phase":             phase,
        "next":              next_step,
        "should_listen":     should_listen,
        "student_name":      session["student"].get("name"),
        "preferred_language":session["student"].get("language"),
        "meta": {
            "teacher_name":          tea.get("teacher_name", ""),
            "teacher_code":          tea.get("teacher_code", ""),
            "teacher_gender":        tea.get("gender", "female"),
            "teacher_voice_id":      voice,
            "teacher_voice_provider":"openai",
            "teacher_base_language": tea.get("base_language", ""),
            "chapter_name":          meta.get("chapter_title", ""),
            "book_name":             meta.get("book_name", ""),
            "chapter_author":        meta.get("author", ""),
            "subject":               meta.get("subject", ""),
            "current_chunk_index":   teaching.get("chunk_idx", 0),
            "current_chunk_stage":   teaching.get("stage", ""),
            "current_chunk_title":   teaching.get("chunk_title", ""),
            "waiting_for_reply":     should_listen,
            "read_line_text":        teaching.get("read_line", ""),
            "explanation_text":      teaching.get("explain_line", ""),
            "bottom_ticker_text":    teaching.get("read_line", ""),
            **(extra_meta or {}),
        },
    }

# =============================================================================
# SESSION BOOTSTRAP  — pull DB data once at session start
# =============================================================================
def bootstrap(session: dict):
    """Load chapter data, teacher profile, and teaching chunks from DB."""
    meta       = session["meta"]
    chapter_id = meta.get("syllabus_chapter_id", "")

    # 1. Enrich chapter metadata
    if chapter_id:
        row = db_chapter_meta(chapter_id)
        if row:
            meta.setdefault("chapter_title", row.get("chapter_title", ""))
            meta.setdefault("book_name",     row.get("book_name", ""))
            meta.setdefault("author",        row.get("author", ""))
            meta.setdefault("subject",       row.get("subject", ""))
            meta.setdefault("board",         row.get("board", "CBSE"))

    # 2. Load teacher profile
    teacher = db_teacher(
        meta.get("subject", ""),
        meta.get("board", "CBSE"),
        str(meta.get("class_level", "")),
    )
    session["teacher"] = teacher if teacher else _default_teacher()

    # 3. Load live chunks (filter out encoded/garbage ones)
    all_chunks = db_live_chunks(chapter_id)
    teach_chunks = [
        c for c in all_chunks
        if c.get("stage_type") in {"READ", "EXPLAIN", "TEACH", "STORY"}
        and not is_krutidev(c.get("cleaned_text") or c.get("read_text") or "")
        and (
            not is_placeholder(c.get("explain_text"))
            or not is_placeholder(c.get("cleaned_text") or c.get("read_text") or "")
        )
    ]
    session["teaching"]["db_chunks"]   = teach_chunks
    session["teaching"]["db_all"]      = all_chunks

    # 4. Load quiz questions
    session["teaching"]["quiz_pool"]   = db_quiz_questions(chapter_id)

    print(f"[bootstrap] chapter_id={chapter_id} chunks={len(teach_chunks)} teacher={session['teacher'].get('teacher_name','?')}")

def _default_teacher() -> dict:
    return {
        "teacher_name":    "Priya Ma'am",
        "teacher_code":    "PRIYA_01",
        "gender":          "female",
        "voice_id":        "nova",
        "voice_provider":  "openai",
        "base_language":   "Hinglish",
        "persona_prompt":  "Warm, energetic, experienced Indian teacher. Mix Hindi and English naturally.",
        "teaching_pattern":"Read line → Explain simply → Indian example → Check understanding",
    }

# =============================================================================
# PHASE 1: INTRO  — Name, Language, School → personal connection
# =============================================================================
def phase_intro(session: dict, msg: str, sid: str) -> dict:
    step = session.get("step", "ASK_NAME")
    st   = session["student"]
    meta = session["meta"]

    # Guard empty/garbage input — re-ask the same question
    if not msg or msg.lower().strip() in {"undefined", "null", "next", "continue", ""}:
        reprompts = {
            "ASK_NAME":     "Hmm, main sun nahi paya! Apna naam phir se batao?",
            "ASK_LANGUAGE": "Kaunsi bhasha mein padhna chahoge? Hindi, English, ya Hinglish?",
            "ASK_SCHOOL":   "Aur school ka naam? Batao please!",
        }
        text = reprompts.get(step, "Zara phir se batao please!")
        return make_resp(text, "INTRO", step, session, sid, should_listen=True)

    # ── ASK_NAME ──────────────────────────────────────────────────────────
    if step == "ASK_NAME":
        name = msg.strip().title()
        st["name"] = name
        session["step"] = "ASK_LANGUAGE"

        text = teacher_say(session,
            f"Student introduced themselves as '{name}'. "
            "Greet them warmly with their name, say something nice about it, "
            "then ask which language they prefer: Hindi, English, Hinglish, or their regional language. "
            "Keep it friendly and fun — 3 sentences max.",
            max_tokens=120)
        session["teaching"]["stage"] = "ASK_LANGUAGE"
        return make_resp(text, "INTRO", "ASK_LANGUAGE", session, sid, should_listen=True)

    # ── ASK_LANGUAGE ──────────────────────────────────────────────────────
    if step == "ASK_LANGUAGE":
        lang = msg.strip()
        st["language"] = lang
        session["step"] = "ASK_SCHOOL"

        text = teacher_say(session,
            f"Student said they prefer '{lang}'. "
            "Confirm this enthusiastically in 1 sentence, then ask which school they study in.",
            max_tokens=80)
        session["teaching"]["stage"] = "ASK_SCHOOL"
        return make_resp(text, "INTRO", "ASK_SCHOOL", session, sid, should_listen=True)

    # ── ASK_SCHOOL ────────────────────────────────────────────────────────
    if step == "ASK_SCHOOL":
        school = msg.strip()
        st["school"] = school
        session["step"]  = "DONE"
        session["phase"] = "CHAPTER_INTRO"

        name    = st.get("name", "beta")
        lang    = st.get("language", "Hinglish")
        chapter = meta.get("chapter_title", "aaj ka chapter")
        subject = meta.get("subject", "")
        book    = meta.get("book_name", "")
        author  = meta.get("author", "")
        cls     = meta.get("class_level", "")
        board   = meta.get("board", "CBSE")

        # GPT generates a warm, personalised chapter intro hook
        hook = teacher_say(session,
            f"Student {name} from '{school}' is ready to study. "
            f"Now deliver a perfect chapter introduction for '{chapter}' ({subject}, {board} Class {cls}, book: {book}{', by ' + author if author else ''}). "
            "Do this in exactly this sequence: "
            "1) Celebrate that school warmly (1 short line). "
            "2) Build CURIOSITY with one surprising real-world fact or relatable situation connected to this chapter topic (2 lines). "
            "3) Tell them today's chapter name and what exciting things they'll learn (2 lines). "
            "4) End with: 'Toh chalte hain, shuru karte hain aaj ki class!' "
            f"Speak naturally in {lang}/Hinglish. Total: 5-6 sentences only.",
            max_tokens=250)

        # Find chapter overview from DB chunks if available
        db_all = session["teaching"].get("db_all", [])
        overview_chunk = next(
            (c for c in db_all if c.get("stage_type") == "CHAPTER_INTRO"), None)
        overview_text = ""
        if overview_chunk and not is_placeholder(overview_chunk.get("explain_text")):
            overview_text = overview_chunk.get("explain_text", "")

        session["teaching"]["stage"] = "chapter_intro"
        return make_resp(hook, "CHAPTER_INTRO", "START_TEACHING", session, sid,
            should_listen=False,
            extra_meta={
                "current_chunk_stage":   "chapter_intro",
                "chapter_overview_text": overview_text,
            })

    # Fallback
    session["phase"] = "TEACHING"
    return _start_teaching(session, sid)

# =============================================================================
# PHASE 2: CHAPTER INTRO  → transitions to TEACHING automatically
# =============================================================================
def phase_chapter_intro(session: dict, msg: str, sid: str) -> dict:
    session["phase"] = "TEACHING"
    session["teaching"]["chunk_idx"] = 0
    session["teaching"]["stage"]     = "read"
    return _start_teaching(session, sid)

def _start_teaching(session: dict, sid: str) -> dict:
    """Deliver a short 'let's begin' bridge before first chunk."""
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "chapter")
    subject = session["meta"].get("subject", "")

    bridge = teacher_say(session,
        f"Say a short 1-sentence 'let's begin' line for chapter '{chapter}' ({subject}). "
        "Energetic, warm, in Hinglish. Then immediately dive into the first chunk.",
        max_tokens=60)
    # Don't wait for student here — auto-advance to first chunk
    return _teach_next_chunk(session, sid, preamble=bridge)

# =============================================================================
# PHASE 3: TEACHING  — the core loop
# =============================================================================
def phase_teaching(session: dict, msg: str, sid: str) -> dict:
    teaching = session["teaching"]
    stage    = teaching.get("stage", "read")
    msg_low  = (msg or "").lower().strip()

    # ── Student answered the teacher's comprehension question ──────────────
    if stage == "waiting_answer":
        if not msg_low or msg_low in {"next","aage","skip","continue","chalo","pass"}:
            # Skipped — give quick answer then IMMEDIATELY go to next chunk
            lang_s    = session["student"].get("language", "Hinglish")
            skip_text = teacher_say(session,
                f"Student skipped. In 1 sentence of {lang_s}/Hinglish: "
                "give the correct answer briefly, then say 'Chalo aage badhte hain!'",
                max_tokens=70)
            teaching["stage"] = "read"
            teaching["xp"]    = teaching.get("xp", 0) + 5
            return _teach_next_chunk(session, sid, preamble=skip_text)

        return _eval_answer(session, msg, sid)

    # ── Student asked a doubt/question ────────────────────────────────────
    is_doubt = (
        "?" in msg and
        any(w in msg_low for w in [
            "kya","kyun","kaise","what","why","how","means","matlab",
            "samajh","doubt","explain","bata","pata nahi","confused",
            "nahi samjha","iska","ka matlab","difference","define",
        ])
    )
    if is_doubt:
        return _handle_doubt(session, msg, sid)

    # ── Continue signal ────────────────────────────────────────────────────
    continue_words = {"next","aage","chalo","ok","okay","haan","yes","ji",
                      "samjha","samajh gaya","samajh aayi","theek hai","got it",
                      "understood","clear","continue","aur batao","batao"}
    if msg_low in continue_words or not msg_low:
        return _teach_next_chunk(session, sid)

    # ── Emotional/general student message ─────────────────────────────────
    general_text = teacher_say(session,
        f"Student said: '{msg}'. "
        "Respond warmly and briefly (1-2 sentences), acknowledge what they said, "
        "then redirect back to the lesson naturally.",
        max_tokens=100)
    return make_resp(general_text, "TEACHING", "CONTINUE", session, sid,
        should_listen=False)

def _teach_next_chunk(session: dict, sid: str, preamble: str = "") -> dict:
    """Deliver the next DB chunk or GPT-generated teaching turn."""
    teaching   = session["teaching"]
    db_chunks  = teaching.get("db_chunks", [])
    idx        = teaching.get("chunk_idx", 0)
    meta       = session["meta"]
    st         = session["student"]
    name       = st.get("name", "beta")
    lang       = st.get("language", "Hinglish")
    chapter    = meta.get("chapter_title", "")
    subject    = meta.get("subject", "")

    # ── End of all DB chunks ───────────────────────────────────────────────
    if idx >= len(db_chunks):
        # If no DB chunks at all, use GPT to teach up to 8 parts
        if not db_chunks and idx < 8:
            return _gpt_teach_chunk(session, sid, idx, preamble)
        return _wrap_up(session, sid, preamble=preamble)

    chunk = db_chunks[idx]
    teaching["chunk_idx"]   = idx + 1
    teaching["chunk_title"] = f"Part {idx + 1}"

    # Extract fields
    raw_text   = (chunk.get("cleaned_text") or chunk.get("read_text") or "").strip()
    explain    = (chunk.get("explain_text") or "").strip()
    ask_text   = (chunk.get("ask_text") or "").strip()

    # Skip still-encoded chunks (shouldn't happen after bootstrap filter)
    if is_krutidev(raw_text) and is_krutidev(explain):
        teaching["chunk_idx"] = idx + 1
        return _teach_next_chunk(session, sid, preamble)

    parts = []
    if preamble:
        parts.append(preamble)

    # 1. READ the original line (only if it's clean Unicode)
    if raw_text and not is_krutidev(raw_text) and len(raw_text) > 20:
        read_line = raw_text[:350]
        teaching["read_line"] = read_line
        parts.append(f"Suno — '{read_line}'")
    else:
        teaching["read_line"] = ""

    # 2. EXPLAIN — use DB value if rich, otherwise GPT
    if not is_placeholder(explain) and not is_krutidev(explain):
        teaching["explain_line"] = explain
        parts.append(explain)
    else:
        gpt_explain = teacher_say(session,
            f"Just explained this line from '{chapter}' ({subject}) to {name}: "
            f"'{(raw_text or explain)[:300]}'. "
            f"Explain it simply in 2-3 sentences of {lang}/Hinglish. "
            "Use one relatable Indian daily-life example. Be warm and enthusiastic.",
            max_tokens=180)
        teaching["explain_line"] = gpt_explain
        parts.append(gpt_explain)

    # 3. ASK — DB question or GPT-generated
    if ask_text and not is_placeholder(ask_text) and len(ask_text) > 10:
        teaching["pending_question"] = ask_text
        parts.append(ask_text)
    else:
        gpt_q = _gpt([
            {"role": "system", "content": _system_prompt(session)},
            {"role": "user",   "content":
                f"After explaining '{(raw_text or explain)[:150]}' to {name}, "
                f"ask ONE short, clear comprehension question in {lang}/Hinglish. "
                "Make it simple, encouraging, and directly related to what was just explained. "
                "Just the question — no preamble."}
        ], max_tokens=70)
        teaching["pending_question"] = gpt_q
        parts.append(gpt_q)

    teaching["stage"]    = "waiting_answer"
    full_text = " ".join(p for p in parts if p)

    return make_resp(full_text, "TEACHING", "AWAIT_ANSWER", session, sid,
        should_listen=True,
        extra_meta={
            "current_chunk_stage": "question",
            "current_chunk_title": f"Part {idx + 1} / {len(db_chunks)}",
        })

def _gpt_teach_chunk(session: dict, sid: str, chunk_num: int, preamble: str = "") -> dict:
    """Pure GPT teaching when no DB chunks available."""
    meta    = session["meta"]
    st      = session["student"]
    name    = st.get("name", "beta")
    lang    = st.get("language", "Hinglish")
    chapter = meta.get("chapter_title", "")
    subject = meta.get("subject", "")
    book    = meta.get("book_name", "")
    cls     = meta.get("class_level", "")
    board   = meta.get("board", "CBSE")

    teaching = session["teaching"]
    teaching["chunk_idx"] = chunk_num + 1

    chunk_text = teacher_say(session,
        f"Teach chunk #{chunk_num + 1} of '{chapter}' ({subject}, {board} Class {cls}, {book}) to {name}. "
        "Follow this EXACT sequence in one response: "
        "1) State one key concept or sentence from this chapter part (quote-style). "
        "2) Explain it simply in 2 sentences with a real Indian daily-life example. "
        "3) Ask ONE clear comprehension question to check understanding. "
        f"Speak naturally in {lang}/Hinglish. Total: 5-6 sentences. End on the question.",
        max_tokens=300)

    teaching["stage"]    = "waiting_answer"
    teaching["chunk_title"] = f"Part {chunk_num + 1}"
    text = (preamble + " " + chunk_text).strip() if preamble else chunk_text

    return make_resp(text, "TEACHING", "AWAIT_ANSWER", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "question",
                    "current_chunk_title": f"Part {chunk_num + 1}"})

def _eval_answer(session: dict, student_answer: str, sid: str) -> dict:
    """Evaluate answer, give feedback, then IMMEDIATELY advance to next chunk.
    Combining both into one response eliminates the frontend auto-continue dependency."""
    teaching = session["teaching"]
    name     = session["student"].get("name", "beta")
    lang     = session["student"].get("language", "Hinglish")
    q        = teaching.get("pending_question", "the question")
    context  = teaching.get("read_line") or teaching.get("explain_line", "")

    teaching["questions_asked"] = teaching.get("questions_asked", 0) + 1
    teaching["xp"]              = teaching.get("xp", 0) + 20

    # Generate warm feedback (short — 1-2 sentences only)
    feedback = teacher_say(session,
        f"Student {name} answered: '{student_answer}'.\n"
        f"The question was: '{q}'\n"
        f"Context: '{context[:150]}'\n\n"
        f"In {lang}/Hinglish, give EXACTLY 1-2 sentences of warm feedback: "
        "If correct: celebrate loudly with their name, add 1 bonus fact. "
        "If wrong: gently say the right answer with a simple analogy. "
        "End with ONLY: 'Chalo, aage badhte hain!'",
        max_tokens=100)

    teaching["stage"] = "read"

    # Check if all chunks done
    db_chunks = teaching.get("db_chunks", [])
    idx       = teaching.get("chunk_idx", 0)
    teach_chunks = [c for c in db_chunks
                    if c.get("stage_type") in {"READ","EXPLAIN","TEACH","STORY"}
                    and not is_krutidev(c.get("cleaned_text") or c.get("read_text") or "")]

    if idx >= len(teach_chunks) and (teach_chunks or idx >= 6):
        # Chapter complete — return feedback + wrap up
        return _wrap_up(session, sid, preamble=feedback)

    # Immediately advance to next chunk with feedback as preamble
    return _teach_next_chunk(session, sid, preamble=feedback)

def _handle_doubt(session: dict, question: str, sid: str) -> dict:
    """Handle a student doubt mid-lesson."""
    name    = session["student"].get("name", "beta")
    lang    = session["student"].get("language", "Hinglish")
    meta    = session["meta"]

    answer = teacher_say(session,
        f"Student {name} has a doubt: '{question}'.\n"
        f"Subject: {meta.get('subject','')} | Chapter: {meta.get('chapter_title','')}\n\n"
        f"In {lang}/Hinglish (3 sentences max): "
        "1) Praise their curiosity warmly. "
        "2) Answer clearly using a real Indian daily-life analogy. "
        "3) End with: 'Ab samajh aaya? Koi aur doubt?'",
        max_tokens=200)

    return make_resp(answer, "TEACHING", "DOUBT_RESOLVED", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "doubt_answer"})

def _wrap_up(session: dict, sid: str, preamble: str = "") -> dict:
    """Chapter complete — summary and quiz offer."""
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "")
    xp      = session["teaching"].get("xp", 0)
    asked   = session["teaching"].get("questions_asked", 0)

    summary = teacher_say(session,
        f"The entire chapter '{chapter}' is now complete for {name}. "
        f"They answered {asked} questions and earned {xp} XP. "
        "In 3-4 sentences: "
        "1) Recap the 2-3 most important things learned today (specific to this chapter). "
        "2) Celebrate their effort genuinely and personally with their name. "
        "3) Say: 'Kya tum ek quick revision quiz dena chahoge?' "
        "Be very proud and energetic!",
        max_tokens=200)

    full_text = (preamble + " " + summary).strip() if preamble else summary
    session["phase"]             = "DONE"
    session["teaching"]["stage"] = "done"
    return make_resp(full_text, "DONE", "CHAPTER_COMPLETE", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "chapter_complete"})

# =============================================================================
# PHASE 4: QUIZ
# =============================================================================
def phase_quiz_ask(session: dict, sid: str) -> dict:
    """Ask the next quiz question."""
    teaching  = session["teaching"]
    pool      = teaching.get("quiz_pool", [])
    q_idx     = teaching.get("quiz_idx", 0)
    name      = session["student"].get("name", "beta")
    meta      = session["meta"]

    if q_idx >= len(pool) or not pool:
        # GPT-generated quiz question
        q_text = teacher_say(session,
            f"Ask {name} one clear revision question about '{meta.get('chapter_title','')}' "
            f"({meta.get('subject','')}). Make it interesting and challenging but fair. "
            "Hinglish, short, end with '— batao!'",
            max_tokens=80)
        teaching["quiz_answer"] = None
    else:
        q = pool[q_idx]
        q_text = (
            f"Quiz time {name}! Dhyan se suno. "
            f"{q.get('question_text','')} — socho aur batao!"
        )
        teaching["quiz_answer"] = q.get("correct_answer")

    teaching["quiz_idx"] = q_idx + 1
    teaching["stage"]    = "quiz_answer"

    return make_resp(q_text, "QUIZ", "AWAIT_QUIZ_ANSWER", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "quiz"})

def phase_quiz_eval(session: dict, msg: str, sid: str) -> dict:
    """Evaluate quiz answer and ask if they want another."""
    teaching = session["teaching"]
    correct  = teaching.get("quiz_answer")
    name     = session["student"].get("name", "beta")

    if correct:
        fb_prompt = (
            f"Student {name} answered '{msg}'. Correct answer: '{correct}'. "
            "2 sentences of Hinglish feedback — celebrate if right, gently correct if wrong. "
            "End with: 'Ek aur question?'"
        )
    else:
        fb_prompt = (
            f"Student {name} answered '{msg}'. "
            "Give 1-sentence encouraging feedback. End with: 'Ek aur question?'"
        )
    feedback = teacher_say(session, fb_prompt, max_tokens=120)
    teaching["stage"] = "quiz_wait"
    teaching["xp"]    = teaching.get("xp", 0) + 15
    return make_resp(feedback, "QUIZ", "NEXT_QUIZ_Q", session, sid,
        should_listen=True,
        extra_meta={"current_chunk_stage": "quiz_feedback"})

# =============================================================================
# FAREWELL
# =============================================================================
def phase_farewell(session: dict, sid: str) -> dict:
    name    = session["student"].get("name", "beta")
    chapter = session["meta"].get("chapter_title", "")
    xp      = session["teaching"].get("xp", 0)

    text = teacher_say(session,
        f"Say a warm, energetic farewell to {name} who completed '{chapter}' "
        f"and earned {xp} XP today. "
        "3 sentences: 1) Congratulate them specifically, "
        "2) Give 1 quick memory tip for this chapter, "
        "3) Say goodbye warmly.",
        max_tokens=150)
    return make_resp(text, "DONE", "FAREWELL", session, sid)

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.post("/start-session")
def start_session(req: StartSession):
    sid = str(uuid.uuid4())

    session: Dict[str, Any] = {
        "phase": "INTRO",
        "step":  "ASK_NAME",
        "student": {
            "name":     "",
            "language": req.language or "Hinglish",
            "school":   "",
        },
        "meta": {
            "board":               req.board or "CBSE",
            "class_level":         str(req.class_level or ""),
            "subject":             req.subject or "",
            "book_name":           req.book_name or "",
            "chapter_title":       req.chapter_title or req.chapter_name or "",
            "chapter_code":        req.chapter_code or "",
            "syllabus_chapter_id": req.syllabus_chapter_id or "",
            "author":              "",
        },
        "teacher":  {},
        "teaching": {
            "history":          [],
            "db_chunks":        [],
            "db_all":           [],
            "quiz_pool":        [],
            "chunk_idx":        0,
            "quiz_idx":         0,
            "stage":            "ASK_NAME",
            "read_line":        "",
            "explain_line":     "",
            "chunk_title":      "",
            "pending_question": "",
            "questions_asked":  0,
            "xp":               0,
        },
    }

    # Load DB data — non-blocking, errors are caught
    try:
        bootstrap(session)
    except Exception as e:
        print(f"[bootstrap] error: {e}")
        session["teacher"] = _default_teacher()

    SESSIONS[sid] = session

    # Opening greeting from teacher
    t_name  = session["teacher"].get("teacher_name", "Priya Ma'am")
    chapter = session["meta"].get("chapter_title", "")
    subject = session["meta"].get("subject", "")

    greeting = teacher_say(session,
        f"You are {t_name}. A new student is starting class for '{chapter}' ({subject}). "
        "Deliver a warm, energetic opening greeting: "
        "1) Introduce yourself briefly (1 sentence). "
        "2) Express excitement about today's class (1 sentence). "
        "3) Ask the student their name with warmth. "
        "Hinglish, natural, human. 3 sentences total.",
        max_tokens=120)

    session["teaching"]["stage"] = "ASK_NAME"
    return make_resp(greeting, "INTRO", "ASK_NAME", session, sid, should_listen=True)


@app.post("/student-reply")
def student_reply(req: StudentReply):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404,
            detail="Session not found. Please tap Start Class again.")

    session = SESSIONS[req.session_id]
    msg     = (req.message or "").strip()
    phase   = session.get("phase", "INTRO")

    # ── INTRO phase ────────────────────────────────────────────────────────
    if phase == "INTRO":
        return phase_intro(session, msg, req.session_id)

    # ── CHAPTER_INTRO phase ────────────────────────────────────────────────
    if phase == "CHAPTER_INTRO":
        return phase_chapter_intro(session, msg, req.session_id)

    # ── TEACHING phase ─────────────────────────────────────────────────────
    if phase == "TEACHING":
        return phase_teaching(session, msg, req.session_id)

    # ── DONE phase (after all chunks — quiz offer) ──────────────────────────
    if phase == "DONE":
        msg_low = msg.lower()
        want_quiz = any(w in msg_low for w in
            ["haan","yes","quiz","ok","sure","chalo","haan ji","bilkul","please"])
        if want_quiz:
            session["phase"] = "QUIZ"
            session["teaching"]["stage"] = "quiz_ask"
            return phase_quiz_ask(session, req.session_id)
        return phase_farewell(session, req.session_id)

    # ── QUIZ phase ─────────────────────────────────────────────────────────
    if phase == "QUIZ":
        stage = session["teaching"].get("stage", "")
        if stage == "quiz_answer":
            return phase_quiz_eval(session, msg, req.session_id)
        # After feedback — student said "yes more" or "no"
        msg_low = msg.lower()
        want_more = any(w in msg_low for w in
            ["haan","yes","ok","sure","chalo","aur","more","ek aur"])
        if want_more:
            session["teaching"]["stage"] = "quiz_ask"
            return phase_quiz_ask(session, req.session_id)
        return phase_farewell(session, req.session_id)

    # Fallback
    return make_resp("Chalo aage badhte hain!", phase, "CONTINUE",
        session, req.session_id)


@app.post("/student/homework-help")
def homework_help(req: HomeworkRequest):
    if req.session_id not in SESSIONS:
        raise HTTPException(status_code=404, detail="Session not found.")

    session = SESSIONS[req.session_id]
    name    = session["student"].get("name", "beta")
    lang    = session["student"].get("language", "Hinglish")
    meta    = session["meta"]

    text = teacher_say(session,
        f"Student {name} needs homework help: '{req.question}'\n"
        f"Subject: {req.subject or meta.get('subject','')} | "
        f"Chapter: {req.chapter_title or req.chapter_name or meta.get('chapter_title','')}\n\n"
        f"In {lang}/Hinglish (3 sentences): "
        "1) Acknowledge warmly. "
        "2) Give a HINT using a daily-life Indian analogy — do NOT give the direct answer. "
        "3) Say: 'Ab try karo — kya lagta hai tumhe?'",
        max_tokens=200)

    return make_resp(text,
        session.get("phase", "TEACHING"), "HOMEWORK_HELP",
        session, req.session_id, should_listen=True,
        extra_meta={"current_chunk_stage": "homework"})


@app.post("/tts")
def tts_endpoint(req: TTSRequest):
    clean = clean_text(req.text or "")
    if not clean:
        return JSONResponse({"ok": False, "error": "Empty text"}, status_code=400)

    # Determine voice
    voice = "nova"
    if req.voice and req.voice in _TTS_VOICE_MAP:
        voice = _TTS_VOICE_MAP[req.voice]
    elif req.teacher_gender:
        gender = req.teacher_gender.lower()
        voice  = _TTS_VOICE_MAP.get(gender, "nova")

    url = generate_audio(clean, voice=voice)
    if not url:
        return JSONResponse({"ok": False, "error": "TTS generation failed"}, status_code=500)

    return {
        "ok":        True,
        "audio_url": url,
        "provider":  "openai",
        "model":     "tts-1-hd",
        "voice":     voice,
    }


@app.get("/health")
def health():
    oai_ok = _get_openai() is not None
    sb_ok  = _get_sb() is not None
    return {
        "ok":                True,
        "service":           "GurukulAI Brain v4",
        "status":            "running",
        "render_domain":     RENDER_DOMAIN,
        "active_sessions":   len(SESSIONS),
        "audio_dir":         AUDIO_DIR,
        "audio_dir_exists":  os.path.isdir(AUDIO_DIR),
        "openai_ready":      oai_ok,
        "openai_key_set":    bool(OPENAI_KEY),
        "supabase_pkg":      _SUPABASE_OK,
        "supabase_ready":    sb_ok,
        "supabase_url_set":  bool(SB_URL),
        "supabase_key_set":  bool(SB_KEY),
        "python_version":    __import__("sys").version,
    }

def extract_text_from_pdf_url(pdf_url):
    import requests
    import tempfile
    import fitz
    import os

    response = requests.get(pdf_url)

    if response.status_code != 200:
        raise Exception(f"Failed to download PDF: {response.status_code}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    doc = fitz.open(tmp_path)

    text = ""
    for page in doc:
        text += page.get_text()

    doc.close()
    os.remove(tmp_path)

    return text

@app.post("/process-pdf")
def process_pdf(payload: dict):
    pdf_url = payload.get("pdf_url")
    board = payload.get("board", "CBSE")
    class_name = payload.get("class_name", "6")
    subject = payload.get("subject", "Unknown")
    chapter = payload.get("chapter", "Unknown")

    if not pdf_url:
        return {"ok": False, "error": "pdf_url is required"}

    # 1. Extract text
    text = extract_text_from_pdf_url(pdf_url)
    print("TEXT LENGTH:", len(text))

    # 2. Split
    paragraphs = split_into_paragraphs(text)
    print("CHUNKS:", len(paragraphs))

    # 3. Save to DB
    rows = []
    for i, para in enumerate(paragraphs):
        rows.append({
            "board": board,
            "class_name": class_name,
            "subject": subject,
            "chapter": chapter,
            "kind": "content",
            "idx": i,
            "text": para
        })

    sb = _get_sb()

    # delete old
    sb.table("chunks") \
        .delete() \
        .eq("board", board) \
        .eq("class_name", class_name) \
        .eq("subject", subject) \
        .eq("chapter", chapter) \
        .execute()

    # insert new
    sb.table("chunks").insert(rows).execute()

    return {
        "ok": True,
        "chunks_saved": len(rows)
    }

@app.post("/bulk-process")
def bulk_process():
    sb = _get_sb()

    BASE_URL = "https://zvfebuoasoomeanevcjz.supabase.co/storage/v1/object/public/gurukulai-private/CBSE"

    classes = ["5", "6", "7", "8", "9", "10"]
    subjects = ["science", "hindi", "english"]

    total_chunks = 0
    processed_files = []

    for cls in classes:
        for subject in subjects:

            # 👇 manually define your PDFs (you can expand later)
            pdfs = [
                "chapter1.pdf",
                "chapter2.pdf",
                "chapter3.pdf"
            ]

            for pdf in pdfs:
                pdf_url = f"{BASE_URL}/class-{cls}/{subject}/{pdf}"

                try:
                    text = extract_text_from_pdf_url(pdf_url)
                    paragraphs = split_into_paragraphs(text)

                    rows = []
                    for i, para in enumerate(paragraphs):
                        rows.append({
                            "board": "CBSE",
                            "class_name": cls,
                            "subject": subject.capitalize(),
                            "chapter": pdf.replace(".pdf", ""),
                            "kind": "content",
                            "idx": i,
                            "text": para
                        })

                    # delete old
                    sb.table("chunks") \
                        .delete() \
                        .eq("board", "CBSE") \
                        .eq("class_name", cls) \
                        .eq("subject", subject.capitalize()) \
                        .eq("chapter", pdf.replace(".pdf", "")) \
                        .execute()

                    # insert
                    sb.table("chunks").insert(rows).execute()

                    total_chunks += len(rows)
                    processed_files.append(pdf_url)

                    print(f"✅ Done: {pdf_url} → {len(rows)} chunks")

                except Exception as e:
                    print(f"❌ Failed: {pdf_url}", str(e))

    return {
        "ok": True,
        "files_processed": len(processed_files),
        "total_chunks": total_chunks
    }

# ===== AUTO CHUNK GENERATOR IMPORTS =====
def split_into_paragraphs(text, chunk_size=400):
    chunks = []
    current = ""

    for line in text.split("\n"):
        line = line.strip()

        if not line:
            continue

        if len(current) + len(line) < chunk_size:
            current += " " + line
        else:
            chunks.append(current.strip())
            current = line

    if current:
        chunks.append(current.strip())

    return chunks
    
def generate_chunks_from_paragraphs(paragraphs, chapter_code):
    chunks = []
    chunk_id = 1

    # ================== STEP 3 (ADD HERE) ==================
    chunks.append({
        "chunk_id": chunk_id,
        "type": "intro",
        "text": f"Namaste beta 😊 aaj hum {chapter_code} padhne wale hain. Dhyaan se suno.",
        "question": "Kya tum ready ho?"
    })
    chunk_id += 1
    # ======================================================

    # 🔁 MAIN LOOP
    for para in paragraphs:
        if len(para) < 40:
            continue

        prompt = f"""
You are a friendly Indian school teacher teaching a Class 6 student.

Explain the following paragraph in:
- Simple Hindi + light Hinglish
- Very short sentences (for voice)
- Add emotion (like "achha beta", "dhyaan se suno")
- First explain concept
- Then give 1 simple example
- Then ask 1 thoughtful question (not generic)

Paragraph:
{para}
"""

        explanation = teacher_say(
            {"student": {"name": "beta"}},
            prompt
        )

        chunks.append({
            "chunk_id": chunk_id,
            "type": "teaching",
            "text": explanation,
            "question": extract_question(explanation)
        })

        chunk_id += 1

        if chunk_id > 12:
            break

    # ================== STEP 4 (RECAP) ==================
    chunks.append({
        "chunk_id": chunk_id,
        "type": "recap",
        "text": "Toh beta, aaj humne important concepts seekhe. Thoda revise karna zaroori hai.",
        "question": "Aaj tumne sabse important kya seekha?"
    })
    # ===================================================

    return chunks

def extract_question(text):
    lines = text.split("\n")
    
    for line in reversed(lines):
        if "?" in line:
            return line.strip()
    
    return "Achha batao, tumne kya samjha?"

def save_chunks_to_file(chapter_code, chunks):
    path = f"./content/{chapter_code}.json"

    os.makedirs("./content", exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "chapter_code": chapter_code,
            "chunks": chunks
        }, f, ensure_ascii=False, indent=2)

def process_full_book(base_path):
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                print(f"Processing: {pdf_path}")

                text = extract_text_from_pdf_url(pdf_url)
                chunks = create_chunks(text)

                # ✅ ADD THIS BLOCK HERE
                from supabase import create_client
                supabase = create_client(SB_URL, SB_KEY)

                parts = pdf_path.replace("\\", "/").split("/")

                board = parts[-6]
                class_name = parts[-5]
                subject = parts[-4]
                book = parts[-3]
                chapter = os.path.splitext(parts[-1])[0]

                for i, chunk in enumerate(chunks):
                    supabase.table("chunks").insert({
                        "board": board,
                        "class_name": class_name,
                        "subject": subject,
                        "chapter": chapter,
                        "kind": "learning",
                        "idx": i,
                        "text": chunk["text"]
                    }).execute()

                print(f"✅ Saved {len(chunks)} chunks")
