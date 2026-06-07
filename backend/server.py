import os
import json
import base64
import re
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

from fastapi import FastAPI, APIRouter, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
from unidecode import unidecode
from rapidfuzz import fuzz
from openai import AsyncOpenAI

# ----------------------- Setup -----------------------
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

mongo_url = os.environ["MONGO_URL"]
db_client = AsyncIOMotorClient(mongo_url)
db = db_client[os.environ["DB_NAME"]]

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
VISION_MODEL = os.environ.get("OPENAI_MODEL_VISION", "gpt-5-mini")
ANSWER_MODEL = os.environ.get("OPENAI_MODEL_ANSWER", "gpt-5")

openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

app = FastAPI()
api_router = APIRouter(prefix="/api")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("quiz")


# ----------------------- State -----------------------
class QuestionState:
    def __init__(self):
        self.reset()

    def reset(self, question_number: Optional[int] = None):
        self.questionNumber: Optional[int] = question_number
        self.questionFragments: List[str] = []
        self.questionText: str = ""
        self.questionComplete: bool = False
        self.generatedAnswer: Optional[str] = None
        self.aliases: List[str] = []
        self.confidence: Optional[float] = None
        self.optionsReceived: Dict[str, str] = {}
        self.matched_option: Optional[str] = None
        self.match_confidence: Optional[float] = None
        self.status: str = "waiting"  # waiting | success | error
        self.error_message: Optional[str] = None
        self.updated_at: datetime = datetime.now(timezone.utc)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "questionNumber": self.questionNumber,
            "questionFragments": list(self.questionFragments),
            "questionText": self.questionText,
            "questionComplete": self.questionComplete,
            "generatedAnswer": self.generatedAnswer,
            "aliases": list(self.aliases),
            "confidence": self.confidence,
            "optionsReceived": dict(self.optionsReceived),
            "matched_option": self.matched_option,
            "match_confidence": self.match_confidence,
            "status": self.status,
            "error_message": self.error_message,
            "updated_at": self.updated_at.isoformat(),
        }


state = QuestionState()
state_lock = asyncio.Lock()


# ----------------------- Normalization & dedup -----------------------
def normalize(s: str) -> str:
    if not s:
        return ""
    s = unidecode(s).lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def is_duplicate_fragment(new_text: str, existing: List[str], threshold: int = 88) -> bool:
    nn = normalize(new_text)
    if not nn:
        return True
    for ex in existing:
        ne = normalize(ex)
        if not ne:
            continue
        if nn == ne or nn in ne or ne in nn:
            return True
        if fuzz.ratio(nn, ne) >= threshold:
            return True
    return False


def is_duplicate_option(letter: str, text: str, existing: Dict[str, str], threshold: int = 88) -> bool:
    if letter in existing:
        ne = normalize(existing[letter])
        nn = normalize(text)
        if nn == ne or fuzz.ratio(nn, ne) >= threshold:
            return True
    return False


def code_compare(candidate: str, aliases: List[str], option_text: str) -> Optional[Dict]:
    """Fast code-level comparison. Returns dict with match+confidence, or None if ambiguous."""
    if not candidate:
        return None
    nc = normalize(candidate)
    no = normalize(option_text)
    if not nc or not no:
        return None

    candidates_norm = [nc] + [normalize(a) for a in (aliases or []) if a]
    candidates_norm = [c for c in candidates_norm if c]

    best = 0
    for c in candidates_norm:
        if c == no:
            return {"match": True, "confidence": 1.0}
        if c in no or no in c:
            return {"match": True, "confidence": 0.92}
        r = max(fuzz.token_set_ratio(c, no), fuzz.partial_ratio(c, no))
        if r > best:
            best = r

    if best >= 88:
        return {"match": True, "confidence": best / 100.0}
    if best <= 45:
        return {"match": False, "confidence": (100 - best) / 100.0}
    return None  # ambiguous


# ----------------------- OpenAI helpers -----------------------
def _extract_json(text: str) -> Any:
    if not text:
        return None
    try:
        return json.loads(text)
    except Exception:
        pass
    # strip code fences
    cleaned = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _looks_like_complete_question(text: str) -> bool:
    """Heuristic: True only if text both opens with an interrogative marker AND ends with '?'."""
    if not text:
        return False
    t = text.strip()
    if not t.endswith("?"):
        return False
    norm = unidecode(t).lower().lstrip("¿ ").strip()
    openers = (
        "que ", "qué ", "cual ", "cuál ", "como ", "cómo ", "cuando ", "cuándo ",
        "donde ", "dónde ", "por que", "por qué", "porque ", "quien ", "quién ",
        "cuanto ", "cuánto ", "cuanta ", "cuánta ", "cuantos ", "cuántos ", "cuantas ", "cuántas ",
    )
    # Either original starts with ¿ OR normalized starts with a known opener
    return t.startswith("¿") or any(norm.startswith(o) for o in openers)


async def ai_extract_from_image(image_b64: str, mime: str) -> Dict:
    """GPT-5 Mini vision: extract questionNumber, type, text, options."""
    if not openai_client:
        return {"questionNumber": None, "items": [], "error": "no_api_key"}

    system = (
        "Eres un experto en extraer contenido de cuestionarios desde imágenes (preguntas numeradas y opciones). "
        "Devuelve ÚNICAMENTE JSON válido, sin texto adicional ni markdown."
    )
    instr = (
        "Analiza la imagen y devuelve este esquema exacto:\n"
        "{\n"
        '  "questionNumber": <int o null>,\n'
        '  "items": [\n'
        "    {\n"
        '      "type": "question_complete" | "question_fragment" | "option" | "unknown",\n'
        '      "text": "<texto, sin la letra de opción ni el número de pregunta>",\n'
        '      "option": "<letra A-E, sólo si type=option, o null>"\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Reglas:\n"
        "- questionNumber: número visible que prefija la pregunta (ej. '15. ¿...?' → 15). null si no es visible.\n"
        "- type='question_complete': SÓLO si la pregunta es autocontenida — empieza con apertura interrogativa (¿, '¿Cuál', '¿Qué', '¿Cómo', '¿Cuándo', '¿Dónde', '¿Por qué', '¿Quién', '¿Cuánto') Y termina con '?'. Si falta el inicio (sólo se ve la parte final con '?'), es un fragmento, no completa.\n"
        "- type='question_fragment': parte parcial de una pregunta. Incluye:\n"
        "    * un trozo inicial sin '?',\n"
        "    * un trozo final que termine con '?' pero SIN apertura interrogativa al principio,\n"
        "    * cualquier porción que claramente continúa otra frase.\n"
        "- type='option': es una opción tipo 'A) Madrid', '(B) París'. Extrae letra en 'option' y texto sin la letra en 'text'.\n"
        "- type='unknown': si no estás seguro.\n"
        "- Una imagen puede contener varios items (pregunta + varias opciones, varias opciones solas, etc.).\n"
        "- NO incluyas el número de pregunta dentro de 'text'. NO incluyas la letra dentro de 'text'.\n"
        "- Devuelve SÓLO el JSON, sin explicaciones."
    )

    try:
        resp = await openai_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instr},
                        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
                    ],
                },
            ],
        )
        content = resp.choices[0].message.content or ""
        data = _extract_json(content)
        if not data:
            return {"questionNumber": None, "items": [], "raw": content, "error": "json_parse_failed"}
        if "items" not in data or not isinstance(data["items"], list):
            data["items"] = []
        return data
    except Exception as e:
        logger.exception("ai_extract_from_image failed")
        return {"questionNumber": None, "items": [], "error": f"openai_error: {e}"}


async def ai_answer_question(question_text: str) -> Dict:
    """GPT-5: generate answer + aliases + confidence."""
    if not openai_client:
        return {"answer": None, "aliases": [], "confidence": 0.0, "error": "no_api_key"}

    system = (
        "Eres un experto académico que responde preguntas de cuestionarios de forma concisa. "
        "Devuelve ÚNICAMENTE JSON válido."
    )
    instr = (
        f"Pregunta: {question_text}\n\n"
        "Devuelve EXACTAMENTE este JSON:\n"
        '{ "answer": "<respuesta breve y directa>", "aliases": ["<sinónimo1>", "<sinónimo2>"], "confidence": <0.0-1.0> }\n'
        "- 'answer' debe ser corto y directo (ej. 'Madrid', 'Imperio Romano', '1789', 'James Watt').\n"
        "- 'aliases' debe incluir 2-5 formas equivalentes (nombres alternativos, variantes, sinónimos).\n"
        "- Sólo JSON, sin explicaciones."
    )

    try:
        resp = await openai_client.chat.completions.create(
            model=ANSWER_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instr},
            ],
        )
        content = resp.choices[0].message.content or ""
        data = _extract_json(content)
        if not data:
            return {"answer": None, "aliases": [], "confidence": 0.0, "error": "json_parse_failed", "raw": content}
        return {
            "answer": data.get("answer"),
            "aliases": data.get("aliases") or [],
            "confidence": float(data.get("confidence") or 0.0),
        }
    except Exception as e:
        logger.exception("ai_answer_question failed")
        return {"answer": None, "aliases": [], "confidence": 0.0, "error": f"openai_error: {e}"}


async def ai_semantic_compare(candidate: str, option_text: str) -> Dict:
    """GPT-5 Mini: decide if candidate answer matches option semantically."""
    if not openai_client:
        return {"match": False, "confidence": 0.0}

    system = "Eres un comparador semántico estricto. Devuelve sólo JSON."
    instr = (
        f"¿La 'respuesta candidata' representa la misma idea/entidad que la 'opción'?\n\n"
        f"Respuesta candidata: {candidate}\n"
        f"Opción: {option_text}\n\n"
        'Devuelve EXACTAMENTE: { "match": true|false, "confidence": <0.0-1.0> }. Sólo JSON.'
    )

    try:
        resp = await openai_client.chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": instr},
            ],
        )
        content = resp.choices[0].message.content or ""
        data = _extract_json(content)
        if not data:
            return {"match": False, "confidence": 0.0}
        return {
            "match": bool(data.get("match", False)),
            "confidence": float(data.get("confidence") or 0.0),
        }
    except Exception:
        logger.exception("ai_semantic_compare failed")
        return {"match": False, "confidence": 0.0}


# ----------------------- WebSocket manager -----------------------
class ConnectionManager:
    def __init__(self):
        self.connections: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.connections.append(ws)
        try:
            await ws.send_json({"type": "state", "data": state.to_dict()})
        except Exception:
            pass

    def disconnect(self, ws: WebSocket):
        if ws in self.connections:
            self.connections.remove(ws)

    async def broadcast(self, message: dict):
        dead = []
        for c in list(self.connections):
            try:
                await c.send_json(message)
            except Exception:
                dead.append(c)
        for d in dead:
            self.disconnect(d)


manager = ConnectionManager()


async def broadcast_state(extra: Optional[dict] = None):
    msg: Dict[str, Any] = {"type": "state", "data": state.to_dict()}
    if extra:
        msg.update(extra)
    await manager.broadcast(msg)


# ----------------------- Processing pipeline -----------------------
async def evaluate_options() -> None:
    """Compare each option against generated answer; update state if a match is found."""
    if not state.generatedAnswer or not state.optionsReceived:
        return

    for letter, opt_text in list(state.optionsReceived.items()):
        result = code_compare(state.generatedAnswer, state.aliases, opt_text)
        if result is None:
            result = await ai_semantic_compare(state.generatedAnswer, opt_text)
        if result.get("match"):
            state.matched_option = letter
            state.match_confidence = float(result.get("confidence", 0.0))
            state.status = "success"
            return

    # no match yet
    state.matched_option = None
    state.match_confidence = None


async def process_extracted(extracted: Dict, log_entry: Dict) -> None:
    items = extracted.get("items") or []
    qn_top = extracted.get("questionNumber")
    item_qns = [i.get("questionNumber") for i in items if i.get("questionNumber")]
    qn = qn_top or (item_qns[0] if item_qns else None)

    async with state_lock:
        # Question number switch handling
        if qn is not None:
            if state.questionNumber is None:
                state.reset(question_number=qn)
            elif qn != state.questionNumber:
                logger.info("Question changed %s -> %s, resetting state.", state.questionNumber, qn)
                state.reset(question_number=qn)

        had_question_change = False

        for it in items:
            t = it.get("type")
            txt = (it.get("text") or "").strip()
            if not txt:
                continue

            if t == "question_complete":
                if state.questionNumber is None and qn is None:
                    continue
                # Safety: downgrade to fragment if it doesn't look like a true complete question
                if not _looks_like_complete_question(txt):
                    t = "question_fragment"
                else:
                    if (not state.questionComplete) or (len(txt) > len(state.questionText)):
                        old = state.questionText
                        state.questionFragments = [txt]
                        state.questionText = txt
                        state.questionComplete = True
                        if normalize(old) != normalize(txt):
                            had_question_change = True
                    # already handled this item
                    continue

            if t == "question_fragment":
                if state.questionNumber is None and qn is None:
                    continue
                if state.questionComplete:
                    continue
                if is_duplicate_fragment(txt, state.questionFragments):
                    continue
                state.questionFragments.append(txt)
                state.questionText = " ".join(state.questionFragments).strip()
                had_question_change = True

            elif t == "option":
                letter = (it.get("option") or "").strip().upper()
                if not letter:
                    continue
                letter = letter[0]
                if not letter.isalpha():
                    continue
                if is_duplicate_option(letter, txt, state.optionsReceived):
                    continue
                state.optionsReceived[letter] = txt

        state.updated_at = datetime.now(timezone.utc)

        # Detect lack of question number
        if state.questionNumber is None and not state.optionsReceived and not state.questionText:
            state.status = "error"
            state.error_message = "No se detectó número de pregunta."
            return

        # Generate answer only when the question is actually formed enough.
        question_is_ready = (
            state.questionComplete
            or _looks_like_complete_question(state.questionText)
            or len(state.optionsReceived) > 0
        )
        if had_question_change and state.questionText and len(state.questionText) > 6 and question_is_ready:
            ans = await ai_answer_question(state.questionText)
            state.generatedAnswer = ans.get("answer")
            state.aliases = ans.get("aliases") or []
            state.confidence = ans.get("confidence") or 0.0
            # invalidate previous match — must re-evaluate
            state.matched_option = None
            state.match_confidence = None
            if not state.generatedAnswer:
                state.status = "error"
                state.error_message = "Error de IA al generar la respuesta."
                return

        # Evaluate options against answer
        if state.generatedAnswer and state.optionsReceived:
            await evaluate_options()
            if state.matched_option is None:
                state.status = "waiting"
                state.error_message = None
        elif state.questionNumber is not None:
            state.status = "waiting"
            state.error_message = None

    await broadcast_state({"log": log_entry})


# ----------------------- HTTP endpoints -----------------------
@api_router.get("/")
async def root():
    return {"message": "Quiz Realtime Platform - Backend Active", "vision_model": VISION_MODEL, "answer_model": ANSWER_MODEL}


@api_router.get("/state")
async def get_state():
    return state.to_dict()


@api_router.post("/state/reset")
async def reset_state():
    async with state_lock:
        state.reset()
    await broadcast_state()
    return {"ok": True}


@api_router.get("/logs")
async def get_logs(limit: int = 50):
    cursor = db.image_logs.find({}, {"_id": 0}).sort("received_at", -1).limit(limit)
    return await cursor.to_list(limit)


@api_router.post("/upload")
async def upload_image(
    image: UploadFile = File(...),
    timestamp: Optional[str] = Form(None),
    device_id: Optional[str] = Form(None),
):
    started = datetime.now(timezone.utc)
    try:
        raw = await image.read()
    except Exception as e:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"read_failed: {e}"})

    if not raw:
        return JSONResponse(status_code=400, content={"ok": False, "error": "empty_image"})

    mime = (image.content_type or "image/jpeg").lower()
    if mime not in ("image/jpeg", "image/png", "image/webp", "image/jpg"):
        mime = "image/jpeg"
    if mime == "image/jpg":
        mime = "image/jpeg"

    b64 = base64.b64encode(raw).decode("ascii")

    log_entry: Dict[str, Any] = {
        "received_at": started.isoformat(),
        "device_id": device_id,
        "client_timestamp": timestamp,
        "size_bytes": len(raw),
        "mime": mime,
        "filename": image.filename,
    }

    try:
        extracted = await ai_extract_from_image(b64, mime)
        log_entry["extracted"] = extracted
    except Exception as e:
        log_entry["error"] = f"extract_failed: {e}"
        async with state_lock:
            state.status = "error"
            state.error_message = f"OCR/IA falló: {e}"
        await broadcast_state({"log": log_entry})
        try:
            await db.image_logs.insert_one(dict(log_entry))
        except Exception:
            logger.exception("Failed to persist image log (after error)")
        return JSONResponse(status_code=200, content={"ok": False, "error": "extract_failed", "log": log_entry})

    if extracted.get("error") and not extracted.get("items"):
        async with state_lock:
            state.status = "error"
            state.error_message = "OCR/IA no pudo procesar la imagen."
        await broadcast_state({"log": log_entry})
    else:
        await process_extracted(extracted, log_entry)

    try:
        await db.image_logs.insert_one(dict(log_entry))
    except Exception:
        logger.exception("Failed to persist image log")

    return {"ok": True, "extracted": extracted, "state": state.to_dict()}


@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)
    except Exception:
        manager.disconnect(ws)


# ----------------------- App wiring -----------------------
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("shutdown")
async def shutdown_db_client():
    db_client.close()
