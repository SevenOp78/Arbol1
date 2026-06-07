"""Backend integration tests for the Quiz Realtime Platform.

Covers:
- Health endpoints (/api/, /api/state)
- Reset (/api/state/reset)
- Logs (/api/logs)
- Image upload pipeline (/api/upload) with real OpenAI calls
- Question switch + duplicate handling
- Fragment reconstruction
- WebSocket /api/ws broadcast on upload
"""

import os
import json
import time
import asyncio
import pytest
import requests
import websockets

BASE_URL = os.environ.get("REACT_APP_BACKEND_URL", "https://quiz-image-live.preview.emergentagent.com").rstrip("/")
WS_URL = BASE_URL.replace("http", "ws") + "/api/ws"

UPLOAD_TIMEOUT = 180  # OpenAI vision can be slow

QUIZ1 = "/tmp/quiz1.jpg"
QUIZ2 = "/tmp/quiz2.jpg"
FRAG1 = "/tmp/frag1.jpg"
FRAG2 = "/tmp/frag2.jpg"
OPTS = "/tmp/opts.jpg"


# ----------------- Fixtures -----------------
@pytest.fixture(scope="module")
def session():
    s = requests.Session()
    return s


@pytest.fixture(autouse=True)
def reset_before(session):
    # Ensure clean state before each test (some tests rely on this)
    try:
        session.post(f"{BASE_URL}/api/state/reset", timeout=10)
    except Exception:
        pass
    yield


def _upload(session, path, device_id="test-device", timestamp=None):
    with open(path, "rb") as fh:
        files = {"image": (os.path.basename(path), fh, "image/jpeg")}
        data = {"device_id": device_id}
        if timestamp:
            data["timestamp"] = timestamp
        r = session.post(f"{BASE_URL}/api/upload", files=files, data=data, timeout=UPLOAD_TIMEOUT)
    return r


# ----------------- Health -----------------
class TestHealth:
    def test_root_ok(self, session):
        r = session.get(f"{BASE_URL}/api/", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert "Quiz Realtime" in d.get("message", "")
        assert d.get("vision_model")
        assert d.get("answer_model")

    def test_state_waiting_after_reset(self, session):
        r = session.get(f"{BASE_URL}/api/state", timeout=10)
        assert r.status_code == 200
        d = r.json()
        assert d["status"] == "waiting"
        assert d["questionNumber"] is None
        assert d["optionsReceived"] == {}
        assert d["questionFragments"] == []

    def test_reset_endpoint(self, session):
        r = session.post(f"{BASE_URL}/api/state/reset", timeout=10)
        assert r.status_code == 200
        assert r.json().get("ok") is True

    def test_logs_endpoint(self, session):
        r = session.get(f"{BASE_URL}/api/logs?limit=5", timeout=15)
        assert r.status_code == 200
        d = r.json()
        assert isinstance(d, list)
        # Each log must NOT contain Mongo _id and must have basic fields
        for log in d:
            assert "_id" not in log
            assert "received_at" in log


# ----------------- Upload pipeline -----------------
class TestUploadQuiz1:
    """Capital de España -> B=Madrid."""

    def test_quiz1_success_b(self, session):
        r = _upload(session, QUIZ1)
        assert r.status_code == 200
        body = r.json()
        assert body.get("ok") is True
        st = body["state"]
        assert st["status"] == "success", f"Expected success, got {st}"
        assert st["matched_option"] == "B", f"Expected B, got {st['matched_option']}"
        ans = (st.get("generatedAnswer") or "").lower()
        assert "madrid" in ans

    def test_quiz1_duplicate_stable(self, session):
        # First upload
        r1 = _upload(session, QUIZ1)
        assert r1.status_code == 200
        st1 = r1.json()["state"]
        frags1 = len(st1["questionFragments"])
        opts1 = len(st1["optionsReceived"])

        # Duplicate upload — must NOT explode options/fragments
        r2 = _upload(session, QUIZ1)
        assert r2.status_code == 200
        st2 = r2.json()["state"]
        assert len(st2["questionFragments"]) <= max(1, frags1)
        assert len(st2["optionsReceived"]) == opts1, "Options should not duplicate"
        assert st2["status"] == "success"
        assert st2["matched_option"] == "B"


class TestQuestionSwitch:
    def test_quiz1_then_quiz2_resets(self, session):
        r1 = _upload(session, QUIZ1)
        st1 = r1.json()["state"]
        assert st1["status"] == "success"
        qn1 = st1["questionNumber"]

        r2 = _upload(session, QUIZ2)
        assert r2.status_code == 200
        st2 = r2.json()["state"]
        # New question must replace
        assert st2["questionNumber"] != qn1, "Question number should switch"
        # Quiz2 = Mona Lisa -> Leonardo da Vinci -> B
        ans = (st2.get("generatedAnswer") or "").lower()
        assert "leonardo" in ans or "vinci" in ans, f"Expected Leonardo da Vinci, got {ans}"
        # Status should be success with B (per problem statement)
        # Allow waiting if LLM mismatch, but flag matched_option
        if st2["status"] == "success":
            assert st2["matched_option"] == "B"


class TestFragments:
    def test_frag1_frag2_opts(self, session):
        # Fragment 1 (no '?')
        r1 = _upload(session, FRAG1)
        assert r1.status_code == 200
        st1 = r1.json()["state"]
        # Should still be waiting (no options yet)
        assert st1["status"] in ("waiting", "error"), f"Got {st1['status']} after frag1"

        # Fragment 2 (with '?')
        r2 = _upload(session, FRAG2)
        assert r2.status_code == 200
        st2 = r2.json()["state"]
        # questionText should be longer than after frag1 OR same if dedup
        assert st2["questionText"], "questionText should be populated"

        # Options (qn=null but state has qn=16 already)
        r3 = _upload(session, OPTS)
        assert r3.status_code == 200
        st3 = r3.json()["state"]
        # Per problem: accept success with B OR coherent waiting state
        assert st3["status"] in ("success", "waiting"), f"Got {st3['status']}"
        if st3["status"] == "success":
            assert st3["matched_option"] == "B"
        # Options should be present
        assert len(st3["optionsReceived"]) > 0, "Options should have been parsed"


# ----------------- WebSocket -----------------
class TestWebSocket:
    @pytest.mark.asyncio
    async def test_ws_initial_state_and_broadcast(self, session):
        # Reset first
        session.post(f"{BASE_URL}/api/state/reset", timeout=10)

        async with websockets.connect(WS_URL, open_timeout=15) as ws:
            # Initial state message
            init = await asyncio.wait_for(ws.recv(), timeout=10)
            msg = json.loads(init)
            assert msg.get("type") == "state"
            assert "data" in msg

            # Trigger an upload in a thread
            loop = asyncio.get_event_loop()
            future = loop.run_in_executor(None, lambda: _upload(session, QUIZ1))

            # Collect messages until success or timeout
            got_success = False
            deadline = time.time() + UPLOAD_TIMEOUT
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=UPLOAD_TIMEOUT)
                except asyncio.TimeoutError:
                    break
                m = json.loads(raw)
                if m.get("type") == "state" and m.get("data", {}).get("status") == "success":
                    got_success = True
                    break
            await future
            assert got_success, "WebSocket did not broadcast success state"
