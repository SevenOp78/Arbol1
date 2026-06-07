import { useEffect, useMemo, useRef, useState, useCallback } from "react";
import "@/App.css";
import { Sheet, SheetContent, SheetHeader, SheetTitle, SheetTrigger } from "@/components/ui/sheet";
import { Terminal, RotateCcw } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;
const WS_URL = (() => {
  if (!BACKEND_URL) return null;
  return BACKEND_URL.replace(/^http/, "ws") + "/api/ws";
})();

function classNames(...xs) {
  return xs.filter(Boolean).join(" ");
}

function relTime(iso) {
  if (!iso) return "";
  try {
    const d = new Date(iso);
    const diff = (Date.now() - d.getTime()) / 1000;
    if (diff < 5) return "ahora";
    if (diff < 60) return `${Math.floor(diff)}s`;
    if (diff < 3600) return `${Math.floor(diff / 60)}m`;
    return d.toLocaleTimeString();
  } catch {
    return "";
  }
}

function WebSocketDot({ status }) {
  const color =
    status === "connected" ? "bg-emerald-500" : status === "connecting" ? "bg-amber-500" : "bg-rose-500";
  const label =
    status === "connected" ? "WebSocket conectado" : status === "connecting" ? "Conectando..." : "Desconectado";
  return (
    <div
      data-testid="websocket-status-indicator"
      title={label}
      aria-label={label}
      className={classNames("w-2.5 h-2.5 rounded-full transition-colors duration-200", color)}
    />
  );
}

function DiagnosticPanel({ state, logs, onReset }) {
  return (
    <div className="flex flex-col h-full font-mono text-xs" data-testid="diagnostic-sheet-content">
      <SheetHeader className="px-1 pb-3 border-b border-neutral-800/40">
        <SheetTitle className="font-mono uppercase tracking-widest text-sm">Diagnóstico</SheetTitle>
      </SheetHeader>

      <section className="py-4 border-b border-neutral-800/40 space-y-1">
        <div className="text-neutral-500 uppercase tracking-widest">Estado actual</div>
        <div className="grid grid-cols-2 gap-y-1 mt-2">
          <span className="text-neutral-500">questionNumber</span>
          <span>{state?.questionNumber ?? "—"}</span>
          <span className="text-neutral-500">status</span>
          <span className="uppercase">{state?.status ?? "—"}</span>
          <span className="text-neutral-500">answer</span>
          <span className="truncate">{state?.generatedAnswer ?? "—"}</span>
          <span className="text-neutral-500">match</span>
          <span>
            {state?.matched_option ? `${state.matched_option}` : "—"}
            {state?.match_confidence ? ` (${(state.match_confidence * 100).toFixed(0)}%)` : ""}
          </span>
          <span className="text-neutral-500">confidence</span>
          <span>{state?.confidence != null ? `${(state.confidence * 100).toFixed(0)}%` : "—"}</span>
          <span className="text-neutral-500">complete</span>
          <span>{state?.questionComplete ? "true" : "false"}</span>
        </div>
        {state?.questionText ? (
          <div className="mt-3">
            <div className="text-neutral-500 uppercase tracking-widest text-[10px]">questionText</div>
            <div className="mt-1 whitespace-pre-wrap break-words text-neutral-300 leading-relaxed">
              {state.questionText}
            </div>
          </div>
        ) : null}
        {state?.optionsReceived && Object.keys(state.optionsReceived).length > 0 ? (
          <div className="mt-3">
            <div className="text-neutral-500 uppercase tracking-widest text-[10px]">opciones</div>
            <ul className="mt-1 space-y-0.5">
              {Object.entries(state.optionsReceived).map(([k, v]) => (
                <li key={k} className="text-neutral-300">
                  <span className="text-neutral-500">{k})</span> {v}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
        <button
          data-testid="reset-state-button"
          onClick={onReset}
          className="mt-4 inline-flex items-center gap-2 px-3 py-1.5 border border-neutral-700 hover:bg-neutral-800 rounded-none uppercase tracking-widest text-[10px]"
        >
          <RotateCcw className="w-3 h-3" /> Reiniciar estado
        </button>
      </section>

      <section className="py-4 flex-1 overflow-y-auto">
        <div className="text-neutral-500 uppercase tracking-widest mb-2">Imágenes recibidas ({logs.length})</div>
        {logs.length === 0 ? (
          <div className="text-neutral-600">Esperando primer envío…</div>
        ) : (
          <ul className="space-y-3">
            {logs.map((l, idx) => (
              <li key={idx} className="border-l-2 border-neutral-800 pl-3">
                <div className="flex justify-between text-[10px] text-neutral-500">
                  <span>{relTime(l.received_at)}</span>
                  <span>{(l.size_bytes / 1024).toFixed(1)} KB</span>
                </div>
                <div className="text-neutral-400 text-[11px] mt-0.5">
                  {l.filename || "image"} · {l.mime} · device: {l.device_id || "—"}
                </div>
                {l.extracted ? (
                  <pre className="mt-1 text-[10px] text-neutral-500 whitespace-pre-wrap break-words leading-snug">
{JSON.stringify(l.extracted, null, 2)}
                  </pre>
                ) : null}
                {l.error ? <div className="text-rose-400 mt-1">{l.error}</div> : null}
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}

export default function App() {
  const [appState, setAppState] = useState(null);
  const [logs, setLogs] = useState([]);
  const [wsStatus, setWsStatus] = useState("connecting");
  const wsRef = useRef(null);
  const reconnectRef = useRef(null);

  useEffect(() => {
    let ws = null;
    let reconnectTimer = null;
    const connect = () => {
      if (!WS_URL) return;
      try {
        setWsStatus("connecting");
        ws = new WebSocket(WS_URL);
        wsRef.current = ws;
        ws.onopen = () => setWsStatus("connected");
        ws.onclose = () => {
          setWsStatus("disconnected");
          if (reconnectTimer) clearTimeout(reconnectTimer);
          reconnectTimer = setTimeout(connect, 1500);
          reconnectRef.current = reconnectTimer;
        };
        ws.onerror = () => {
          setWsStatus("disconnected");
          try { ws && ws.close(); } catch (_e) { /* noop */ }
        };
        ws.onmessage = (ev) => {
          try {
            const msg = JSON.parse(ev.data);
            if (msg?.type === "state" && msg.data) {
              setAppState(msg.data);
            }
            if (msg?.log) {
              setLogs((prev) => [msg.log, ...prev].slice(0, 100));
            }
          } catch (_e) {
            /* ignore */
          }
        };
      } catch (_e) {
        setWsStatus("disconnected");
      }
    };

    const loadInitial = async () => {
      try {
        const r = await fetch(`${API}/state`);
        setAppState(await r.json());
      } catch (_e) { /* noop */ }
      try {
        const r = await fetch(`${API}/logs?limit=20`);
        const d = await r.json();
        if (Array.isArray(d)) setLogs(d);
      } catch (_e) { /* noop */ }
    };
    loadInitial();
    connect();

    return () => {
      if (reconnectTimer) clearTimeout(reconnectTimer);
      try { ws && ws.close(); } catch (_e) { /* noop */ }
    };
  }, []);

  const status = appState?.status || "waiting";
  const matched = appState?.matched_option;
  const questionNumber = appState?.questionNumber;
  const errorMsg = appState?.error_message;

  const handleReset = useCallback(async () => {
    try {
      await fetch(`${API}/state/reset`, { method: "POST" });
    } catch (_e) { /* noop */ }
  }, []);

  const containerBg = useMemo(() => {
    if (status === "success") return "bg-[#050505] text-white";
    if (status === "error") return "bg-[#002EB8] text-white";
    return "bg-white text-neutral-900";
  }, [status]);

  const dotInvert = status !== "waiting";

  return (
    <div
      data-testid="main-display-view"
      className={classNames(
        "h-screen w-screen overflow-hidden relative transition-colors duration-150 ease-in-out",
        containerBg,
      )}
      style={{ fontFamily: "'IBM Plex Mono', ui-monospace, monospace" }}
    >
      {/* corner controls */}
      <div className="absolute top-6 right-6 flex items-center gap-4 z-50">
        <WebSocketDot status={wsStatus} />
        <Sheet>
          <SheetTrigger asChild>
            <button
              data-testid="diagnostic-toggle-button"
              aria-label="Abrir panel de diagnóstico"
              className={classNames(
                "p-2 rounded-none transition-opacity",
                dotInvert ? "text-white/40 hover:text-white" : "text-neutral-500 hover:text-neutral-900",
              )}
            >
              <Terminal className="w-4 h-4" />
            </button>
          </SheetTrigger>
          <SheetContent
            side="right"
            className="bg-[#0a0a0a] text-neutral-200 border-l border-neutral-800 w-full sm:max-w-md"
          >
            <DiagnosticPanel state={appState} logs={logs} onReset={handleReset} />
          </SheetContent>
        </Sheet>
      </div>

      {/* small question pill (top-center) */}
      {questionNumber != null && status === "success" ? (
        <div
          data-testid="question-number"
          className="absolute top-8 left-1/2 -translate-x-1/2 uppercase tracking-[0.35em] text-xs text-white/60"
        >
          Pregunta {questionNumber}
        </div>
      ) : null}

      {/* WAITING */}
      {status === "waiting" ? (
        <div
          data-testid="status-waiting"
          className="absolute inset-0 flex flex-col items-center justify-center"
        >
          <div className="text-[10px] uppercase tracking-[0.4em] text-neutral-400">
            {questionNumber != null ? `Pregunta ${questionNumber}` : "Esperando captura"}
          </div>
          <div className="mt-4 text-2xl text-neutral-400">
            {appState?.questionText
              ? "Esperando opciones…"
              : questionNumber != null
              ? "Reconstruyendo pregunta…"
              : "Esperando imagen…"}
          </div>
          {appState?.questionText ? (
            <div className="mt-6 max-w-2xl text-center text-neutral-500 text-sm leading-relaxed px-6">
              {appState.questionText}
            </div>
          ) : null}
          {appState?.generatedAnswer ? (
            <div className="mt-4 text-[10px] uppercase tracking-[0.4em] text-neutral-300">
              Respuesta IA: <span className="text-neutral-700">{appState.generatedAnswer}</span>
              {appState.optionsReceived && Object.keys(appState.optionsReceived).length > 0 ? (
                <span className="ml-3 text-neutral-400">
                  · {Object.keys(appState.optionsReceived).length} opciones recibidas
                </span>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}

      {/* SUCCESS */}
      {status === "success" ? (
        <div
          data-testid="status-success"
          className="absolute inset-0 flex items-center justify-center pointer-events-none select-none"
        >
          <div
            data-testid="giant-answer-letter"
            className="font-black leading-none tracking-tighter"
            style={{
              fontFamily:
                "'Cabinet Grotesk', 'Archivo Black', 'Inter', system-ui, sans-serif",
              fontSize: "min(70vh, 45vw)",
            }}
          >
            {matched || "?"}
          </div>
          {appState?.match_confidence != null ? (
            <div className="absolute bottom-8 left-1/2 -translate-x-1/2 text-[10px] uppercase tracking-[0.4em] text-white/40">
              confianza · {(appState.match_confidence * 100).toFixed(0)}%
            </div>
          ) : null}
        </div>
      ) : null}

      {/* ERROR */}
      {status === "error" ? (
        <div
          data-testid="status-error"
          className="absolute inset-0 flex flex-col items-center justify-center px-8"
        >
          <div className="text-[10px] uppercase tracking-[0.4em] text-white/70">Error</div>
          <div
            data-testid="error-description"
            className="mt-6 max-w-3xl text-center text-2xl md:text-3xl leading-relaxed text-white"
          >
            {errorMsg || "Error desconocido."}
          </div>
          <button
            data-testid="error-reset-button"
            onClick={handleReset}
            className="mt-10 px-4 py-2 border border-white/40 hover:bg-white/10 uppercase tracking-[0.3em] text-xs"
          >
            Reiniciar
          </button>
        </div>
      ) : null}
    </div>
  );
}
