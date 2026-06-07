# Quiz Realtime Platform – PRD

## Original Problem Statement (resumen)
Plataforma web que recibe imágenes de cuestionarios desde una app móvil (HTTP POST multipart/form-data),
detecta preguntas numeradas y opciones múltiples mediante GPT-5-mini (visión/OCR), genera la respuesta
correcta con GPT-5, compara contra las opciones (código + IA semántica) y muestra el resultado en
tiempo real (WebSocket) con una interfaz minimalista de 3 estados (blanco/negro/azul).

## Stack
- Backend: FastAPI (Python) + MongoDB + WebSocket nativo (`/api/ws`)
- Frontend: React + Tailwind + shadcn Sheet
- IA: OpenAI GPT-5-mini (visión), GPT-5 (respuesta) — key del usuario en backend/.env
- Comparación: unidecode + rapidfuzz (código), fallback GPT-5-mini (semántico)

## Endpoints
- `POST /api/upload` — multipart con `image`, `timestamp?`, `device_id?`
- `GET /api/state` — estado actual
- `POST /api/state/reset` — limpia el estado
- `GET /api/logs?limit=N` — historial persistente
- `WS /api/ws` — broadcast en tiempo real de estado + logs

## Architecture & state
- Estado global en memoria (single QuestionState) — protegido con asyncio.Lock.
- Una imagen → GPT-5-mini extrae JSON {questionNumber, items[{type,text,option}]}.
- Tipos: question_complete | question_fragment | option | unknown.
- Cambio de questionNumber → reset automático del contexto.
- Dedup tolerante a OCR (unidecode + rapidfuzz, ratio≥88).
- Fragmentos se concatenan; question_complete reemplaza si es más larga.
- Respuesta IA se genera sólo cuando la pregunta parece completa o llegaron opciones.
- Re-evaluación de opciones al cambiar respuesta o al recibir nuevas opciones.
- Estados UI: waiting (fondo blanco), success (negro + letra gigante), error (azul).

## What's been implemented (2026-06-07)
- Pipeline OCR + extracción JSON con GPT-5-mini visión.
- Generación de respuesta + aliases + confianza con GPT-5.
- Comparación híbrida (código rápido → IA semántica si ambiguo).
- Persistencia de logs de imágenes en MongoDB (`image_logs`).
- WebSocket con broadcast de estado y log entries.
- UI minimalista en 3 estados con tipografía Cabinet Grotesk + IBM Plex Mono.
- Diagnostic Sheet con historial de imágenes y JSON extraído.
- Reset manual desde panel; reset automático al detectar nuevo questionNumber.
- Validación: question_complete sólo si abre con "¿"/palabra interrogativa Y termina con "?".
- **Modo demo (2026-06-07)**: botón discreto en la esquina superior derecha para subir imagen desde la UI (file picker), con indicador de progreso "SUBIENDO IMAGEN…" y manejo de errores. Reutiliza `POST /api/upload`.

## Robustness validated
- Imágenes duplicadas (mismo frag o misma opción) → no duplican estado.
- Fragmentos múltiples → se concatenan correctamente.
- Question switch (15 → 16 → 17) → reset automático.
- "complete" mal clasificado → downgrade a fragment.

## Next Action Items / Backlog (P1)
- Endpoint `GET /api/history` con preguntas pasadas resueltas (no sólo logs).
- Soporte multi-sesión / multi-usuario (varias preguntas en paralelo) → opcional.
- Métricas: aciertos, confianza promedio, tasa de duplicados ignorados.
- Modo "demostración" para probar sin app móvil: subir imagen desde la UI.
- Persistir el estado por pregunta en Mongo (no sólo en memoria).
- Soporte para más letras de opción (F, G…) — actualmente A-E asumidas.

## Credentials
- OpenAI API key del usuario en `backend/.env` (OPENAI_API_KEY).
- Modelos: gpt-5-mini (vision) y gpt-5 (answer) — confirmados accesibles.
