import os
import csv
import asyncio
import anthropic
import httpx
import unicodedata
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime, timezone, timedelta
from difflib import SequenceMatcher

app = FastAPI()

CLAUDE_API_KEY     = os.getenv("CLAUDE_API_KEY")
ZAPI_INSTANCE_ID   = os.getenv("ZAPI_INSTANCE_ID")
ZAPI_TOKEN         = os.getenv("ZAPI_TOKEN")
ZAPI_CLIENT_TOKEN  = os.getenv("ZAPI_CLIENT_TOKEN", "")
GROQ_API_KEY       = os.getenv("GROQ_API_KEY")

conversations: dict = {}
product_cache: list = []
cache_updated = None

# ─── BÚSQUEDA MEJORADA ───────────────────────────────────────

def strip_accents(text: str) -> str:
    return ''.join(
        c for c in unicodedata.normalize('NFD', text)
        if unicodedata.category(c) != 'Mn'
    )

SYNONYMS = {
    "bomba":       ["inflador"],
    "inflar":      ["inflador"],
    "inflable":    ["inflador"],
    "velita":      ["vela"],
    "globito":     ["globo"],
    "tarta":       ["torta"],
    "sombrero":    ["gorro"],
    "gorrito":     ["gorro"],
    "pito":        ["silbato"],
    "flauta":      ["silbato"],
    "confetti":    ["confeti"],
    "recuerdo":    ["souvenir"],
    "escarcha":    ["glitter"],
    "antifaz":     ["mascara", "careta"],
    "mascarita":   ["antifaz", "careta"],
    "pitillo":     ["sorbete"],
    "canita":      ["sorbete"],
    "serpentin":   ["serpentina"],
    "matasuegra":  ["matasuegras"],
    "cumple":      ["cumpleanos"],
    "cumpleano":   ["cumpleanos"],
    "papel":       ["papel", "tissue"],
    "tissue":      ["papel"],
    "cinta":       ["cinta", "ribbon"],
    "moño":        ["mono", "cinta"],
    "mono":        ["mono", "moño"],
    "bolsita":     ["bolsa"],
    "cajita":      ["caja"],
    "platito":     ["plato"],
    "vasito":      ["vaso"],
}

def expand_words(words: list) -> list:
    expanded = set(words)
    for w in words:
        if w in SYNONYMS:
            expanded.update(SYNONYMS[w])
        if len(w) > 4 and w.endswith('s'):
            expanded.add(w[:-1])
        if len(w) > 5 and w.endswith('es'):
            expanded.add(w[:-2])
    return list(expanded)

# ─── PRODUCTOS ───────────────────────────────────────────────

def load_products():
    global product_cache, cache_updated
    print("Cargando productos desde CSV...")
    products = []
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "productos.csv")
    for enc in ("utf-8-sig", "latin-1"):
        try:
            with open(csv_path, encoding=enc, newline="") as f:
                reader = csv.reader(f)
                for i, row in enumerate(reader):
                    if i < 3:
                        continue
                    if len(row) < 3:
                        continue
                    nombre = row[1].strip()
                    precio_str = row[2].strip().replace(",", "")
                    if not nombre or not precio_str:
                        continue
                    try:
                        precio = float(precio_str)
                        if precio <= 0:
                            continue
                        products.append({
                            "item": nombre,
                            "precios": [{"precio": precio}],
                            "rubro": {"nombre": ""},
                        })
                    except ValueError:
                        continue
            break
        except UnicodeDecodeError:
            continue
        except FileNotFoundError:
            print("ERROR: productos.csv no encontrado")
            break
        except Exception as e:
            print(f"Error cargando CSV: {e}")
            break
    product_cache = products
    cache_updated = datetime.now()
    print(f"✅ {len(product_cache)} productos cargados")

async def refresh_cache_loop():
    while True:
        await asyncio.sleep(3600)
        load_products()

def search_products(query: str) -> list:
    query_norm = strip_accents(query.lower())
    raw_words = [w for w in query_norm.split() if len(w) > 2]
    query_words = expand_words(raw_words)
    scored = []
    for p in product_cache:
        name_norm = strip_accents(p.get("item", "").lower())
        name_words = name_norm.split()
        score = 0.0
        for qword in query_words:
            if any(qword in nword or nword in qword for nword in name_words):
                score += 3
            else:
                best = max(
                    (SequenceMatcher(None, qword, nword).ratio() for nword in name_words),
                    default=0
                )
                if best > 0.72:
                    score += best * 2
        if score > 0:
            precio = None
            for pr in p.get("precios", []):
                val = float(pr.get("precio", 0))
                if val > 0:
                    precio = val
                    break
            if precio:
                scored.append({
                    "nombre": p.get("item"),
                    "precio": precio,
                    "rubro": p.get("rubro", {}).get("nombre", ""),
                    "score": score,
                })
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:10]

# ─── MENSAJES ────────────────────────────────────────────────

async def send_message(phone: str, text: str) -> bool:
    url = f"https://api.z-api.io/instances/{ZAPI_INSTANCE_ID}/token/{ZAPI_TOKEN}/send-text"
    payload = {"phone": phone, "message": text}
    headers = {}
    if ZAPI_CLIENT_TOKEN:
        headers["Client-Token"] = ZAPI_CLIENT_TOKEN
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            print(f"Z-API send: {r.status_code}")
            return r.status_code == 200
        except Exception as e:
            print(f"Error enviando: {e}")
            return False

async def transcribe_audio(audio_url: str) -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.get(audio_url)
            if r.status_code != 200:
                return ""
            audio_bytes = r.content
            headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}
            files = {"file": ("audio.ogg", audio_bytes, "audio/ogg")}
            data = {"model": "whisper-large-v3", "language": "es"}
            r2 = await client.post(
                "https://api.groq.com/openai/v1/audio/transcriptions",
                headers=headers,
                files=files,
                data=data,
            )
            if r2.status_code == 200:
                transcription = r2.json().get("text", "")
                print(f"Audio transcripto: {transcription}")
                return transcription
    except Exception as e:
        print(f"Error transcribiendo: {e}")
    return ""

# ─── PROMPT ──────────────────────────────────────────────────

def get_saludo() -> str:
    hora = datetime.now(timezone(timedelta(hours=-3))).hour
    if hora < 12:
        return "Hola, buen día"
    elif hora < 19:
        return "Hola, buenas tardes"
    else:
        return "Buenas noches"

def build_system_prompt(products_info: str = "") -> str:
    sec = ""
    if products_info:
        sec = "\nPRODUCTOS DISPONIBLES:\n" + products_info + "\n\n"
    saludo = get_saludo()
    return (
        f"Sos Max, vendedor de Cotimax, cotillonería en Córdoba, Argentina. "
        f"Sos un vendedor con experiencia: conocés bien los productos, sabés qué necesita cada cliente según su evento, y te interesa genuinamente ayudar a que la fiesta salga bien.\n\n"
        "CÓMO SOS:\n"
        "- Hablás de forma natural y cálida, como alguien que disfruta su trabajo\n"
        "- Español rioplatense, semi-formal: ni muy distante ni demasiado confianzudo\n"
        "- JAMÁS: 'che', 'boludo', malas palabras ni expresiones de exceso de confianza\n"
        "- SIEMPRE en español, sin importar el idioma del cliente\n"
        "- Respuestas cortas y concretas (2-4 líneas), 1-2 emojis cuando suman al mensaje\n"
        "- Sos proactivo: pensás qué más podría necesitar el cliente según lo que pide\n"
        "- Cuando no entendés bien qué buscan, preguntás: 'Para qué evento es?' o '¿Qué tenés en mente?'\n"
        "- Si el cliente menciona el tipo de evento (cumple, casamiento, quince, etc.), lo usás para sugerir mejor\n\n"
        "TU TRABAJO:\n"
        "- Informás precios de lo que tenemos\n"
        "- Sugerís productos complementarios que tengan sentido para el evento\n"
        "- Ayudás al cliente a armar mejor su fiesta con lo que ofrecemos\n"
        "- Para envíos y formas de pago: 'Para eso te ayuda nuestro personal directamente 😊'\n\n"
        "NEGOCIO:\n"
        "- Vendemos cotillón, artículos de repostería, librería y artículos para fiestas\n"
        "- Estamos en Av. Donato Álvarez 8720, Córdoba\n"
        "- Horario: lunes a sábado de 9:00 a 13:00 y de 16:30 a 20:00 hs\n\n"
        "SALUDO INICIAL:\n"
        f"- Primer mensaje del cliente → respondé con: '{saludo}! Bienvenido/a a Cotimax 😊 ¿En qué te puedo ayudar?'\n"
        "- Mensajes siguientes: no repetís el saludo\n\n"
        "SOBRE LOS PRODUCTOS:\n"
        "- Analizá el CONTEXTO completo antes de responder. 'Globo' puede ser látex, helio, metalizado — si hay dudas, preguntás.\n"
        "- Antes de decir que no tenemos algo, revisá bien: puede estar con otro nombre, variante, plural o singular diferente.\n"
        "- Si hay varias opciones, mostrás las más relevantes con precio.\n"
        "- NUNCA menciones stock ni cantidades — solo precios.\n"
        "- NUNCA digas 'no tenemos en stock' ni nada de disponibilidad.\n"
        "- Si algo definitivamente no lo tenemos, lo decís con amabilidad y ofrecés lo más parecido.\n"
        + sec +
        "REGLA PRINCIPAL: Sos el vendedor. Informás precios, ayudás a elegir bien y dejás al cliente con ganas de comprar."
    )

# ─── PROCESO ─────────────────────────────────────────────────

async def process_message(phone: str, text: str) -> str:
    if len(product_cache) == 0:
        return "¡Hola! Ya casi estoy lista, estoy cargando los productos. ¿Me mandás el mensaje de nuevo en un minuto? 😊"
    if phone not in conversations:
        conversations[phone] = []
    conversations[phone].append({"role": "user", "content": text})
    if len(conversations[phone]) > 10:
        conversations[phone] = conversations[phone][-10:]
    products = search_products(text)
    products_info = ""
    if products:
        lines = [
            f"- {p['nombre']}: ${p['precio']:,.0f}" + (f" ({p['rubro']})" if p["rubro"] else "")
            for p in products
        ]
        products_info = "\n".join(lines)
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        system=build_system_prompt(products_info),
        messages=conversations[phone],
    )
    reply = response.content[0].text
    conversations[phone].append({"role": "assistant", "content": reply})
    return reply

# ─── WEBHOOK ─────────────────────────────────────────────────

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()

        if body.get("fromMe"):
            return JSONResponse({"status": "ignored"})

        phone = body.get("phone", "")
        if not phone:
            return JSONResponse({"status": "ignored"})

        text = ""
        is_audio = False

        # Mensaje de texto
        if "text" in body and isinstance(body["text"], dict):
            text = body["text"].get("message", "").strip()

        # Mensaje de audio
        elif "audio" in body and isinstance(body["audio"], dict):
            is_audio = True
            # Z-API puede transcribir automaticamente
            transcription = body["audio"].get("transcriptionText", "")
            if transcription:
                text = transcription.strip()
            else:
                audio_url = body["audio"].get("audioUrl", "")
                if audio_url:
                    text = await transcribe_audio(audio_url)

        if not text:
            if is_audio:
                await send_message(phone, "Lo siento, no pude entender el audio. ¿Podés escribirme? 😊")
            return JSONResponse({"status": "ignored"})

        print(f"📩 {phone}: {text}")
        reply = await process_message(phone, text)
        await send_message(phone, reply)
        print(f"📤 {phone}: {reply}")
        return JSONResponse({"status": "ok"})

    except Exception as e:
        print(f"Error webhook: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "productos": len(product_cache),
        "cache_actualizado": cache_updated.isoformat() if cache_updated else None,
    }

@app.on_event("startup")
async def startup():
    load_products()
    asyncio.create_task(refresh_cache_loop())
