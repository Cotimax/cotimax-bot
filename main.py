import os
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
DUX_TOKEN          = os.getenv("DUX_TOKEN")
DUX_DEPOSIT_ID     = os.getenv("DUX_DEPOSIT_ID", "1")
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

async def load_products():
    global product_cache, cache_updated
    print("Cargando productos de Dux...")
    all_products = []
    offset = 0
    limit = 50
    async with httpx.AsyncClient(timeout=30) as client:
        while True:
            try:
                r = await client.get(
                    "https://erp.duxsoftware.com.ar/WSERP/rest/services/items",
                    params={"offset": offset, "limit": limit, "idDeposito": DUX_DEPOSIT_ID},
                    headers={"Authorization": DUX_TOKEN},
                )
                if r.status_code != 200:
                    print(f"Dux error: {r.status_code}")
                    break
                data = r.json()
                results = data.get("results", [])
                all_products.extend(results)
                total = data.get("paging", {}).get("total", 0)
                offset += limit
                if offset >= total:
                    break
                await asyncio.sleep(6)
            except Exception as e:
                print(f"Error cargando productos: {e}")
                break
    product_cache = all_products
    cache_updated = datetime.now()
    print(f"✅ {len(product_cache)} productos cargados")

async def refresh_cache_loop():
    while True:
        await asyncio.sleep(300)
        await load_products()

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
        sec = "\nPRODUCTOS ENCONTRADOS:\n" + products_info + "\n"
    saludo = get_saludo()
    return (
        "Sos Max, asistente virtual de Cotimax, cotillonería ubicada en Córdoba, Argentina.\n\n"
        "PERSONALIDAD:\n"
        "- Tono semi-formal y amigable, como un vendedor real y cercano\n"
        "- Usás español rioplatense (vos, dale) con tono prolijo y respetuoso\n"
        "- NUNCA uses: 'che', 'boludo', malas palabras ni expresiones de demasiada confianza\n"
        "- SIEMPRE respondés en español, sin importar el idioma en que escriba el cliente\n"
        "- Sos entusiasta con los productos y ofrecés artículos relacionados para complementar\n"
        "- Respuestas cortas y claras, 2-4 líneas, 1-2 emojis máximo\n"
        "- Si no sabés algo lo decís con naturalidad, sin inventar\n\n"
        "SALUDO INICIAL:\n"
        f"- Cuando sea el primer mensaje del cliente, saludá con: '{saludo}! Bienvenido/a a Cotimax 😊 ¿En qué te puedo ayudar?'\n"
        "- En mensajes siguientes no repitas el saludo\n\n"
        "NEGOCIO:\n"
        "- Vendemos cotillón, artículos de repostería, librería y artículos para fiestas\n"
        "- Estamos en Av. Donato Álvarez 8720, Córdoba\n"
        "- Horario: lunes a sábado de 9:00 a 13:00 y de 16:30 a 20:00 hs\n"
        "- Para envíos y formas de pago, informá que nuestro personal los va a ayudar\n\n"
        "CUANDO PREGUNTAN POR PRODUCTOS:\n"
        "- Analizá el CONTEXTO completo de lo que pide el cliente. El cliente puede usar sinónimos, apodos o nombres distintos al que figura en la lista.\n"
        "- Antes de decir que no tenemos algo, revisá bien la lista: puede aparecer con otro nombre, variante o abreviación.\n"
        "- Si hay varios productos relacionados, mostrá todas las opciones con precio.\n"
        "- NUNCA menciones stock, disponibilidad ni cantidades. Solo informás precios.\n"
        "- NUNCA digas 'no tenemos en stock' ni nada relacionado con stock.\n"
        "- Para pagos o envíos: decí 'Para coordinar eso, nuestro personal te va a ayudar 😊'\n"
        "- Ofrecé siempre algo complementario para sumar a la venta.\n"
        "- Si el producto definitivamente NO está en la lista, decí 'No manejamos ese artículo' y ofrecé algo similar.\n"
        + sec +
        "REGLA DE ORO: Sos el vendedor. Tu función es informar precios. Si está en la lista, das el precio. Si no está, lo decís con amabilidad y ofrecés alternativas."
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
        model="claude-haiku-4-5",
        max_tokens=350,
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
    asyncio.create_task(load_products())
    asyncio.create_task(refresh_cache_loop())
