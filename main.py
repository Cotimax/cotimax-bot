import os
import asyncio
import anthropic
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from datetime import datetime
from difflib import SequenceMatcher

app = FastAPI()

CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
DUX_TOKEN = os.getenv("DUX_TOKEN")
DUX_DEPOSIT_ID = os.getenv("DUX_DEPOSIT_ID", "1")
MAYTAPI_PRODUCT_ID = os.getenv("MAYTAPI_PRODUCT_ID")
MAYTAPI_TOKEN = os.getenv("MAYTAPI_TOKEN")
MAYTAPI_PHONE_ID = os.getenv("MAYTAPI_PHONE_ID")

conversations = {}
product_cache = []
cache_updated = None

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
                print(f"Error: {e}")
                break
    product_cache = all_products
    cache_updated = datetime.now()
    print(f"Productos cargados: {len(product_cache)}")

async def refresh_cache_loop():
    while True:
        await asyncio.sleep(1800)
        await load_products()

def search_products(query, max_results=6):
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]
    scored = []
    for p in product_cache:
        name = p.get("item", "").lower()
        score = 0.0
        for word in query_words:
            if word in name:
                score += 3
            else:
                ratio = SequenceMatcher(None, word, name).ratio()
                if ratio > 0.6:
                    score += ratio
        if score > 0:
            precio = None
            for pr in p.get("precios", []):
                val = float(pr.get("precio", 0))
                if val > 0:
                    precio = val
                    break
            if precio:
                scored.append({"nombre": p.get("item"), "precio": precio, "rubro": p.get("rubro", {}).get("nombre", ""), "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:max_results]

async def send_message(phone, text):
    url = f"https://api.maytapi.com/api/{MAYTAPI_PRODUCT_ID}/{MAYTAPI_PHONE_ID}/sendMessage"
    headers = {"x-maytapi-key": MAYTAPI_TOKEN, "Content-Type": "application/json"}
    payload = {"to_number": phone, "type": "text", "message": text}
    async with httpx.AsyncClient(timeout=15) as client:
        try:
            r = await client.post(url, json=payload, headers=headers)
            return r.status_code == 200
        except Exception as e:
            print(f"Error enviando: {e}")
            return False

def build_system_prompt(products_info=""):
    sec = ""
    if products_info:
        sec = "\nPRODUCTOS ENCONTRADOS:\n" + products_info + "\n"
    return ("Sos Coti, asistente virtual de Cotillon Casa Alberto, cotilloneria en Cordoba Argentina.\n\n"
            "PERSONALIDAD:\n"
            "- Respondés amigable y natural, como persona real\n"
            "- Usas español rioplatense: vos, che, dale\n"
            "- Sos entusiasta y ayudas a cerrar la venta\n"
            "- Respuestas cortas 2-4 lineas, 1-2 emojis\n"
            "- Si no sabes algo lo dices sin inventar\n\n"
            "NEGOCIO:\n"
            "- Vendemos cotillon, reposteria, libreria y articulos para fiestas\n"
            "- Estamos en Cordoba Argentina\n"
            "- Para envios y pagos invita a coordinar directamente\n\n"
            "CUANDO PREGUNTAN PRODUCTOS:\n"
            "- Informa precios exactos de productos disponibles\n"
            "- Si hay similares muestra opciones con precio\n"
            "- Ofrece complementarios para sumar venta\n"
            "- Nunca inventes precios ni productos\n"
            + sec +
            "Si el cliente quiere comprar, pregunta cantidad y coordina el pedido.")

async def process_message(phone, text):
    if phone not in conversations:
        conversations[phone] = []
    conversations[phone].append({"role": "user", "content": text})
    if len(conversations[phone]) > 10:
        conversations[phone] = conversations[phone][-10:]
    products = search_products(text)
    products_info = ""
    if products:
        lines = [f"- {p['nombre']}: ${p['precio']:,.0f}" + (f" ({p['rubro']})" if p["rubro"] else "") for p in products]
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

@app.post("/webhook")
async def webhook(request: Request):
    try:
        body = await request.json()
        if body.get("type") != "message":
            return JSONResponse({"status": "ignored"})
        message = body.get("message", {})
        if message.get("fromMe"):
            return JSONResponse({"status": "ignored"})
        text = message.get("text", "").strip()
        phone = body.get("conversation", "")
        if not text or not phone:
            return JSONResponse({"status": "ignored"})
        print(f"Msg de {phone}: {text}")
        reply = await process_message(phone, text)
        await send_message(phone, reply)
        return JSONResponse({"status": "ok"})
    except Exception as e:
        print(f"Error webhook: {e}")
        return JSONResponse({"status": "error"}, status_code=500)

@app.get("/health")
async def health():
    return {"status": "ok", "productos": len(product_cache)}

@app.on_event("startup")
async def startup():
    asyncio.create_task(load_products())
    asyncio.create_task(refresh_cache_loop())
