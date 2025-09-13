from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, String, Text, Integer, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from urllib.parse import quote_plus
import os
import uuid

app = FastAPI()

# ---------------- Dataset folder ---------------- #
DATASET_DIR = "text_dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# ---------------- MySQL Database ---------------- #
MYSQL_USER = "root"
MYSQL_PASSWORD = quote_plus("Disha@2710")
MYSQL_HOST = "localhost"
MYSQL_PORT = "3306"
MYSQL_DB = "text_to_text"

DATABASE_URL = f"mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOST}:{MYSQL_PORT}/{MYSQL_DB}"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------- Database Model ---------------- #
class TextEntry(Base):
    __tablename__ = "text_entries"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input_text = Column(Text)
    translated_text = Column(Text)
    src_lang = Column(String(50))
    tgt_lang = Column(String(50))
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ---------------- Language Map ---------------- #
LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "marathi": "mar_Deva"
}

# ---------------- Load Translation Model ---------------- #
print("🔄 Loading IndicTrans2 1B model...")
MODEL_NAME = "ai4bharat/indictrans2-en-indic-1B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).eval()

# ---------------- Request Schema ---------------- #
class TranslationRequest(BaseModel):
    text: str
    src: str  # english, hindi, marathi
    tgt: str  # english, hindi, marathi

# ---------------- API Endpoint ---------------- #
@app.post("/translate")
async def translate(req: TranslationRequest):
    # Validate language
    if req.src not in LANG_MAP or req.tgt not in LANG_MAP:
        return JSONResponse({"error": "Invalid src/tgt language"}, status_code=400)

    src_lang, tgt_lang = LANG_MAP[req.src], LANG_MAP[req.tgt]

    try:
        # Prepare input for model
        input_text = f"{src_lang} {tgt_lang} {req.text}"
        inputs = tokenizer([input_text], return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        # Generate translation
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=512, num_beams=4, use_cache=False)
        translated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # ---------------- Save in MySQL ---------------- #
        db = SessionLocal()
        entry = TextEntry(
            input_text=req.text,
            translated_text=translated_text,
            src_lang=req.src,
            tgt_lang=req.tgt
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        db.close()

        # ---------------- Save as text file in dataset folder ---------------- #
        file_id = str(uuid.uuid4())
        file_path = os.path.join(DATASET_DIR, f"{file_id}.txt")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(f"Input ({req.src}): {req.text}\n")
            f.write(f"Translated ({req.tgt}): {translated_text}\n")

        # ---------------- Return response ---------------- #
        return JSONResponse({
            "message": "Text translated and saved successfully",
            "input_text": req.text,
            "translated_text": translated_text,
            "src_lang": req.src,
            "tgt_lang": req.tgt,
            "entry_id": entry.id,
            "file_url": f"/{DATASET_DIR}/{file_id}.txt"
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)