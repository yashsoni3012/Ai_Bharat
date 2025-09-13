from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import create_engine, Column, String, Integer, Text, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import torch, os, tempfile, soundfile as sf, uuid, numpy as np, gc
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
from parler_tts import ParlerTTSForConditionalGeneration

# ---------------- FastAPI app ---------------- #
app = FastAPI()

@app.get("/")
async def root():
    return {"status": "ok", "message": "Speech-to-speech API is running 🚀"}

# ---------------- Dataset folder ---------------- #
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)
app.mount("/dataset", StaticFiles(directory=DATASET_DIR), name="dataset")

# ---------------- PostgreSQL Database ---------------- #
DATABASE_URL = "postgresql+psycopg://speech_to_speech_user:ubVTaqH1oFxs5sVeAnsIEKOa0ocnlFTD@dpg-d32054juibrs739dp65g-a:5432/speech_to_speech"
engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

# ---------------- Database Model ---------------- #
class AudioEntry(Base):
    __tablename__ = "audio_entries"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    input_file = Column(String(255))
    output_file = Column(String(255))
    src_lang = Column(String(50))
    tgt_lang = Column(String(50))
    input_text = Column(Text)
    translated_text = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ---------------- Language Map ---------------- #
LANG_MAP = {
    "english": "eng_Latn",
    "hindi": "hin_Deva",
    "marathi": "mar_Deva"
}

# ---------------- Translate Speech Endpoint ---------------- #
@app.post("/translate_speech")
async def translate_speech(
    file: UploadFile,
    src: str = Form(...),
    tgt: str = Form(...)
):
    if src not in LANG_MAP or tgt not in LANG_MAP:
        raise HTTPException(status_code=400, detail="Invalid src/tgt language")

    src_lang, tgt_lang = LANG_MAP[src], LANG_MAP[tgt]

    try:
        # ---------------- Save Input WAV File ---------------- #
        input_filename = f"{uuid.uuid4()}_{file.filename}"
        input_path = os.path.join(DATASET_DIR, input_filename)
        with open(input_path, "wb") as f:
            f.write(await file.read())

        # ---------------- Step 1: ASR ---------------- #
        asr_model = pipeline("automatic-speech-recognition", model="openai/whisper-tiny.en")
        data, samplerate = sf.read(input_path)
        if len(data.shape) > 1:
            data = np.mean(data, axis=1)
        asr_output = asr_model({"array": data, "sampling_rate": samplerate})
        input_text = asr_output["text"]

        # cleanup ASR
        del asr_model
        gc.collect()

        # ---------------- Step 2: Translation ---------------- #
        trans_model_name = "ai4bharat/indictrans2-en-indic-dist-200M"
        trans_tokenizer = AutoTokenizer.from_pretrained(trans_model_name, trust_remote_code=True)
        trans_model = AutoModelForSeq2SeqLM.from_pretrained(
            trans_model_name,
            trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ).eval()

        trans_input = f"{src_lang} {tgt_lang} {input_text}"
        inputs = trans_tokenizer([trans_input], return_tensors="pt", padding=True, truncation=True).to(trans_model.device)
        with torch.no_grad():
            outputs = trans_model.generate(**inputs, max_length=256, num_beams=4, use_cache=False)
        translated_text = trans_tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

        # cleanup Translation
        del trans_model, trans_tokenizer, inputs, outputs
        gc.collect()

        # ---------------- Step 3: TTS ---------------- #
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tts_model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(device)
        tts_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
        description_tokenizer = AutoTokenizer.from_pretrained(tts_model.config.text_encoder._name_or_path)

        description = f"A neutral male speaker speaking {tgt} with moderate speed and pitch."
        prompt_input_ids = tts_tokenizer(translated_text, return_tensors="pt").input_ids.to(device)
        description_input_ids = description_tokenizer(description, return_tensors="pt").input_ids.to(device)

        with torch.inference_mode():
            audio_output = tts_model.generate(input_ids=description_input_ids, prompt_input_ids=prompt_input_ids)

        # ---------------- Save Output WAV File ---------------- #
        output_filename = f"{uuid.uuid4()}.wav"
        output_path = os.path.join(DATASET_DIR, output_filename)
        sf.write(output_path, audio_output.cpu().numpy().squeeze(), tts_model.config.sampling_rate)

        # cleanup TTS
        del tts_model, tts_tokenizer, description_tokenizer, audio_output
        gc.collect()

        # ---------------- Save metadata in PostgreSQL ---------------- #
        db = SessionLocal()
        entry = AudioEntry(
            input_file=input_filename,
            output_file=output_filename,
            src_lang=src,
            tgt_lang=tgt,
            input_text=input_text,
            translated_text=translated_text
        )
        db.add(entry)
        db.commit()
        db.refresh(entry)
        db.close()

        # ---------------- Return file URL and metadata ---------------- #
        return JSONResponse({
            "message": "Audio translated and saved successfully",
            "input_file_url": f"/dataset/{input_filename}",
            "output_file_url": f"/dataset/{output_filename}",
            "input_text": input_text,
            "translated_text": translated_text
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
