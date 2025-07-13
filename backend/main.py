from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPProcessor, CLIPModel, WhisperProcessor, WhisperForConditionalGeneration
from PIL import Image
import soundfile as sf
import torch
import io
import tempfile, os, logging
import torchaudio
import httpx
from pydantic import BaseModel
import base64
""" class ChatRequest(BaseModel):
    prompt: str """
GROQ_API_KEY = "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"
app = FastAPI()
# Load Whisper model and processor (do this once at startup)
whisper_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large").to("cuda")
whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-large")
# Load CLIP model and processor (do this once at startup)
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to("cuda")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

@app.post("/upload-image/")
async def upload_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt").to("cuda")
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)
    return {"features": image_features.cpu().tolist()}



@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    import tempfile, os
    file_extension = os.path.splitext(file.filename)[1] if file.filename else '.wav'
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file_path = temp_file.name
    try:
        # Load audio (supports MP3, WAV, etc.)
        waveform, sample_rate = torchaudio.load(temp_file_path)
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        # Convert to numpy for Whisper
        audio_input = waveform.squeeze().numpy()
        inputs = whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = whisper_model.generate(inputs["input_features"])
            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return {"transcription": transcription}
    except Exception as e:
        return {"error": str(e)}
    finally:
        if os.path.exists(temp_file_path):
            os.unlink(temp_file_path)
            
            
            
""" @app.post("/chat/")
async def chat(request: ChatRequest):
    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {"role": "user", "content": request.prompt}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload)
        data = response.json()
        if "choices" in data and data["choices"]:
            answer = data["choices"][0]["message"]["content"]
            return {"response": answer}
        elif "error" in data:
            return {"error": data["error"]}
        else:
            return {"error": "Unexpected API response: 'choices' key missing"} """
            
            
@app.post("/multi-modal-chat/")
async def multimodal_chat(
    prompt: str = Form(...),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    # --- Image Processing ---
    image_content_block = None
    image_status = "No image provided."
    if image is not None:
        image_bytes = await image.read()
        if image_bytes:
            try:
                img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                # Encode image as base64 data URL
                buffered = io.BytesIO()
                img.save(buffered, format="JPEG")
                b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
                data_url = f"data:image/jpeg;base64,{b64_string}"
                image_content_block = {"type": "image_url", "image_url": {"url": data_url}}
                image_status = "Image uploaded and encoded."
            except Exception as e:
                image_status = f"Invalid image file: {str(e)}"
        else:
            image_status = "No image uploaded."

    # --- Audio Processing ---
    transcription = None
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
                temp_file.write(audio_bytes)
                temp_file_path = temp_file.name
            try:
                waveform, sample_rate = torchaudio.load(temp_file_path)
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                if sample_rate != 16000:
                    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
                    waveform = resampler(waveform)
                audio_input = waveform.squeeze().numpy()
                inputs = whisper_processor(audio_input, sampling_rate=16000, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    generated_ids = whisper_model.generate(inputs["input_features"])
                    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            except Exception as e:
                transcription = f"Invalid audio file: {str(e)}"
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            transcription = "No audio uploaded."
    else:
        transcription = "No audio provided."

    # --- Compose Groq Vision Model Payload ---
    content_blocks = []
    content_blocks.append({"type": "text", "text": f"User prompt: {prompt}"})
    content_blocks.append({"type": "text", "text": f"Transcribed audio: {transcription}"})
    content_blocks.append({"type": "text", "text": image_status})
    if image_content_block:
        content_blocks.append(image_content_block)
    content_blocks.append({"type": "text", "text": "Please answer the user's question using all available information."})

    api_url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
        "messages": [
            {"role": "user", "content": content_blocks}
        ]
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(api_url, headers=headers, json=payload)
        data = response.json()
        if "choices" in data and data["choices"]:
            answer = data["choices"][0]["message"]["content"]
            return {"response": answer}
        elif "error" in data:
            return {"error": data["error"]}
        else:
            return {"error": "Unexpected API response: 'choices' key missing"}

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)