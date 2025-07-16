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

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    
    # Get the original filename and extension
    original_filename = file.filename or "audio"
    file_extension = os.path.splitext(original_filename)[1].lower()
    
    print(f"Processing audio file: {original_filename}, detected extension: {file_extension}")
    
    # If no extension or unknown extension, default to .webm (common for web recordings)
    if not file_extension or file_extension not in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm']:
        file_extension = '.webm'
        print(f"Defaulting to WebM format")
    
    # Create temp file with detected extension
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_file.flush()  # Ensure content is written to disk
        temp_file_path = temp_file.name
        
    print(f"Created temp file: {temp_file_path}, size: {len(content)} bytes")
    
    try:
        # Load audio with torchaudio (supports many formats including WebM)
        print(f"Attempting to load with torchaudio...")
        waveform, sample_rate = torchaudio.load(temp_file_path)
        print(f"Successfully loaded with torchaudio: shape={waveform.shape}, sample_rate={sample_rate}")
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
            print(f"Converted to mono: shape={waveform.shape}")
        
        # Resample to 16kHz (Whisper's expected sample rate)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
            print(f"Resampled to 16kHz: shape={waveform.shape}")
        
        # Convert to numpy for Whisper
        audio_input = waveform.squeeze().numpy()
        print(f"Converted to numpy: shape={audio_input.shape}")
        
        # Process with Whisper
        inputs = whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
        with torch.no_grad():
            generated_ids = whisper_model.generate(inputs["input_features"])
            transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        print(f"Transcription successful: {transcription[:50]}...")
        return {"transcription": transcription.strip()}
        
    except Exception as e:
        print(f"torchaudio failed: {str(e)}")
        # If torchaudio fails, try with soundfile as fallback
        try:
            print(f"Attempting to load with soundfile...")
            audio_input, sample_rate = sf.read(temp_file_path)
            print(f"Successfully loaded with soundfile: shape={audio_input.shape}, sample_rate={sample_rate}")
            
            # Convert to float32 if needed
            if audio_input.dtype != 'float32':
                audio_input = audio_input.astype('float32')
            
            # Convert to mono if stereo
            if len(audio_input.shape) > 1:
                audio_input = audio_input.mean(axis=1)
                print(f"Converted to mono: shape={audio_input.shape}")
            
            # Process with Whisper
            inputs = whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
            with torch.no_grad():
                generated_ids = whisper_model.generate(inputs["input_features"])
                transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            print(f"Transcription successful with soundfile: {transcription[:50]}...")
            return {"transcription": transcription.strip()}
            
        except Exception as e2:
            print(f"soundfile failed: {str(e2)}")
            # If soundfile fails, try with librosa as final fallback
            try:
                import librosa
                
                print(f"Attempting to load with librosa...")
                audio_input, sample_rate = librosa.load(temp_file_path, sr=None)
                print(f"Successfully loaded with librosa: shape={audio_input.shape}, sample_rate={sample_rate}")
                
                # Convert to float32 if needed
                if audio_input.dtype != 'float32':
                    audio_input = audio_input.astype('float32')
                
                # Process with Whisper
                inputs = whisper_processor(audio_input, sampling_rate=sample_rate, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    generated_ids = whisper_model.generate(inputs["input_features"])
                    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                
                print(f"Transcription successful with librosa: {transcription[:50]}...")
                return {"transcription": transcription.strip()}
                
            except Exception as e3:
                print(f"librosa failed: {str(e3)}")
                return {"error": f"Could not process audio file. Tried torchaudio: {str(e)}, soundfile: {str(e2)}, librosa: {str(e3)}. File extension: {file_extension}. Please try uploading a different audio format (WAV, MP3, FLAC recommended)."}
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except:
                pass
            
            
            
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
    transcription = "No audio provided."
    if audio is not None:
        audio_bytes = await audio.read()
        if audio_bytes:
            # Get file extension, default to .webm for web recordings
            original_filename = audio.filename or "audio"
            file_extension = os.path.splitext(original_filename)[1].lower()
            if not file_extension or file_extension not in ['.wav', '.mp3', '.flac', '.m4a', '.ogg', '.webm']:
                file_extension = '.webm'
                
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file.write(audio_bytes)
                temp_file.flush()
                temp_file_path = temp_file.name
            try:
                # Try torchaudio first (better WebM support)
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
                    transcription = whisper_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            except Exception as e:
                transcription = f"Error processing audio: {str(e)} (format: {file_extension})"
            finally:
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
        else:
            transcription = "No audio uploaded."

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