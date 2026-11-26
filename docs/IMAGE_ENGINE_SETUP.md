# AAROHAN AI — IMAGE STUDIO 1.0 SETUP & USER GUIDE

**Project:** AAROHAN AI  
**Author & Developer:** Datta Sable  
**Official UI:** http://127.0.0.1:8000/app/images  

---

## 1. Quick Start

1. Start the AAROHAN server:
   ```powershell
   & "C:\Users\sable\.venv\Scripts\python.exe" -m uvicorn aarohan.api:app --host 127.0.0.1 --port 8000
   ```
2. Navigate to **Image Studio**: [http://127.0.0.1:8000/app/images](http://127.0.0.1:8000/app/images).
3. Type any natural prompt (e.g., *"Create a photorealistic mountain landscape at sunset"*) and click **Generate Image**.
4. Click **Enhance Prompt** to automatically expand lighting, camera perspective, and artistic fidelity tokens.

---

## 2. ComfyUI Local Integration (Optional)

AAROHAN supports seamless integration with [ComfyUI](https://github.com/Comfy-Org/ComfyUI):

1. **Clone and Launch ComfyUI**:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   python main.py --port 8188 --lowvram --fp16-vae
   ```
2. **Configuration**:
   AAROHAN defaults to `http://127.0.0.1:8188`. To specify a custom endpoint, set the environment variable:
   ```powershell
   $env:COMFYUI_URL = "http://127.0.0.1:8188"
   ```
3. **Health Verification**:
   When ComfyUI is running, AAROHAN's Image Studio automatically marks it **Online** and queries its installed checkpoints via `/api/images/health`. If ComfyUI is stopped, AAROHAN automatically falls back to the native local engine with zero downtime.

---

## 3. Conversational Chat Generation & Editing

You can generate and edit images directly in **AI Chat** (`/app`):
- **Generate**: *"Create an image of a futuristic AI laboratory"* $\to$ returns the generated image directly in chat.
- **Edit**: *"Make it brighter and change the background"* $\to$ detects previous image in dialogue context, applies image transformation, and returns the modified result.
- **Variation**: *"Generate another variation with a warm sunset palette"* $\to$ produces an artistic variation asset.

---

## 4. API Reference

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/images/generate` | Text-to-image synthesis |
| `POST` | `/api/images/edit` | Conversational image manipulation |
| `POST` | `/api/images/enhance-prompt` | Natural language prompt optimization |
| `GET` | `/api/images/library` | Persistent SQLite image catalog |
| `GET` | `/api/images/health` | Hardware telemetry & provider status |
| `GET` | `/api/images/models` | Available model checkpoints |
| `GET` | `/api/images/view/{filename}` | Preview image asset |
| `GET` | `/api/images/download/{filename}` | Download image asset with attachment header |
| `POST` | `/api/images/library/{id}/favorite` | Toggle bookmark star |
| `DELETE` | `/api/images/library/{id}` | Delete asset and file |
