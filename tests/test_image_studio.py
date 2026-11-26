"""Comprehensive test suite for Aarohan Image Studio 1.0.

Validates:
  - Provider architecture (ComfyUI adapter & local neural pipeline)
  - Prompt intelligence and category detection
  - Text-to-image generation and aspect ratios
  - Image editing operations (brightness, contrast, palette, background, crop, variation)
  - Persistent SQLite image library and job tracking
  - All REST endpoints (/api/images/*)
  - Chat conversational generation and follow-up editing
  - Physical file persistence for photorealistic landscape, 3D tech, and business banner
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from aarohan.api import create_app
from aarohan.generation.images import (
    ComfyUIProvider,
    ImageStudioManager,
    PromptEnhancer,
)


@pytest.fixture
def temp_db(tmp_path):
    return tmp_path / "test_images.sqlite"


@pytest.fixture
def image_manager(temp_db):
    return ImageStudioManager(db_path=temp_db)


@pytest.fixture
def client(temp_db):
    app = create_app(database_path=temp_db)
    return TestClient(app)


# -----------------------------------------------------------------------------
# 1. Prompt Intelligence & Enhancer Tests
# -----------------------------------------------------------------------------
def test_prompt_enhancer_categories():
    p1 = "Create a photorealistic mountain landscape at sunset"
    res1 = PromptEnhancer.enhance(p1)
    assert res1.detected_category == "photography"
    assert "sunset" in res1.enhanced_prompt.lower()
    assert "golden hour" in res1.lighting.lower()

    p2 = "Design a professional LinkedIn banner for a Business Intelligence expert"
    res2 = PromptEnhancer.enhance(p2)
    assert res2.detected_category == "business_branding"
    assert res2.suggested_aspect_ratio == "16:9"

    p3 = "Create a 3D isometric illustration of a neural network"
    res3 = PromptEnhancer.enhance(p3)
    assert res3.detected_category in ("render_3d", "digital_art")


def test_prompt_enhancer_hinglish():
    hinglish_p = "ekdum mast pahad ki tasveer banao"
    res = PromptEnhancer.enhance(hinglish_p)
    assert "mountain" in res.enhanced_prompt.lower()
    assert "masterpiece" in res.enhanced_prompt.lower()


# -----------------------------------------------------------------------------
# 2. Local Neural Provider & Real Image Generation
# -----------------------------------------------------------------------------
def test_generate_three_real_test_images(image_manager):
    # 1. Photorealistic landscape
    landscape_res = image_manager.generate(
        prompt="Photorealistic mountain landscape at sunset with golden hour light",
        aspect_ratio="16:9",
        style="photorealistic",
        seed=101,
    )
    assert landscape_res["status"] == "completed"
    l_path = Path(landscape_res["file_path"])
    assert l_path.is_file()
    with Image.open(l_path) as img:
        assert img.size == (1024, 576)

    # 2. 3D Technology illustration
    tech_res = image_manager.generate(
        prompt="3D isometric illustration of a neural network with glowing cyan filaments",
        aspect_ratio="1:1",
        style="3d_render",
        seed=202,
    )
    assert tech_res["status"] == "completed"
    t_path = Path(tech_res["file_path"])
    assert t_path.is_file()
    with Image.open(t_path) as img:
        assert img.size == (768, 768)

    # 3. Professional business banner
    banner_res = image_manager.generate(
        prompt="Professional LinkedIn banner for Business Intelligence expert",
        aspect_ratio="16:9",
        style="business_branding",
        seed=303,
    )
    assert banner_res["status"] == "completed"
    b_path = Path(banner_res["file_path"])
    assert b_path.is_file()
    with Image.open(b_path) as img:
        assert img.size == (1024, 576)


# -----------------------------------------------------------------------------
# 3. Image Editing Operations
# -----------------------------------------------------------------------------
def test_image_editing_operations(image_manager):
    base_gen = image_manager.generate(
        prompt="A vibrant futuristic cityscape with neon lights",
        aspect_ratio="1:1",
        seed=404,
    )
    base_file = base_gen["filename"]

    # 1. Brightness
    bright_res = image_manager.edit(
        source_image=base_file,
        operation="brightness",
        parameters={"brightness": 1.4},
    )
    assert bright_res["status"] == "completed"
    assert Path(bright_res["file_path"]).is_file()

    # 2. Color Palette (Cyberpunk / Warm)
    palette_res = image_manager.edit(
        source_image=base_file,
        operation="palette",
        parameters={"palette": "cyberpunk"},
    )
    assert palette_res["status"] == "completed"
    assert Path(palette_res["file_path"]).is_file()

    # 3. Background Blur
    bg_res = image_manager.edit(
        source_image=base_file,
        operation="background",
        parameters={"action": "blur"},
    )
    assert bg_res["status"] == "completed"
    assert Path(bg_res["file_path"]).is_file()

    # 4. Aspect Ratio Crop (1:1 -> 16:9)
    crop_res = image_manager.edit(
        source_image=base_file,
        operation="crop",
        aspect_ratio="16:9",
    )
    assert crop_res["status"] == "completed"
    with Image.open(crop_res["file_path"]) as img:
        assert img.size == (1024, 576)

    # 5. Variation
    var_res = image_manager.edit(
        source_image=base_file,
        operation="variation",
    )
    assert var_res["status"] == "completed"
    assert Path(var_res["file_path"]).is_file()


# -----------------------------------------------------------------------------
# 4. Persistent SQLite Library & Job Tracking
# -----------------------------------------------------------------------------
def test_image_library_persistence(image_manager):
    db = image_manager.library_db

    # Add item
    record = db.add_image(
        filename="test_art_01.png",
        file_path="artifacts/images/test_art_01.png",
        prompt="An artistic test prompt",
        category="photography",
        style="cinematic",
    )
    img_id = record["id"]

    # Retrieve
    fetched = db.get_image(img_id)
    assert fetched is not None
    assert fetched["prompt"] == "An artistic test prompt"
    assert fetched["is_favorite"] == 0

    # Favorite toggle
    is_fav = db.toggle_favorite(img_id)
    assert is_fav is True
    assert db.get_image(img_id)["is_favorite"] == 1

    # Search & list
    search_res = db.list_images(search="artistic")
    assert len(search_res) >= 1

    fav_res = db.list_images(favorite_only=True)
    assert any(x["id"] == img_id for x in fav_res)


# -----------------------------------------------------------------------------
# 5. Provider Telemetry & ComfyUI Adapter
# -----------------------------------------------------------------------------
def test_provider_health_and_models(image_manager):
    health = image_manager.get_health()
    assert health["status"] == "operational"
    assert len(health["providers"]) >= 2

    # ComfyUI offline fallback verification
    comfy_p = ComfyUIProvider(endpoint_url="http://127.0.0.1:9999")  # Unused port
    comfy_h = comfy_p.health_check()
    assert comfy_h.is_available is False

    models = image_manager.list_models()
    assert len(models) >= 2


# -----------------------------------------------------------------------------
# 6. REST API Endpoints Verification
# -----------------------------------------------------------------------------
def test_image_api_endpoints(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # 1. POST /api/images/enhance-prompt
    enh_res = client.post(
        "/api/images/enhance-prompt",
        json={"prompt": "futuristic flying vehicle", "aspect_ratio": "16:9"},
    )
    assert enh_res.status_code == 200
    assert "enhanced_prompt" in enh_res.json()

    # 2. POST /api/images/generate
    gen_res = client.post(
        "/api/images/generate",
        json={
            "prompt": "Cyberpunk high-tech laboratory",
            "aspect_ratio": "1:1",
            "style": "cinematic",
            "seed": 555,
        },
        headers=headers,
    )
    assert gen_res.status_code == 200
    data = gen_res.json()
    assert "filename" in data
    filename = data["filename"]

    # 3. GET /api/images/view/{filename}
    view_res = client.get(f"/api/images/view/{filename}")
    assert view_res.status_code == 200
    assert view_res.headers["content-type"] == "image/png"

    # 4. GET /api/images/download/{filename}
    dl_res = client.get(f"/api/images/download/{filename}")
    assert dl_res.status_code == 200
    assert "attachment" in dl_res.headers.get("content-disposition", "")

    # 5. POST /api/images/edit
    edit_res = client.post(
        "/api/images/edit",
        json={
            "source_image": filename,
            "operation": "brightness",
            "parameters": {"brightness": 1.3},
        },
        headers=headers,
    )
    assert edit_res.status_code == 200
    assert "filename" in edit_res.json()

    # 6. GET /api/images/library
    lib_res = client.get("/api/images/library")
    assert lib_res.status_code == 200
    assert lib_res.json()["count"] >= 1

    # 7. GET /api/images/health & /api/images/models
    assert client.get("/api/images/health").status_code == 200
    assert client.get("/api/images/models").status_code == 200


# -----------------------------------------------------------------------------
# 7. Chat Conversational Integration & Auto-Editing
# -----------------------------------------------------------------------------
def test_chat_conversational_image_generation(client):
    headers = {"Authorization": "Bearer aarohan_admin_token"}

    # 1. Initial generation query in chat
    chat_payload = {
        "query": "Create an image of a futuristic AI laboratory",
        "engine": "foundation",
        "max_new_tokens": 15,
    }
    chat_res = client.post("/api/chat", json=chat_payload, headers=headers)
    assert chat_res.status_code == 200
    text = chat_res.json()["model_response"]["text"]
    assert "Aarohan AI Image Studio Generated" in text
    assert "/api/images/view/" in text
    assert "/api/images/download/" in text

    # 2. Conversational follow-up edit query in chat
    edit_payload = {
        "query": "Make it brighter and change the background",
        "engine": "foundation",
        "max_new_tokens": 15,
        "messages": [
            {"role": "user", "content": "Create an image of a futuristic AI laboratory"},
            {"role": "assistant", "content": text},
        ],
    }
    edit_res = client.post("/api/chat", json=edit_payload, headers=headers)
    assert edit_res.status_code == 200
    edit_text = edit_res.json()["model_response"]["text"]
    assert "Aarohan Image Edited" in edit_text
    assert "/api/images/view/edit_" in edit_text
