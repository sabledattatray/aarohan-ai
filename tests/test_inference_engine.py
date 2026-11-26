"""Tests for the unified provider-independent inference interface."""

from pathlib import Path
import pytest
import torch

from aarohan.inference import (
    BaseInferenceEngine,
    CustomTransformerInferenceEngine,
    FoundationInferenceEngine,
    InferenceManager,
    ModelCapability,
    ModelInfo,
    ModelType,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = PROJECT_ROOT / "checkpoints" / "aarohan-sft-001.pt"
DEFAULT_TOK = PROJECT_ROOT / "artifacts" / "tokenizers" / "wikitext2-bpe" / "tokenizer.json"


def test_base_interface_cannot_be_instantiated():
    with pytest.raises(TypeError):
        BaseInferenceEngine()  # type: ignore


def test_foundation_engine_metadata_and_capabilities():
    engine = FoundationInferenceEngine(model_name_or_path="Qwen/Qwen2.5-0.5B-Instruct")
    assert isinstance(engine, BaseInferenceEngine)
    
    info = engine.get_model_info()
    assert isinstance(info, ModelInfo)
    assert info.model_id == "Qwen/Qwen2.5-0.5B-Instruct"
    assert info.model_type == ModelType.FOUNDATION_PRETRAINED
    assert ModelCapability.TEXT_GENERATION in engine.available_capabilities()
    assert ModelCapability.STREAMING in engine.available_capabilities()
    assert ModelCapability.CHAT_TEMPLATE in engine.available_capabilities()
    
    health = engine.health_check()
    assert "status" in health
    assert health["model_id"] == "Qwen/Qwen2.5-0.5B-Instruct"


def test_custom_transformer_engine_metadata():
    if not DEFAULT_CKPT.is_file() or not DEFAULT_TOK.is_file():
        pytest.skip("Custom checkpoint or tokenizer not found on disk")

    engine = CustomTransformerInferenceEngine(
        checkpoint_path=DEFAULT_CKPT,
        tokenizer_path=DEFAULT_TOK,
        device="cpu",
    )
    assert isinstance(engine, BaseInferenceEngine)

    info = engine.get_model_info()
    assert isinstance(info, ModelInfo)
    assert info.model_type == ModelType.CUSTOM_TRANSFORMER
    assert "aarohan_custom" in info.model_id
    assert ModelCapability.TEXT_GENERATION in engine.available_capabilities()
    assert ModelCapability.STREAMING in engine.available_capabilities()
    
    health = engine.health_check()
    assert health["checkpoint_exists"] is True
    assert health["tokenizer_exists"] is True


def test_inference_manager_lifecycle():
    mgr = InferenceManager(
        custom_checkpoint_path=DEFAULT_CKPT if DEFAULT_CKPT.is_file() else None,
        custom_tokenizer_path=DEFAULT_TOK if DEFAULT_TOK.is_file() else None,
    )
    
    foundation_eng = mgr.get_engine("foundation")
    assert isinstance(foundation_eng, FoundationInferenceEngine)

    models = mgr.list_models()
    assert len(models) >= 1
    model_types = [m["model_type"] for m in models]
    assert ModelType.FOUNDATION_PRETRAINED.value in model_types

    health_all = mgr.health_check_all()
    assert "foundation" in health_all
    assert health_all["foundation"]["status"] in ("healthy", "ready_to_load")


def test_custom_transformer_generate_and_stream_cpu():
    if not DEFAULT_CKPT.is_file() or not DEFAULT_TOK.is_file():
        pytest.skip("Custom checkpoint or tokenizer not found on disk")

    engine = CustomTransformerInferenceEngine(
        checkpoint_path=DEFAULT_CKPT,
        tokenizer_path=DEFAULT_TOK,
        device="cpu",
    )

    # Test complete generate()
    result = engine.generate(
        "Hello Aarohan",
        system_prompt="You are AAROHAN AI.",
        max_new_tokens=8,
        temperature=0.7,
    )
    assert result.text is not None
    assert result.generated_tokens > 0
    assert result.device == "cpu"
    assert result.warning is not None  # Experimental model warning

    # Test stream()
    stream_chunks = list(engine.stream("Hello", max_new_tokens=4, temperature=0.7))
    assert len(stream_chunks) > 0
    
    # Test unload()
    assert engine.unload() is True
    assert engine.is_loaded() is False
