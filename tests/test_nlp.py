"""Unit and integration tests for Aarohan 2.0 NLP and Language Engine."""

from aarohan.nlp.context import ContextResolver
from aarohan.nlp.intent import Intent, IntentClassifier
from aarohan.nlp.normalizer import TextNormalizer


def test_spelling_and_typo_normalization():
    normalizer = TextNormalizer()

    # 1. Typical typos
    res1 = normalizer.normalize("how to conect sql wit powerbi")
    assert "connect" in res1.normalized
    assert "with" in res1.normalized
    assert "Power BI" in res1.normalized
    assert res1.original == "how to conect sql wit powerbi"

    # 2. Dataset typo
    res2 = normalizer.normalize("creat a 10k banking datset")
    assert "create" in res2.normalized
    assert "dataset" in res2.normalized

    # 3. Explanation typo
    res3 = normalizer.normalize("explane me this code")
    assert "explain" in res3.normalized

    # 4. Abbreviation typos
    res4 = normalizer.normalize("wht is fabric and hw it wrks")
    assert "what" in res4.normalized
    assert "how" in res4.normalized
    assert "works" in res4.normalized

    # 5. Phrase mapping
    res5 = normalizer.normalize("make excel and give download")
    assert "Create an Excel file and provide a download link" in res5.normalized

    # 6. Scientific law typo
    res6 = normalizer.normalize("tell me the formula of third low of thermodynamics")
    assert "third law of thermodynamics" in res6.normalized


def test_technical_terminology_protection():
    normalizer = TextNormalizer()

    # Technical terms should not be altered
    tech_query = "SELECT id, count(*) FROM users WHERE sql = 1 -- Power BI, PyTorch, CUDA, SwiGLU, RMSNorm"
    res = normalizer.normalize(tech_query)
    assert res.normalized == tech_query

    # Code block preservation
    code_query = "Here is my code:\n```python\ndef conect():\n    pass\n```"
    res_code = normalizer.normalize(code_query)
    assert "def conect():" in res_code.normalized


def test_intent_classification():
    classifier = IntentClassifier()

    # Dataset Generation Intent
    r1 = classifier.classify("create a 10,000 banking transactions dataset")
    assert r1.primary_intent == Intent.DATASET_GENERATION

    # Excel Generation Intent
    r2 = classifier.classify("export these records to excel xlsx spreadsheet")
    assert r2.primary_intent == Intent.EXCEL_GENERATION

    # Image Generation Intent
    r3 = classifier.classify("create an image of a small boy playing cricket")
    assert r3.primary_intent == Intent.IMAGE_GENERATION

    # SQL Assistance Intent
    r4 = classifier.classify("write a SQL query to join customer and accounts tables")
    assert r4.primary_intent == Intent.SQL_ASSISTANCE

    # Research Intent
    r5 = classifier.classify("research the latest features in Microsoft Fabric and give me sources")
    assert Intent.RESEARCH in [m.intent for m in r5.intents] or Intent.WEB_SEARCH in [m.intent for m in r5.intents]

    # Multi-intent request
    multi = classifier.classify("Research Indian banking data, create a dataset and make a chart")
    assert multi.is_multi_intent is True
    intents = multi.intent_names
    assert "RESEARCH" in intents or "WEB_SEARCH" in intents
    assert "DATASET_GENERATION" in intents
    assert "CHART_GENERATION" in intents


def test_context_resolution():
    resolver = ContextResolver()

    # Standalone query (no follow-up)
    res_standalone = resolver.resolve("What is a Transformer architecture?")
    assert res_standalone.has_follow_up is False

    # Contextual follow-up query
    history = [
        {"role": "user", "content": "can you create a banking dataset of 1k records"},
        {"role": "assistant", "content": "Here is a preview of the banking dataset."},
    ]
    res_followup = resolver.resolve("i need this data in excel format", history=history)
    assert res_followup.has_follow_up is True
    assert "banking dataset" in res_followup.expanded_query
