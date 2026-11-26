# Aarohan

Aarohan is a research project for an independently trained, decoder-only language model and the systems around it. Its source of requirements is [AAROHAN_Master_PRD.txt](AAROHAN_Master_PRD.txt).

## Current status

Phase 0 has started. This repository currently contains architecture and development planning; it does not yet contain a trained model, tokenizer, memory service, or working training pipeline. No performance claims have been made.

The user-provided environment report confirms Python 3.12.10, PyTorch 2.14.1+cu132, CUDA availability, and an NVIDIA GeForce RTX 2060 with 6 GB VRAM. NumPy was missing at that point and is now listed as a dependency. RAM is still unknown. The workspace shell is separate from the user's `C:\Users\sable\.venv` and cannot currently execute that interpreter; see [docs/environment.md](docs/environment.md).

## Phase 0 setup

1. Install Python 3.11 or newer and Git. Install a PyTorch build appropriate for the machine; CUDA is optional.
2. From the repository root, create and activate a virtual environment:

   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

3. Run `python scripts/environment_report.py` to record Python, PyTorch, device, memory and CUDA information.
4. Follow the staged checklist in [docs/development-checklist.md](docs/development-checklist.md). Do not start large-scale training before the model architecture and small-run recovery checks pass.
5. Run `python -m pytest` and `python scripts/train_tiny.py --device cuda` for the initial synthetic pipeline smoke test. This tiny synthetic task validates code flow and checkpoint reload only; it is not language training or an intelligence result.

## Project boundaries

- The foundation model will use a project-owned tokenizer and randomly initialized PyTorch weights. Pretrained LLM weights and commercial inference APIs are not core dependencies.
- Model, data, memory, research and evaluation are separate components. Document processing and research outputs are untrusted until reviewed and source-validated.
- Experimental checkpoints require independent evaluation before promotion. Memory updates do not imply model parameter updates.
- Dataset use requires recorded provenance and license review; public availability alone is not permission to train.

## Documents

- [Architecture](docs/architecture.md)
- [Environment constraints](docs/environment.md)
- [Development checklist](docs/development-checklist.md)
- [Dataset licensing register](data/README.md)

## Original corpus starter

The initial text corpus is deliberately empty. Write original, non-private paragraphs into `data/corpora/original/corpus.txt` and separate, non-overlapping paragraphs into `validation.txt`. Complete `manifest.json` only after reviewing authorship, permission, and privacy. Check readiness with:

```powershell
python ".\scripts\validate_corpus.py"
```

The validator blocks empty or unreviewed data and reports file hashes. It cannot determine authorship, copyright status, or whether text contains private information.

