# Experimental local query flow

`scripts/chat_local.py` runs a local model continuation and a memory search for one query. It displays retrieved records, source labels, and verification status separately from model output. Memory text is not inserted into the model prompt, so this is not yet retrieval-grounded generation. The toy checkpoints produce unreliable text; use this command only to exercise local wiring.

```powershell
python ".\scripts\chat_local.py" "causal attention" --device cuda
```

Add `--verified-only` to hide memories that have not been checked by a person. No network or external inference API is used.

