# Local document ingestion

The current ingester accepts local UTF-8 `.txt`, `.md`, `.markdown`, `.html`, `.htm`, and text-based `.pdf` files up to 5 MiB. PDFs are limited to 100 pages; source records include page numbers. Encrypted and image-only PDFs are rejected, and OCR is not supported. Install optional PDF support with `python -m pip install -r requirements-data.txt`. It chunks text with configurable overlap and stores each chunk in local memory with the document filename, SHA-256, and chunk number. HTML script, style, template, and noscript contents are discarded without execution. Every chunk stays unverified until a person checks it. The ingester never runs document content as code or treats its instructions as commands.

Do not ingest private documents, personal information about other people, passwords, or sensitive files. Sensitive classification is blocked because encryption and access controls are not implemented. Only ingest documents you have permission to use.

Example:

```powershell
python ".\scripts\ingest_document.py" ".\docs\architecture.md"
python ".\scripts\memory_cli.py" search "component contracts"
```

Source attribution validation, semantic embeddings, and contradiction review remain future work. This local feature is an early knowledge-ingestion step, not a complete document-research system.

