# Original AAROHAN corpus

Add your own original, non-private text to `corpus.txt`. This folder is intentionally shipped without sample training text. Do not paste in copyrighted stories, song lyrics, class materials, or web text unless the rights and intended use have been checked and recorded.

Before training:

1. Write or contribute text specifically for this project.
2. Fill out `manifest.json` with its author, creation date, scope, and permission.
3. Run `python scripts/validate_corpus.py` from the project root.
4. Keep a separate held-out validation file; never evaluate on the same text used for training.

The starter pipeline only checks provenance and separates text by lines. It does not decide whether text is safe, private, or legally reusable. Review the content yourself and ask a trusted adult if rights are unclear. The current AI-generated sample is deliberately tiny and is only for exercising code; it is not a meaningful language-training corpus.

