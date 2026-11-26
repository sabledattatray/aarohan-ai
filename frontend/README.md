# AAROHAN AI - Next.js & Tailwind Web Interface

This is the Next.js frontend for the AAROHAN AI Platform as specified in PRD Prompt 15.

## Features
- AI Chat with hybrid retrieval evidence display.
- Separation of neural outputs, memory records, and verification status badges.
- Memory Explorer, Knowledge Graph, Learning Queue, and Model Registry integration.
- Proxies requests to FastAPI backend on `http://127.0.0.1:8000`.

## Quick Start
```bash
# 1. Install dependencies
npm install

# 2. Run development server
npm run dev
```

Visit [http://localhost:3000](http://localhost:3000) to view the interface.
Note: You can also access the embedded web interface directly from the FastAPI server at [http://127.0.0.1:8000/app](http://127.0.0.1:8000/app).
