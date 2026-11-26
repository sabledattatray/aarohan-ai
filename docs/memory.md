# Local memory CLI

The current SQLite memory store is a development baseline. It persists locally, supports keyword search, and labels entries unverified unless a person supplies verification evidence. Sensitive-classified entries are rejected by the storage layer until encryption and access controls exist. It is not the PRD's later PostgreSQL/pgvector memory service.

Use only non-sensitive project information while testing. Do not store passwords, home/school addresses, phone numbers, private conversations, or personal details about other people.

```powershell
python ".\scripts\memory_cli.py" add --content "AAROHAN currently uses a local PyTorch Transformer." --source "project architecture notes"
python ".\scripts\memory_cli.py" search "local Transformer"
python ".\scripts\memory_cli.py" search "component contracts" --source-contains "architecture.md"
```

The database defaults to `data/memory/aarohan.sqlite`. New entries are unverified. A person-verified entry requires both `--source` and `--verified-by`; a generated answer alone is not verification. Use `get`, `correct`, and `delete` with the ID returned by `add` or `search`.

