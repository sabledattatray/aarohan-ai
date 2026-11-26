# Environment inspection

Inspection date: 2026-11-25 (workspace local date).

## Initial workspace inspection

- Workspace contains `AAROHAN_Master_PRD.txt` and the pre-existing `gpu_test.py`.
- The PRD is populated (60,757 bytes) and identifies the project as proposed, with Phase 0 not yet implemented.
- `git` is available on PATH, but the workspace is not currently a Git repository.
- `python`, `py` and `uv` were not found on PATH in the isolated workspace shell.
- Windows hardware inventory queries via CIM were denied in this execution environment.
- The pre-existing `gpu_test.py` requires CUDA and exits with failure on CPU-only machines, so it is not a general environment check.
- Git metadata was initialized, but Git status from the sandbox is blocked because the project is owned by `DATTA/sable` while the sandbox runs as `DATTA/CodexSandboxOffline`. No global `safe.directory` exception was added.

## User-provided virtual-environment report

The user supplied output from `C:\Users\sable\.venv` on 2026-11-25:

- Python 3.12.10, Windows 11 x64, AMD processor family 23 model 113, 16 logical CPUs.
- PyTorch 2.14.1+cu132; CUDA 13.2 available.
- NVIDIA GeForce RTX 2060, 6,442,123,264 bytes reported VRAM (about 6 GiB).
- RAM could not be read by the current report script.
- The follow-up environment report confirmed the same Python/PyTorch/CUDA/GPU details, with `ram_bytes: null` because optional `psutil` was not installed. `psutil` is now included in the project requirements so the next report can read installed system RAM.
- NumPy was not installed and produced a PyTorch warning. It has been added to `requirements.txt`.
- Pytest 8.4.2 was subsequently installed from the project requirements.
- User-reported `python -m pytest`: 4 tests passed in 6.92 seconds.
- The first direct CUDA smoke-run attempt failed before training because direct script execution did not include the repository root on `sys.path` (`ModuleNotFoundError: aarohan`). The script now explicitly adds its project root before importing the package.
- User-reported CUDA smoke run after that fix: RTX 2060, 135,488 parameters, 20 steps, initial synthetic training loss 4.204927, final training loss 3.402413, validation loss 3.374615, and checkpoint reload maximum logit difference 0.0. The first error shown in the pasted output is the pre-fix attempt; the second invocation succeeded.
- Follow-up environment report: 17,058,258,944 bytes RAM (about 15.9 GiB), reported by psutil.
- User-reported original toy corpus validation passed: 10 training paragraphs and 5 validation paragraphs; training SHA-256 `650622074b4ae08d0511a296f0df21c3ec76b556c3253923f781b7b2587ce745`; validation SHA-256 `ba8375ecfa181133987b247394426615b8917f5e4b51b4ce60889f089ea440d9`. Provenance manifest identifies the corpus as AI-generated; it is only a toy pipeline dataset.
- User-reported tokenizer validation: 9 tests passed in 3.64 seconds; AAROHAN-BPE 0.1.0 trained successfully with actual vocabulary 265 (256 byte symbols plus required special tokens), using only the training corpus. Tokenizer metadata records training corpus and provenance manifest hashes.
- User-reported complete tokenizer/data-preparation validation: 10 tests passed in 2.79 seconds; prepared distinct tensors with 474 train tokens and 235 validation tokens. Corpus manifest SHA-256 `612bc7c0dcc3375bb7722d2d14acf1f9ab323a0f5799e0be41078ecfd8b44a53`; tokenizer SHA-256 `7b909f2e73fc308da37d18a4693cf9168d6796fe99328e547af1f1b029163521`; validation text was not used for tokenizer training.
- User-reported full check after trainer test was added: 11 tests passed in 4.15 seconds. CUDA toy-corpus run on RTX 2060: 148,352 parameters, 100 steps, context 32, batch 8; training loss 5.598822 -> 3.092667; held-out validation loss 3.144345; checkpoint reload max logit difference 0.0. This is an AI-generated toy-corpus pipeline result, not evidence of general language capability.
- User-reported local generation milestone: 13 tests passed in 3.43 seconds; `generate_local.py` loaded the checkpoint and tokenizer with matching SHA-256 on CUDA and produced 40 generated tokens. Output was garbled, as expected from only 474 toy training tokens.
- User-reported full test suite after hardware benchmark and registry work: 22 tests passed in 3.99 seconds.
- User-reported RTX 2060 hardware baseline (PyTorch 2.14.1+cu132, CUDA 13.2, batch 2, 2 warmup + 10 measured steps): tiny 148,352-parameter model at context 32 achieved 6,463.7 tokens/sec with 20,132,352 bytes peak allocated; the 6,490,656-parameter configuration (4 layers, 6 heads, width 288, vocab 4096, context 256) achieved 26,918.6 tokens/sec with 200,828,416 bytes peak allocated and 236,978,176 bytes peak reserved. Random-token benchmark only; not model quality.
- User-reported AAROHAN-0.01 architecture smoke run: 22 tests passed in 3.89 seconds; 5,387,328 parameters, 4 layers, 6 heads, width 288, context 64, batch 4, 20 CUDA steps. Loss 5.599546 -> 3.014965; full 234-target validation loss 3.186465; exact checkpoint reload (0.0 max logit difference). Toy data only.
- User-reported larger-checkpoint inference: `generate_local.py` loaded the 5,387,328-parameter checkpoint with matching tokenizer SHA-256 and generated 20 tokens on CUDA. Output was garbled, as expected from the toy training data.
- User-reported AAROHAN-0.01 toy comparison: step-100 small model validation loss 3.168759 (perplexity 23.7779), step-120 small model 3.018733 (20.4654), and 5.39M-parameter model after 20 steps 3.186465 (24.2027), all on the same 234-token split. The larger candidate has fewer training steps, so this does not compare architecture quality; it remains unpromoted. Registry ID `71d73ccf20cff5f8`.
- User-reported memory CLI check: 28 tests passed in 4.18 seconds; one non-sensitive, unverified project note persisted and was found by keyword search in a separate CLI invocation.
- User-reported document-ingestion check: 31 tests passed in 4.82 seconds; `architecture.md` ingested into 14 chunks, all marked unverified and source-linked with the document SHA-256. Keyword search for `component contracts` returned source-attributed chunks. No document instructions were executed.
- User-reported HTML ingestion test suite: all 32 collected tests reached 100% with no failures shown (final pytest summary line was not included in the pasted output).
- User-reported PDF ingestion suite: 33 tests passed in 4.88 seconds. PDF test uses a mocked PdfReader, confirming page attribution and encrypted-file rejection logic; a real PDF extraction run has not yet been reported.
- User searched the project recursively for `*.pdf` and found none. The attempted `your-real-file.pdf` command used a placeholder path; no real PDF was ingested.
- User-reported memory source-filter test run: 34 collected tests reached 100% with no failures shown (final pytest summary line omitted).
- User-reported sensitive-memory guard test run: 35 collected tests reached 100% with no failures shown (final pytest summary line omitted).
- User-reported local orchestration check: 36 tests passed in 4.49 seconds; `chat_local.py` retrieved three unverified source-linked architecture memories and generated a separate 40-token continuation from the 5.39M toy checkpoint on CUDA. Output was visibly garbled, as expected; retrieved evidence was not injected into the model prompt.
- User-tested real PDF integration: created a one-page original PDF, ingested one chunk, then retrieved the exact text using source-filtered search. The record retained page 1, file SHA-256 `4b7e076e1a9b46e4c371fabd2b4c7b92d1c7ddfb9523eb3592192f58f470a5c2`, and unverified status. PDF-to-memory path confirmed end to end.
- User-reported WikiText-2 download succeeded after adult review: pinned revision `6231e49f19a707241d6b84d9cff60a3a86b85a85`, raw subset `wikitext-2-raw-v1`; train 36,718 rows (SHA-256 `49280f3dd993cbdfd322186194133d5d1935f81e1c91064c239f2a70ebb02701`), validation 3,760 rows (`a10eb5a42c4264c45dd38bfef4da6a74ddef5fc22cb419e1a89d54c4cc6d748f`), test 4,358 rows (`3ee89cd6a2ab912afd5d01e98867b46a67d9ec7a9eca0910e5e4c3cdd4cc1925`). Test split remains separate. Hugging Face cache reported Windows symlink support unavailable; files still downloaded successfully with higher possible cache disk use.
- User-reported resume implementation tests: 16 tests passed in 4.02 seconds, including deterministic CPU equivalence between uninterrupted and checkpoint-resumed training.
- User-reported CUDA resume: resumed the 100-step checkpoint to 120 total steps on RTX 2060; initial resumed loss 3.007076, final loss 2.846033, validation loss 2.990457, reload max logit difference 0.0. Output checkpoint was a separate `toy-corpus-resumed.pt` file.
- User-reported checkpoint evaluation: 17 tests passed in 3.80 seconds. On the complete 234-token held-out stream, step-100 checkpoint loss/perplexity were 3.168759 / 23.7779; step-120 were 3.018733 / 20.4654 (loss delta -0.150025). Both used identical validation and tokenizer hashes. The earlier trainer validation values were slightly lower because that trainer version omitted the final partial window; the trainer has now been updated to score all validation targets and a regression test checks parity with the evaluator.
- User reran the complete suite after the validation-window fix: 17 tests passed in 4.10 seconds.
- User-reported registry milestone: 19 tests passed in 4.57 seconds. Evaluation report saved for the 100/120-step checkpoints; candidate IDs `86ca666c66f9163b` and `43514aef0cfb2430` were registered with matching evaluation evidence. Neither was activated; both are toy-corpus candidates.

These are user-supplied results, not measurements rerun by the workspace process. The 6 GB GPU is a useful target for small experiments; it does not make the PRD's proposed 8-layer, width-512 model configuration automatically feasible for all sequence lengths and batch sizes.

## Next inspection

The user has installed Python and PyTorch in a venv outside the workspace's accessible filesystem. When that venv is attached or these files are run from the user's terminal, run `python scripts/environment_report.py`, `python -m pytest`, and `python scripts/train_tiny.py --device cuda`. Capture output in a dated experiment record. Record GPU driver details if available. GPU tests are conditional; CPU support remains required.

The Codex execution process could see the venv executable but could not launch it due to sandbox setup restrictions. Therefore tests and training have not yet run from Codex. Run them from the activated project terminal and share the output for review.

