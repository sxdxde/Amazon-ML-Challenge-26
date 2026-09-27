# RUN.md — entity_resolution.ipynb on the M1 Pro (16 GB)

All commands run from `student_resource/`.

## 1. Setup

```bash
bash setup_mac.sh
```

- The script aborts if the interpreter is x86_64 (Rosetta). If that happens, rerun it with `PYTHON=/path/to/arm64/python3.12`.
- It creates or refreshes `.venv_mlx`, installs the pins from `requirements.txt`, and registers the Jupyter kernel **er-mac**.
- If it prints `brew install libomp`, run that command, then rerun `bash setup_mac.sh`.

Then open Jupyter and select the **er-mac** kernel:

```bash
.venv_mlx/bin/jupyter lab entity_resolution.ipynb
```

Notes:
- `mlx-embeddings` is **not installed**: it is GPL-3.0, which the rules forbid. The bge-m3 encoder is implemented in `mlx.nn` in `er_mlx_encoder.py`. It loads the same `mlx-community/bge-m3-mlx-fp16` weights, and cell 1a re-checks it against sentence-transformers.
- Model weights are already cached in `hf_cache/`; the notebook sets `HF_HOME` to point there.

## 2. Prepare the Mac

- Plug in the power adapter.
- Turn **off** Low Power Mode (System Settings → Battery).
- Close browsers, Docker, IDE indexers and other heavy apps. The GPU shares the 16 GB of RAM.
- Free disk space: the full run needs about **40 GB**, mostly test embeddings (~18 GB at `EMB_DIM=256` with 3 fields) and the feature parquet files.

## 3. First run: smoke mode

1. In cell 0, keep `SMOKE = True`.
2. Run the cells top to bottom (Run → Run All Cells).
3. Paste back to Claude Code **every** line starting with `>>> PASTE TO CLAUDE:`, plus any `❌ FAIL` line and any traceback.

What to expect:
- Smoke mode uses about 300 train S1 and 200 test S1 (all countries kept), 3 folds and 2 MLP epochs.
- Writes go to `artifacts_smoke/`, `output_smoke/` and `reports/summary_smoke.md`.
- Cells 2 and 3 still read the full TSVs (~1.5 GB of text), so each takes a few minutes.
- Cell 13 validates both `output_smoke/` (every test S1 has a row; unsampled ones are empty) and a full-size all-empty `output_smoke/full_dummy/`.

If **cell 1a FAILs**, set `USE_ST_MPS = True` in cell 0 and rerun from cell 0.

After a fix, Claude will tell you which cells to rerun. Cached stages are skipped automatically when their parameters are unchanged.

## 4. Full run

In cell 0:
1. Set `EMB_BATCH` to the best batch size printed by cell 1b.
2. Set `SMOKE = False`.
3. Run all cells.

The full-run time is dominated by embedding ~11.7M test records plus the train sample. Cell 1b prints the projection:
- If it is too long, set `EMB_FIELDS = ('name', 'addr')`. That drops the longest texts (~45% of the time), and dense retrieval then uses `[name; addr]`.
- `TRAIN_S1_N = 150_000` keeps the training matrix under ~3 GB so peak RSS stays below ~11 GB. Lower it if memory pressure turns red.

For a headless run that keeps the Mac awake:

```bash
mkdir -p runs logs
caffeinate -dimsu .venv_mlx/bin/papermill -k er-mac entity_resolution.ipynb runs/full_run.ipynb 2>&1 | tee logs/full.log
```

If the run is interrupted, just rerun. Every stage resumes from `artifacts/`:
- embeddings resume per 50k-record chunk;
- blocking and features resume per country group;
- the models resume per stage.

`FORCE = True` recomputes everything.

## 5. Monitoring

- **Activity Monitor → Memory:** the Memory Pressure graph should stay green or yellow. Red means swapping; lower `TRAIN_S1_N`, `EMB_BATCH` or `FEAT_CHUNK`.
- **GPU load:** `sudo powermetrics --samplers gpu_power -i 2000`.
- Each notebook section prints its wall time, current RSS and peak RSS. Cell 14 tabulates them in `reports/summary.md`.

## 6. Packaging

```bash
DEST=code/business_entity_resolution/src
mkdir -p "$DEST"
cp entity_resolution.ipynb setup_mac.sh requirements.txt RUN.md \
   er_common.py er_text.py er_mlx_encoder.py er_embed.py er_block.py er_feats.py er_models.py er_decide.py er_lgb_worker.py "$DEST"/
.venv_mlx/bin/jupyter nbconvert --to script "$DEST/entity_resolution.ipynb"
```

Notes:
- `code/business_entity_resolution/src/` already holds `.py` files from the earlier pipeline (`blocking.py`, `train_gbm.py`, …). Decide whether to keep them or remove them before zipping, so the submission contains one coherent entry point.
- Fill in `Documentation_template.md` using `reports/summary.md` and the methodology cell at the end of the notebook.

Then zip:

```bash
zip -r submission.zip code/business_entity_resolution output/matching_results.tsv output/candidate_pairs.tsv Documentation_template.md
```

Before submitting, rerun the validator:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

## Licenses: please confirm on the Hugging Face cards

| Repo loaded | License | Base model | License |
|---|---|---|---|
| `mlx-community/bge-m3-mlx-fp16` | MIT | `BAAI/bge-m3` | MIT |
| `BAAI/bge-m3` (sentence-transformers: parity check / `USE_ST_MPS`) | MIT | — | — |
