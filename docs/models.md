# Models used — license & size register

Every model was checked against its Hugging Face model card / API metadata on 2026-09-25
(`https://huggingface.co/api/models/<id>` → `license:` tag and `cardData.license`).
Rule: MIT or Apache-2.0, ≤ 8B parameters. Pretrained weights only; no external data.

| Model | Role | License (HF card) | Params | Revision (sha at check time) |
|---|---|---|---|---|
| `BAAI/bge-m3` | dense retrieval embeddings (blocking) + cosine feature | MIT | ~568M (XLM-R large, 24L×1024, vocab 250k) | `5617a9f61b028005a4858fdac845db406aefb181` |
| `Qwen/Qwen3-Embedding-0.6B` | alternative embedder (not default) | Apache-2.0 | 595,776,512 | `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3` |
| `BAAI/bge-reranker-v2-m3` | cross-encoder, fine-tuned as pair classifier | Apache-2.0 | 567,755,777 | `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e` |
| `intfloat/multilingual-e5-small` | smoke / dev-profile CPU embedder | MIT | 117,654,272 | `614241f622f53c4eeff9890bdc4f31cfecc418b3` |
| `cross-encoder/ms-marco-MiniLM-L6-v2` | smoke / dev-profile CPU cross-encoder (fine-tuned the same way) | Apache-2.0 | 22,714,113 | `233902d25c440f23af6f7d6e94d2946bac0bee0a` |

Non-neural learners: LightGBM (MIT) gradient-boosted trees trained from scratch on the provided train data
— or, on Linux systems with glibc < 2.28 where LightGBM cannot be installed, XGBoost 2.1.4 (Apache-2.0)
with equivalent leaf-wise settings (`src/gbm_backend.py`, chosen automatically and logged in the run);
scikit-learn (BSD-3) TF-IDF / isotonic regression.

No model is used to look up, geocode or enrich records; all models only encode the provided text.

The **full** (submission) run uses only `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3`; the small models
are used only by the `smoke`/`dev` profiles (config.py) for CPU testing. `Qwen/Qwen3-Embedding-0.6B`
was license-checked as an alternative embedder but is not used.
