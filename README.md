# One table, from search to training

A demo notebook for the talk *Build an Open Multimodal Data Stack for Search, Curation, and Training* (Lei Xu). It is the live demo for slide 11, and the "Demo notebook" linked from the closing slide.

[`multimodal-lakehouse-demo.ipynb`](multimodal-lakehouse-demo.ipynb) takes all 5,000 COCO val2017 images through four steps on a single LanceDB table: raw image bytes, captions, metadata, CLIP embeddings and derived features all live in that table. Search, curation, feature engineering and training run on it directly, with no exports in between.

## The four steps

1. **Search.** One hybrid query finds unlabeled night street scenes with pedestrians: CLIP vector search for "pedestrians crossing a dark street at night", full-text search for "pedestrian" over the captions, and the filter `label IS NULL AND brightness < 0.35 AND people_score > 0`. `brightness` is computed from the pixels and `people_score` from the CLIP embedding, both as LanceDB Functions. The rows come back with their image bytes.
2. **Tag a slice.** Near-duplicates are flagged by embedding distance with a LanceDB Function. The low-light, deduplicated hits become a materialized view, `night_peds`, tagged `night-peds-v1`.
3. **Add a column.** A `quality` score is computed in place from the image bytes. The data-file listing shows existing files are not rewritten, and appending rows computes only the new ones.
4. **Train a step.** PyTorch reads `night-peds-v1` directly through `lance.torch.data.LanceDataset`. A linear probe trains on the CLIP embeddings to predict whether a scene has a vehicle, and its checkpoint records the table version. After more data arrives, checking out the tag reproduces the training data exactly.

## Run it

**Locally** (tested with Python 3.12 on macOS, CPU only):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab multimodal-lakehouse-demo.ipynb
```

**In Colab:** open the notebook and run all cells. The first cell installs the packages. If pip asks you to restart the session, restart and run all again.

The first run downloads about 525 MB of COCO data (Lance format, from [`lance-format/coco-captions-2017-lance`](https://huggingface.co/datasets/lance-format/coco-captions-2017-lance)) and about 600 MB of OpenCLIP ViT-B-32 weights. Both are cached, so later runs skip the downloads. No Hugging Face login is needed.

Runtime: about 3 minutes on an Apple M-series laptop after the downloads. Most of the time goes to computing CLIP embeddings for 5,000 images. On a 2-vCPU Colab CPU runtime that step is several times slower; a GPU runtime is used automatically when one is available. Lower `NUM_IMAGES` in section 1 to trade dataset size for time.

The notebook writes to `./data` (the LanceDB database, recreated on each run), `./.cache` (downloads) and `./checkpoints`.

## LanceDB Functions and Geneva

The notebook calls its Python UDF columns (`brightness`, `people_score`, `dup_of`, `quality`) LanceDB Functions. Function columns run on LanceDB Enterprise. To run on a laptop or in Colab, the notebook uses the [`geneva`](https://docs.lancedb.com/geneva) package, which has the same declare, attach, backfill pattern and runs it on a local Ray instance.

## Editing the notebook

The notebook is generated from [`tools/build_notebook.py`](tools/build_notebook.py) so the cell sources are easy to review and diff. To change a cell, edit that file, then rebuild and re-execute:

```bash
python tools/build_notebook.py
jupyter nbconvert --to notebook --execute --inplace multimodal-lakehouse-demo.ipynb
```

## Production

[LanceDB Enterprise](https://docs.lancedb.com/enterprise) runs the same tables as a distributed service with data in object storage. See the [architecture overview](https://docs.lancedb.com/enterprise/architecture) and the [Geneva docs](https://docs.lancedb.com/geneva).
