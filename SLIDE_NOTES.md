# Slide notes

Notes on the code in the deck *Open Multimodal Data Stack*, checked against lancedb 0.39, pylance 12 and geneva 0.17.

## Slide code as of October 5, 2026

The code on these slides was corrected to match the current APIs.

| Slide | Was | Now | Why |
|---|---|---|---|
| LanceDB: an open multimodal lakehouse | `image` type `blob` | `image` type `binary` (`video` stays `blob`) | Vector, full-text and hybrid queries can't return blob columns: `.to_pandas()` raises an error. Plain binary comes back in search results, so "rows come back with image bytes" holds. |
| Pattern 1: Search | `tbl.search("low-light street scene", query_type="hybrid")` | `tbl.search(query_type="hybrid").vector("low-light street scene").text("pedestrian")` | A single string is sent to both the vector and the full-text query, so the old code never searched captions for "pedestrian". |
| Pattern 1: Search | `.where("label IS NULL")`, panel "label is empty" | `.where("label IS NULL AND brightness < 0.35")`, panel "unlabeled, brightness < 0.35" | Optional, matches the demo. Without a brightness filter, most top results are daytime scenes. |
| Pattern 2: Curation | `tbl.tags.create(...)`, `tbl.checkout(...)` | `night_peds.tags.create(...)`, `night_peds.checkout(...)`, comment "(the night_peds view)" | Optional. Both forms run. Tagging `tbl` freezes the whole table, so naming the view makes "freeze this slice" literal. |
| Pattern 3: Feature engineering | no code | `tbl.add_columns({"quality": quality})`, `tbl.backfill("quality")` | New. This is the Geneva pattern; `backfill` defaults to `where="quality IS NULL"`, so it computes only rows without a value. |
| Pattern 4: Training | `open_table("train")` | `open_table("train", version=37)` | Ties the code to the "ckpt-0412 ↔ table v37" panel. `version=` also accepts a tag name. |
| Pattern 4: Training | `DataLoader(Permutation.identity(ds), batch_size=64, shuffle=True)` | `p = Permutation.identity(ds).select_columns(["image", "label"])`, then `DataLoader(p, batch_size=64, shuffle=True)` | Both run. Without `select_columns`, every batch reads every column of the table. |

The new card on Pattern 3 uses a solid orange border and slightly rounder corners than the other cards. The Slides API can't create the gradient border or set the corner radius used elsewhere in the deck.

## Slides added on October 5, 2026

Three slides were added, taking the deck from 13 to 16 slides.

| Slide | Position | Where the content comes from |
|---|---|---|
| Lance vs Parquet on AI workloads | after "Three properties that make it work" | **75× and 61×:** Xuanwo, [Lance Format v2.2 Benchmarks](https://www.lancedb.com/blog/lance-format-v2-2-benchmarks-half-the-storage-none-of-the-slowdown), LanceDB blog, April 6, 2026. These are geomeans across datasets for Lance v2.2 vs Parquet (Rust crate 57.2.0) on local NVMe, EC2 c7i.4xlarge. The post says: "Lance v2.2 is 75x faster than Parquet for blob fetches"; "On LAION-10M alone, Parquet needed 889 seconds to fetch 1,000 random images; Lance v2.2 finished the same workload in single-digit seconds"; "Lance v2.2 is 61x faster than Parquet for dataset backfills when adding new columns"; "Lance adds a new column in 13 ms, while Parquet takes 520 seconds to rewrite the dataset". The same post's summary table says "~68x" for blob access, so expect that number from anyone who has read it. The post also reports Parquet 40% faster on narrow column projections, so the slide doesn't claim Lance wins every scan. **43×:** Chaurasia, [A Practical LLM Pretraining Pipeline with LanceDB](https://www.lancedb.com/blog/practical-llm-pretraining), LanceDB blog, September 15, 2026. Training GPT-2 124M on 8 H100s from S3, Lance reads 3.16M tokens/s and random Parquet reads reach 74k. Pre-shuffled Parquet and MosaicML Streaming keep pace. That is why the footnote says the win is a global shuffle with no pre-shuffled copy. |
| Pattern 2: Dedupe and clusters | after Pattern 2: Curation | Computed from the demo table (`./data/images`). **Near-duplicates:** the notebook's `dup_of` column flagged 150 of 4,950 rows: 146 of the 150 synthetic burst copies, plus 4 pairs already in COCO. The church photos are COCO 000000481404 and 000000471991, at a CLIP cosine distance of 0.034. **Clusters:** k-means (k = 8, scikit-learn, `random_state=0`) on the normalized `clip_emb` of the 5,000 COCO val2017 images. Cluster names come from the most distinctive caption words in each cluster. Each thumbnail is the 640×480 image nearest its cluster's centroid. "Low-light street scenes with people" filters the street cluster to `brightness < 0.35 AND people_score > 0`, which leaves 83 images. The photos are inserted from `images.cocodataset.org`. |
| Pattern 5: Evaluation | after Pattern 4 | Closes the loop on slide 3, which lists Evaluation as a stage. The `add_columns` and `backfill` calls follow the Pattern 3 form. The query `tbl.search("pedestrians at night").where("pred_0412 != label").limit(200).to_pandas()` was run against a copy of the demo table, with `pred_0412` filled in by `tbl.update` in place of a model. It returns only labeled rows where the prediction differs, because rows with a null label fail the comparison. |

## Caveats the slides don't show

- **Pattern 4 uses Geneva.** `db.create_materialized_view("train", q, auto_refresh=True)` and `mv.refresh()` match Geneva's API: `db = geneva.connect(...)`, and `q` is a query on a Geneva table. LanceDB's own `create_materialized_view(name, source, *, select, where, limit)` takes a table name instead of a query and has no `auto_refresh`. `auto_refresh=True` takes effect only on LanceDB Enterprise (`db://`). I checked these Geneva calls against their signatures but didn't run them.
- **"New rows computed on arrival" (Pattern 3) needs Enterprise.** With `auto_backfill=True` on the UDF, LanceDB Enterprise backfills new rows automatically. On a local or object-storage connection you call `backfill()` yourself.
- **If `image` goes back to `blob`.** End the search with `.to_arrow()` instead of `.to_pandas()`, then `images = tbl.fetch_blobs("image", results)`. That returns the bytes for exactly those rows from the same table (tested).

## How the notebook differs from the slides

The notebook (`multimodal-lakehouse-demo.ipynb`) runs the same four patterns locally, with these differences:

| Topic | Slides | Notebook | Why |
|---|---|---|---|
| Connect | not shown | `lancedb.connect("./data", read_consistency_interval=timedelta(0), storage_options={"new_table_enable_stable_row_ids": "true"})` | LanceDB's local materialized views require stable row ids, set when the table is created. The interval lets the table handle see writes made by Geneva jobs. |
| Search query | vector "low-light street scene", `brightness < 0.35`, `limit(500)` | vector "pedestrians crossing a dark street at night", filter adds `people_score > 0`, `limit(100)` | Tuned for 5,000 COCO images. With brightness alone, many top results are empty night streets. `people_score` is a CLIP zero-shot column computed as a LanceDB Function. About 400 unlabeled images pass the filter, so relevance falls off well before rank 500. |
| Materialized view | Geneva `create_materialized_view("train", q, auto_refresh=True)` | LanceDB `db.create_materialized_view("night_peds", "images", select=..., where=...)` | LanceDB's version runs locally without a Ray job. Geneva's is the one that works with Enterprise. |
| Training loader | `Permutation` + `DataLoader(shuffle=True)` | `lance.torch.data.LanceDataset` with `sampler=ShardedBatchSampler(rank=0, world_size=1, randomize=True)` and `DataLoader(..., batch_size=None)` | Both work. |
| Reading a version | `open_table("train", version=37)` | `lance.dataset(uri, version="night-peds-v1")` | Same idea. Both accept a version number or a tag name. |

## Pitfalls found while building the demo

- **`LanceDataset(ds, batch_size=64, shuffle=True)` doesn't shuffle.** There's no `shuffle` argument, and unknown keyword arguments are silently ignored. Pass a sampler such as `ShardedBatchSampler(..., randomize=True)`.
- **`DataLoader(LanceDataset(...))` needs `batch_size=None`.** `LanceDataset` already yields batches. With the default `batch_size=1`, tensors come out as (1, 64, 512).
- **The docs' blob recipe is out of date.** The `lance-encoding:blob` field metadata is rejected for file format 2.2 and later. Use `lancedb.blob("image")` (blob v2) instead.
- **`create_fts_index` is deprecated.** Use `tbl.create_index("captions", config=FTS())`.
- **`lancedb.streaming.StreamingDataset` crashed on the demo's tables.** It panics in Rust ("target schema is not superset of current schema") on both demo tables, though not on fresh tables or on copies of the same data. The trigger seems to be the tables' write history. Keep it off slides until that's resolved.

## Local-only calls and their Enterprise equivalents

- **LanceDB Functions.** Function columns (`lancedb.udf`) run on LanceDB Enterprise. Locally the notebook uses `geneva.udf`, `add_columns` and `backfill` with `local_ray_context()`. On Enterprise, connect Geneva with `geneva.connect("db://...", api_key=..., host_override=...)` and the same `backfill` call runs as a managed job.
- **Materialized views.** LanceDB's `db.create_materialized_view(...)` raises "supported only on local databases" on remote connections. Geneva's `create_materialized_view` works with Enterprise connections.
- **Whole-table reads.** On Enterprise, `RemoteTable` has no `to_pandas()` or `to_arrow()`. Read through a query instead, for example `tbl.search().where(...).to_pandas()`.
