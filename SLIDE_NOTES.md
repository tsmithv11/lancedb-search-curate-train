# Slide notes

Differences between the slide code and the current Python APIs used in the notebook (lancedb 0.39, pylance 12, geneva 0.17), so the slides can be updated to match.

| Slide | Notebook | Why |
|---|---|---|
| `lancedb.connect("./data")` | adds `read_consistency_interval=timedelta(0)` and `storage_options={"new_table_enable_stable_row_ids": "true"}` | Materialized views require stable row ids, which are set when the table is created. The interval makes the table handle see writes made by Geneva jobs. |
| `tbl.search("low-light street scene", query_type="hybrid")` | runs as written; to search a different keyword, use `.vector("low-light street scene").text("pedestrian")` | A single string is used for both the vector and the full-text query. |
| `tbl.search("low-light street scene", ...).where("label IS NULL").limit(500)` | `.vector("pedestrians crossing a dark street at night").text("pedestrian").where("label IS NULL AND brightness < 0.35 AND people_score > 0").limit(100)` | Not an API change; a better query for this data. Without the brightness filter, keyword matches on "pedestrian" outweigh the vector side and most top results are daytime scenes. With brightness alone, many are empty night streets. `people_score` is a CLIP zero-shot column (people prompts minus no-people prompts) computed as a LanceDB Function. About 400 unlabeled images pass the filter, so relevance falls off well before rank 500. |
| `tbl.tags.create("night-peds-v1", tbl.version)` | runs as written, on the view's table (`view.table`) | |
| `LanceDataset(ds, batch_size=64, shuffle=True)` | `sampler=ShardedBatchSampler(rank=0, world_size=1, randomize=True)` | `LanceDataset` has no `shuffle` argument. Unknown keyword arguments are ignored, so `shuffle=True` silently does nothing. |
| `DataLoader(LanceDataset(...))` | `DataLoader(LanceDataset(...), batch_size=None)` | `LanceDataset` already yields batches. With the default `batch_size=1`, PyTorch adds an extra leading dimension (1, 64, 512). |
| image column as blob | plain `binary` column | Vector, full-text and hybrid queries can't return Blob API (`lance-encoding:blob`) columns. Plain binary comes back in search results. |

## Local-only calls and their Enterprise equivalents

- **LanceDB Functions.** Function columns (`lancedb.udf`) run on LanceDB Enterprise. Locally the notebook uses `geneva.udf`, `add_columns` and `backfill` with `local_ray_context()`. On Enterprise, connect Geneva with `geneva.connect("db://...", api_key=..., host_override=...)` and the same `backfill` call runs as a managed job.
- **Materialized views.** `db.create_materialized_view(...)` in lancedb is local-only. Geneva provides `create_materialized_view` for Enterprise connections.
- **Whole-table reads.** On Enterprise, `RemoteTable` has no `to_pandas()` or `to_arrow()`. Read through a query instead, for example `tbl.search().where(...).to_pandas()`.
