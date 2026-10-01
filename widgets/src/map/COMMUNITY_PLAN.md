# Makepad map as a community project: a proposal

Status: **draft for discussion** (OctoSense fork, 1 Oct 2026). Nothing here is decided. The map code is upstream makepad's work, so step 0 is a conversation with the makepad maintainers.

## Where things are today

| Part | Location |
| --- | --- |
| Renderer, `MapView`, nav modes (`2d`, `3d`, `plan`, `follow`, `follow3d`) | `widgets/src/map/` (`view.rs`, `tile.rs`, `geometry.rs`, `style.rs`, `label.rs`, `icons.rs`, `drape.rs`, `nav.rs`, `overlay.rs`, `tile_draw.rs`) |
| Bake pipeline: OSM PBF → z0–14 vector tiles, baked fill triangulation, `.mkmap` writer, routing and search databases | `libs/map_build/` (`osm_pbf.rs`, `native/`, `native/bake.rs`, `versatiles.rs`, `repack.rs`, `mkmap.rs`, `nav_build.rs`) |
| Command-line tools and planet/Europe scripts | `tools/map_tiles/` (see its README), `tools/map_bake/` |
| `.mkmap` reader | `libs/mbtile_reader/src/mkmap.rs` |
| Offline routing and place search | `libs/map_nav/` |
| Example apps | `examples/map`, `apps/route` |

Techniques already in place:
- A zoom pyramid (z0–14) with overzoom keyframes past z14.
- Baked polygon triangulation stored as an additive protobuf field that MVT readers ignore.
- Retained per-tile draw lists.
- 2.5D axonometric building extrusion.
- 3D terrain from Terrarium elevation.
- Hillshade with landcover drape.
- `.mkmap`: a sharded container with Hilbert-ordered tile ids, Brotli-packed directories, and positioned (range-read) access.

Known gaps:
- The 3D drive view loads tiles but draws none.
- The camera never zooms out far enough for distant routes.
- There is no general GeoJSON or GIS layer API.
- POIs are thin, because the VersaTiles/Shortbread base keeps a lean tag set.
- There is no satellite imagery.
- There is no low-end ("lite") rendering profile.
- The only hosted archive is a dated file on makepad.nl, the authors' demo server; old ones are deleted.

## 1. Community structure

- **Packaging:** offer the map as its own repo or crates:
  - client SDK (renderer, `MapView`, offline routing and search);
  - bake pipeline;
  - a published `.mkmap` **format spec**.
- **Format choice:** either publish `.mkmap`, or carry the baked-geometry extension on **PMTiles**, which has an existing community, tooling and CDN recipes. Decide this with the maintainers.
- **Public builds:** run a reproducible bake in CI, and publish benchmarks next to it: bytes per tile and milliseconds per tile (the existing Amsterdam bake report), plus frame time and memory on named reference devices, including a low-end one. Contributions then come with numbers.
- **Licences:** code under makepad's licence. Data under ODbL, so the derived database stays open. Every extra source declares its own licence.

## 2. Cheap serving: bake on the server, serve static files

The client already reads byte ranges of static files, so a plain CDN works and nothing runs per request.
- **Incremental bakes:** consume OSM's minutely or daily diffs, re-bake only the changed tiles, and publish small updates clients can patch in.
- **Deduplication:** store identical tiles (ocean, empty land) once.
- **Hosting:** object storage with zero egress (for example Cloudflare R2), or an open-data hosting programme (AWS Open Data, Source Cooperative).
- **Stable URLs:** versioned archives behind a stable "latest" manifest, never a hard-coded dated file. Apps take the URL from config.

## 3. Richer data

- **POIs:** the pipeline already has a `full` profile that keeps every OSM tag. Ship a richer POI layer as an optional, per-region download.
- **Places, addresses, buildings:** Overture Maps Foundation releases open places, buildings and addresses. Check each theme's licence.
- **Terrain:** Terrarium tiles (as in AWS Terrain Tiles) are already read. Copernicus DEM (30 m, global) is another open source.
- **Imagery** is the hard part. Open, global, high-resolution imagery is scarce:
  - Sentinel-2 cloudless mosaics give about 10 m globally (the licence varies by year; some are non-commercial);
  - USGS NAIP gives about 1 m for the US only, in the public domain;
  - OpenAerialMap holds community drone and aerial imagery, with patchy coverage.

  Commercial imagery (Esri, Mapbox, Google) cannot be reused. Expect "medium-zoom global context, sharp only where open data exists". Imagery is also the most storage-heavy layer.

## 4. Crowdsourcing: an open, privacy-respecting Waze

Two kinds of data, two homes:
- **Permanent facts** (new shops, hours, speed limits, missing roads) go back to **OpenStreetMap**, through its editing API with the contributor's own account. The model is StreetComplete: small, local "quests" with clear answers. They appear on everyone's map at the next incremental bake.
- **Live, short-lived data** (traffic speed, hazards, closures) must not go into OSM. It needs a small real-time service, the only part a CDN cannot serve. The privacy design:
  - opt-in;
  - speeds aggregated on the device;
  - uploads only when enough contributors share a road segment;
  - no raw tracks;
  - reports that expire;
  - moderation against spam and fake reports.
- **OctoSense:** a Maps app agent can turn a photo of a shop front into suggested OSM tags, which the person approves before anything is submitted. This follows ADR 0002 (event-driven app agents) and ADR 0004 (approvals by the person).
- **Fun:** quests near you, streaks, local leaderboards, and seeing your edit land on the map.

## 5. Low-end devices

Every GPU renderer meets the same driver bugs; the difference is exposure and fallbacks.
- **API fallback:** offer OpenGL ES alongside Vulkan on Android, where low-end Vulkan drivers are weaker.
- **Driver list:** keep a small known-bad-driver list.
- **Lite profile:** 2D or 2.5D, effects off (`water_fx`, `foliage_fx`, shadows), and a smaller tile budget, chosen automatically on weak GPUs.
- **Measure first:** time to first map, sustained frame and GPU time in 2D, 2.5D and 3D, memory, and battery over 10 minutes. Compare Vulkan with OpenGL ES, and effects on with effects off, on reference devices (for example a OnePlus 6), ideally against a Mapbox demo of the same view.

## Suggested order

1. A self-hosted CDN archive with a stable manifest. This also unblocks OctoSense #165, which currently points at makepad.nl.
2. The format spec, public bakes and benchmarks.
3. The low-end lite profile, plus the OpenGL ES fallback investigation.
4. Optional POI and address layers.
5. StreetComplete-style OSM quests.
6. Live traffic and hazards, last, because it is the only part that needs running servers.

## Open questions

- Do the makepad maintainers want the map as a separate community project, and under what governance?
- Is the format `.mkmap` or PMTiles plus an extension?
- What do planet bake compute, storage (imagery especially) and the real-time service actually cost? No numbers exist yet; the first bake on our own infrastructure should produce them.
- What is the licence matrix across OSM (ODbL), Overture themes, DEM and imagery sources?
