#!/usr/bin/env bash
# Download the 60 s sample of the forest walks dataset, complete one scan per second with FOSCo-Net 2.0 and 1.0,
# and render figures of the first three scans.
set -euo pipefail
fosconet download sample --out data
fosconet download segmenter
fosconet run data/companion/sample/walks_2026-08-12_sample60s --out results --every 20 --figures 3 \
    --map data/companion/map/walks_2026-08-12_map_local_enu.pcd --ply
echo "figures in results/figures, point clouds in results/predictions/*/*.ply"
