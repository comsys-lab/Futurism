# File-by-file source audit

This audit covers every native source/header and both build files in the prepared
Futurism repository. Comparisons were taken before adding the provenance comments.
Only attribution comments and documentation were changed during the audit.

## Pinned references and method

- Kubism: [`0575f83773d7a37afdee967e50a394da3fcec267`](https://github.com/comsys-lab/Kubism/tree/0575f83773d7a37afdee967e50a394da3fcec267), committed `2025-07-14T12:16:04Z`.
- KMCUDA: [`7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b`](https://github.com/src-d/kmcuda/tree/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b), committed `2022-10-11T16:58:55Z`.

The local Futurism base is `8abb12cab22310fe4b96cbd4f8e04bc02a61b898`.
Its history contains the pinned Kubism commit as an ancestor. The prepared source
also includes local uncommitted source changes and the documented build/default
configuration changes; it is not a clean export of that base commit.

The official pinned GitHub trees are the comparison references. A separate local
Kubism copy was also inspected, but differs from the official tree and is not used
as the canonical baseline. Newer commits on other local Futurism branches were not
silently substituted for the prepared snapshot.

Byte equality was checked first. For different files, line differences were
inspected. Reference/pre-notice/current SHA-256 values, line counts and matching
line counts are recorded in [SOURCE_AUDIT.json](SOURCE_AUDIT.json). Matching lines
are textual evidence, not a percentage of intellectual contribution or ownership.
The `published_sha256` values identify the files at the time of this audit and
will naturally change if sources are edited later.

## Results before attribution comments

| Published file | Compared with Kubism | Compared with KMCUDA | Observed relationship |
| --- | --- | --- | --- |
| [`src/CMakeLists.txt`](../src/CMakeLists.txt) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/CMakeLists.txt) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/CMakeLists.txt) | Native-only build configuration rewritten for this repository. |
| [`src/fp_abstraction.h`](../src/fp_abstraction.h) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/fp_abstraction.h) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/fp_abstraction.h) | Only blank lines differ from KMCUDA; inherited unchanged from Kubism before the new notice. |
| [`src/kmcuda.cc`](../src/kmcuda.cc) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/kmcuda.cc) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/kmcuda.cc) | Kubism host path retained; Futurism adds raw-sample forwarding and optional power-hook calls. |
| [`src/kmcuda.h`](../src/kmcuda.h) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/kmcuda.h) | [Identical](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/kmcuda.h) | Public API header preserved byte-for-byte, including upstream project references. |
| [`src/kmeans.cu`](../src/kmeans.cu) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/kmeans.cu) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/kmeans.cu) | KMCUDA/Kubism kernels retained and extended; CUB compaction, task partitioning, raw-sample interface, transposed centroids, buffer initialization and timing changes. |
| [`src/kmeans_cpu.cc`](../src/kmeans_cpu.cc) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/kmeans_cpu.cc) | Absent | Kubism CPU implementation extended with sample-wise access, squared-distance helpers, NEON and optional batch4 processing. |
| [`src/kmeans_cpu.h`](../src/kmeans_cpu.h) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/kmeans_cpu.h) | Absent | Kubism declarations extended/changed for the distance, NEON and batch4 interfaces. |
| [`src/knn.cu`](../src/knn.cu) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/knn.cu) | [Identical](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/knn.cu) | KMCUDA KNN implementation preserved byte-for-byte through Kubism. |
| [`src/metric_abstraction.h`](../src/metric_abstraction.h) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/metric_abstraction.h) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/metric_abstraction.h) | Differences from KMCUDA are comments/formatting; the distance formulas are retained. |
| [`src/power_metrics.h`](../src/power_metrics.h) | Absent | Absent | Absent from both pinned trees; added in local development to declare helpers already implemented in Kubism. |
| [`src/private.h`](../src/private.h) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/private.h) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/private.h) | Shared infrastructure retained; Futurism changes managed allocation and adds raw-sample/batch4 interfaces. |
| [`src/read_input.c`](../src/read_input.c) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/read_input.c) | Absent | Kubism CSV reader/runner retained; adds only the input-size printf and blank lines. |
| [`src/setup.py`](../src/setup.py) | [Different](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/setup.py) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/setup.py) | Native-only build helper rewritten for this repository; old package metadata/bindings removed. |
| [`src/transpose.cu`](../src/transpose.cu) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/transpose.cu) | [Different](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/transpose.cu) | Kubism comments out two KMCUDA logging calls; inherited unchanged before the new notice. |
| [`src/tricks.cuh`](../src/tricks.cuh) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/tricks.cuh) | [Identical](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/tricks.cuh) | KMCUDA CUDA helpers preserved byte-for-byte, including their external reference links. |
| [`src/wrappers.h`](../src/wrappers.h) | [Identical](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/wrappers.h) | [Identical](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/src/wrappers.h) | KMCUDA wrapper declarations preserved byte-for-byte. |

Four files are identical to both upstreams: `kmcuda.h`, `knn.cu`, `tricks.cuh` and
`wrappers.h`. They remain byte-identical after this audit. Three more native files
were identical to Kubism before notices: `fp_abstraction.h`,
`metric_abstraction.h` and `transpose.cu`. Their differences from KMCUDA are blank
lines/comments or disabled logging, rather than replacement algorithms.

Consequently, a claim that the entire repository was rewritten would be inaccurate.
The substantial Futurism work is concentrated in the cooperative K-means path,
CPU distance/layout implementations, compaction and data/centroid handling.

## Credits and changes retained

The official Kubism headers in `kmeans_cpu.cc`, `kmeans_cpu.h` and `read_input.c`
already say `Implemented by: Seondeok Kim`. These strings are retained verbatim.
No personal copyright line, inferred author identity or new ownership claim was
added for the current maintainer.

Twelve source/build files receive factual provenance notices. For files otherwise
unchanged from Kubism, the notice says so and identifies the inherited formatting
or logging differences. Modified files describe the observable Futurism changes.
The four files identical to both upstreams have not received added headers.

`power_metrics.h` is first added in local commit
`2de4fcbd65d3bdf025367ddbca54699ff132a588`; it does not occur in either pinned
upstream tree. Its declarations refer to functions already in Kubism. Its first
appearance in local history does not establish who personally authored it.

## License evidence

KMCUDA includes `LICENSE.md` (Apache 2.0) and no NOTICE file in its pinned tree.
The repository's root LICENSE matches that file byte-for-byte.

Kubism's pinned tracked tree has no LICENSE, NOTICE or COPYING file. Its
`src/setup.py` declares Apache in the license field and classifier, while keeping
KMCUDA's package author and repository metadata.

Kubism and Futurism are from the same lab. The upstream Apache 2.0 license,
existing source credits and modification notices are retained. The new root
NOTICE records the source chain; no separate license scheme or personal copyright
statement is added for the lab's changes. See [PROVENANCE.md](PROVENANCE.md).

## Scope and verification

The audit does not attribute individual lines to individual people. The public
reference trees plus local Git ancestry establish provenance, but cannot prove
all external contributions or private licensing agreements. The reference links
already present in CUDA helpers are preserved. Generated CMake files, binaries,
profiling output, datasets and legacy scripts are outside the publication scope.

Every source edit in this audit is a prepended comment: the exact pre-audit file
body is retained. The original `futurism` working tree, its experiment scripts,
compiled artifacts and dataset files were not changed by this audit.
