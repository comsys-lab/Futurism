# Source provenance and license scope

Futurism builds on [Kubism](https://github.com/comsys-lab/Kubism), which in turn
contains code from [KMCUDA](https://github.com/src-d/kmcuda). This relationship is
supported by file comparisons and Git ancestry, rather than inferred from names
alone. Some sources are unchanged upstream code; others contain substantial
Futurism changes. The repository should not be described as an entirely original
or entirely rewritten implementation.

## Audited source versions

- Kubism: [`0575f83773d7a37afdee967e50a394da3fcec267`](https://github.com/comsys-lab/Kubism/tree/0575f83773d7a37afdee967e50a394da3fcec267).
- KMCUDA: [`7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b`](https://github.com/src-d/kmcuda/tree/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b).
- Futurism source snapshot: local base commit
  `8abb12cab22310fe4b96cbd4f8e04bc02a61b898`, plus the working-tree source changes
  and the native-build preparation described below.

The pinned Kubism commit is an ancestor of that Futurism base commit. This proves
that the local history contains Kubism; it does not establish individual ownership
of every later change. The original working tree and history remain separate from
this source-only repository.

All 14 native source/header files and both native build files were compared with
both upstream versions. See [the file-by-file audit](SOURCE_AUDIT.md) and its
[machine-readable hashes and comparisons](SOURCE_AUDIT.json).

## License and attribution

KMCUDA includes an [Apache 2.0 license](https://github.com/src-d/kmcuda/blob/7dcd28c49a6c76bb9c5e1b19c23976a1f76e1b3b/LICENSE.md).
The root [LICENSE.md](../LICENSE.md) is an unchanged copy of that file and is
retained for KMCUDA-derived code. The pinned KMCUDA tree contains no `NOTICE` file.

The pinned Kubism tree contains no `LICENSE`, `NOTICE` or `COPYING` file. Its
[`src/setup.py`](https://github.com/comsys-lab/Kubism/blob/0575f83773d7a37afdee967e50a394da3fcec267/src/setup.py)
declares `Apache Software License` and an Apache license classifier. The same
metadata also retains KMCUDA's author and repository URLs.

Kubism and Futurism are from the same lab. This preparation retains the inherited
Apache 2.0 license, existing source credits and modification notices. No separate
license scheme or personal copyright statement is introduced for the lab's changes.

The root [NOTICE](../NOTICE) records both sources and preserves the existing
Kubism implementation credit. It is a new provenance notice, not a recovered
upstream NOTICE file.

For the redistribution conditions of Apache 2.0, including retention of existing
notices and marking modified files, see [section 4](https://www.apache.org/licenses/LICENSE-2.0#redistribution).
The modified/inherited-modified files carry factual source and change notices;
the four files identical to both upstreams are preserved byte-for-byte.

## Implementation changes visible after Kubism

The audited Futurism sources add or change:

- Raw row-major samples for CPU processing, sample-wise access and ARM NEON
  distance routines, plus an optional batch4 path and matching declarations.
- CUB-based passed-point and CPU/GPU mark compaction.
- CPU/GPU work partitioning and fixed experimental threshold tables.
- Transposed GPU centroid access, background buffer initialization and changes
  to managed/pinned buffer handling and timing.
- Raw-sample forwarding in the host entry point and internal interfaces.
- An optional power-hook declaration header and a CSV input-size message.

The underlying CPU/GPU cooperative structure, original CPU functions and CSV
runner are already present in Kubism. KMCUDA supplies shared infrastructure,
initialization, Yinyang/Lloyd code, KNN, distance and transpose support. These are
observed code relationships, not a claim about who personally authored each change.

## Native-build preparation

This copy preserves native implementation behavior while applying the selected
Futurism experiment settings: raw/sample-wise/NEON and both CUB flags enabled,
batch4 disabled, and the existing `futurism_1M_128` thresholds. Native output names
remain `src/libfuturism.so` and `src/futurism`.

The native CMake configuration and `setup.py` build entry point were rewritten.
Unused Python/R bindings, their tests and packaging dependencies are excluded;
`setup.py` no longer advertises a Python module or an upstream package release.
The `0.0.0` value is an internal setuptools placeholder.

The new basic build/run scripts accept caller-supplied dataset paths. Personal
absolute paths were removed. Datasets, NCU reports, experiment logs, sweep/tuning
scripts, legacy shell scripts, Git metadata and old remotes are excluded from the
publication files. No new maintainer identity or lab repository URL is assumed.
