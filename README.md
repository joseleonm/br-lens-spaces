# Bollobás–Riordan Polynomials of Lens Spaces

**Primary database and interactive explorers for the Bollobás–Riordan polynomial
of Heegaard ribbon graphs of lens spaces L(p, q).**

Live explorers: **https://joseluis.github.io/br-lens-spaces/**

---

## Overview

The lens space L(p, q) admits a Heegaard splitting whose attaching curve defines a
ribbon graph G(p; ±1, ±q) embedded on the torus T². This repository provides:

1. **A primary database** (`data/BR_polynomials.json`) containing the full
   Bollobás–Riordan polynomial of G(p; ±1, ±q), represented as a list of monomials
   X^s · y^t · z^{2g}, for all orbits of L(p, q) under orientation-preserving
   homeomorphism with 3 ≤ p ≤ 20 (45 cases).

2. **Interactive HTML explorers** (one per case, served via GitHub Pages) for p ≤ 17
   (35 cases), allowing the user to select spanning subgraphs, inspect the
   corresponding monomial and ribbon-graph invariants, and visualize the polynomial
   as a heat-map in real time.

3. **Computation source code** (SageMath + Cython) to reproduce or extend the database.

---

## Database format

`data/BR_polynomials.json` is a JSON object with a single key `"cases"`, whose value
is a list of records. Each record has the following fields:

| Field | Type | Description |
|---|---|---|
| `p` | int | Parameter p of the lens space L(p, q) |
| `q` | int | Parameter q (canonical representative of the orbit) |
| `terms` | list | Monomials of BR(G) as `[s, t, z_exp, coefficient]` |
| `S2` | int | First non-trivial coefficient of the S-profile (crest value) |
| `m2` | int | Multiplicity m₂ |
| `F` | int | Invariant F (second elementary symmetric function of determinants) |
| `G` | int | Invariant G (third elementary symmetric function of determinants) |
| `profile` | list | Full S-profile T₀ |

The tuple (S₂, m₂, F, G) distinguishes all 35 prime-p orbits up to p = 100, including
the 24 pairs with equal (S₂, m₂).

---

## Computational complexity

The computation enumerates all 2^{2p} spanning subgraphs of G, reduced by the
Z/pZ symmetry to necklace classes.

| p  | Homeo. classes | Canonical pairs (approx.) | Time (1 core) |
|---:|:--------------:|--------------------------:|--------------:|
| 5  | 1              | 220                       | < 0.1 s       |
| 7  | 2              | 2 340                     | ~0.5 s        |
| 11 | 4              | 382 K                     | ~5 s          |
| 13 | 5              | 5 M                       | ~60 s         |
| 17 | 7              | 1 G                       | hours         |
| 19 | 8              | 14 G                      | days          |

Cases p = 18, 19, 20 were computed on the **CIMAT–Mérida Supercomputer**
(Unidad CIMAT Mérida, Centro de Investigación en Matemáticas). This infrastructure
made it possible to extend the database beyond what is feasible on a personal
computer.

Benchmarks for the parallel implementation on an 8-core Apple M2 (via Cython + multiprocessing):

| p  | Cases | Time (8 workers + Cython) |
|---:|------:|--------------------------:|
| 13 | 4     | ~50 s                     |
| 17 | 5     | ~8 min                    |

---

## Interactive explorers

Open `index.html` locally, or visit the GitHub Pages URL above.
Each explorer page shows:

- **Left panel** — lattice grid for G(p; ±1, ±q); click edges to build a subgraph F.
- **Right panel** — 3-D torus rendering with selected edges highlighted.
- **BR ribbon** — ribbon-graph invariants k(F), f(F), g(F) and the monomial of F.
- **z-slice heat-map** — all monomials at the genus level of F, coloured by cascade position.

The Steiner-tree hint (dashed grey) shows a canonical spanning tree achieving minimum genus.

---

## Reproducing the computation

### Dependencies

- SageMath ≥ 10.4
- Python ≥ 3.10 with `numpy`
- Cython (optional but strongly recommended for p ≥ 13)

### Build the Cython extension

```bash
cd src/
sage -python setup_BR.py build_ext --inplace
```

This generates `BR_lens_core.cpython-*.so`. Without it the computation falls back to
pure Python (~100× slower).

### Run the parallel computation

```bash
cd src/
sage BR_lens_parallel.sage
```

Output is written to `../data/BR_polynomials.json`.

### Regenerate the HTML explorers

```bash
pip install numpy
python src/generate_explorers.py
```

This reads `data/BR_polynomials.json` and writes one `br_L{p}_{q}.html` per case.

---

## Acknowledgements

This work was supported by **SECIHTI research grant CBF2023-2024-4059**:
*"Interacciones topológico-computacionales"*.

The computation for p = 18, 19, 20 was carried out on the **Supercomputer of
Unidad CIMAT–Mérida** (Centro de Investigación en Matemáticas, Mérida, México).

Portions of the implementation were developed with AI assistance and have been
rigorously verified against the mathematical theory.

---

## Citation

If you use this database or the explorers in your research, please cite:

```
Jose L. Leon.
Bollobás–Riordan Polynomials of Lens Spaces: database and interactive explorers.
Zenodo, 2026. DOI: [to be assigned upon Zenodo release]
```

---

## License

Data (`data/`) and HTML explorers: [CC BY 4.0](LICENSE).
Source code (`src/`): MIT License.
