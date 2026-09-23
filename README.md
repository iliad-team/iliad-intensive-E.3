# iliad-intensive-E.3

Source for the compact-proofs Colab notebook of **E.3 Worst-Case Interpretability**
(Iliad Intensive), by Louis Jaburi. The worksheet page lives in the website repo:
https://iliad-intensive.org/safety/worst-case-interp/

The notebook is authored as a Python master file, `gen/masters/master_E_3.py`, in
the ARENA cell format (`# ! CELL TYPE`, `# ! FILTERS`, `# ! TAGS` headers; markdown
cells in `r'''…'''`). Every push to `master` regenerates the exercises and solutions
notebooks and force-pushes them, with the images, to the
[`build`](https://github.com/iliad-team/iliad-intensive-E.3/tree/build) branch.
Nothing built is committed here; `.ipynb` files never are.

## Notebooks

- **Compact Proofs of Model Performance via Mechanistic Interpretability**:
  [exercises](https://colab.research.google.com/github/iliad-team/iliad-intensive-E.3/blob/build/compact_proofs/E.3_Compact_Proofs_exercises.ipynb)
  · [solutions](https://colab.research.google.com/github/iliad-team/iliad-intensive-E.3/blob/build/compact_proofs/E.3_Compact_Proofs_solutions.ipynb)
  (Colab). Source: `gen/masters/master_E_3.py` + `gen/support/compact_proofs/img/`.

## Local build

```bash
pip install -r gen/requirements-gen.txt      # once
python gen/core/main.py --chapters='E.3'     # -> build/exercises/compact_proofs/
```

## Provenance

Forked from [LouisYRYJ/Proof_based_approach_tutorial](https://github.com/LouisYRYJ/Proof_based_approach_tutorial).
The master file was exported from Louis's `proof_public.ipynb` by the generator's own
`ipynb → py` step; his cell text is unchanged. The generator itself is the one used by
[iliad-team/iliad-intensive-D.2](https://github.com/iliad-team/iliad-intensive-D.2)
(ported from ARENA).
