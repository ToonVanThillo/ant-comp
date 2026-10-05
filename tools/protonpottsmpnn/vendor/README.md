# vendor/ — supply HBPLUS here yourself

`hbplus.tar.gz` is **deliberately git-ignored and must never be committed.** This repo is
public, and HBPLUS is supplied under a confidentiality agreement (`confid.txt` in the
tarball) whose clauses 1 and 4 require it to be kept in confidence and "in a reasonably
secure place to prevent unauthorised access". Publishing it here would breach that.

## What to put here

| File | Source | sha256 (as verified 2026-10-02) |
| --- | --- | --- |
| `hbplus.tar.gz` | <https://www.ebi.ac.uk/thornton-srv/software/HBPLUS/> (academic licence), or the `HBPLUS` GitHub mirror by Ian McDonald | `937467447bd2e429630cc9a226d7d967f728ca2d39bac3ffc70eb378f498fa20` |

That is all. `modal_image.py` untars it, runs `make`, and exports `HBPLUS_PATH`. With the
file absent the image build fails immediately with an explanatory error, by design.

## Why the tool needs it at all

Proton-PottsMPNN's v6 protonation labeller reads H-bond geometry from HBPLUS, and
`prepare_potts_input` runs it on **every design call** — not only in the `labeller/`
scripts, despite what the upstream README's table says. Measured: stubbing the binary
shows it invoked during design featurisation as `hbplus -h 3.2 -d 4.0 <tmp>.pdb <tmp>.pdb`.

## Citation obligation

Clause 2 of the agreement: any publication using it must cite

> McDonald IK & Thornton JM (1994), "Satisfying Hydrogen Bonding Potential in Proteins",
> *Journal of Molecular Biology* 238:777-793.

Copyright 1991-3 Ian McDonald, Dorica Naylor, David Jones, S Hubbard, R A Laskowski and
Janet M Thornton. All Rights Reserved.
