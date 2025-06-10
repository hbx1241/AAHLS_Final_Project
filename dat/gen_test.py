#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate CSR data and P-way partitions using greedy nnz-balanced row partitioning.

Outputs (global):
  - row_ptr.txt   : CSR row pointers
  - col_idx.txt   : CSR column indices
  - val.txt       : CSR values
  - x.txt         : Dense vector x

Outputs (per-partition):
  - row_ptr_{pid}.txt
  - col_idx_{pid}.txt
  - val_{pid}.txt
  - region_{pid}.txt  (row_start row_end)
"""
import numpy as np
from scipy.sparse import random as sprand

# -------------------------------------------------------------------
# Parameters
# -------------------------------------------------------------------
NROWS    = 256       # number of rows
NCOLS    = 256       # number of columns
DENSITY  = 0.30      # fraction of non-zeros
RNG_SEED = 2906      # random seed
P        = 2         # number of partitions
PAD      = 4         # pad nnz to multiple of PAD

rng        = np.random.default_rng(42)
# -------------------------------------------------------------------
# Build random CSR matrix
# -------------------------------------------------------------------
S        = sprand(NROWS, NCOLS, density=DENSITY, format='csr',
                   random_state=RNG_SEED,
                   #data_rvs=lambda n: np.random.uniform(1.0, 100.0, size=n)
                   data_rvs=rng.standard_normal
                  ).astype(np.float32)
row_ptr  = S.indptr.astype(np.int32)
cols     = S.indices.astype(np.int32)
vals     = S.data.astype(np.float32)
nnz      = int(vals.size)

# Dense x vector
x = np.linspace(1.0, float(NCOLS), NCOLS, dtype=np.float32)


# -------------------------------------------------------------------
# File writers with headers
# -------------------------------------------------------------------
def save_row_ptr(fname, M, N, nnz, rptr):
    with open(fname, 'w') as f:
        f.write(f"{M} {N} {nnz}\n")
        f.write(f"# row_ptr (len={len(rptr)})\n")
        for v in rptr:
            f.write(f"{v}\n")


def save_col_idx(fname, vec):
    with open(fname, 'w') as f:
        f.write(f"# col_idx (len={len(vec)})\n")
        for v in vec:
            f.write(f"{v}\n")


def save_val(fname, vec):
    with open(fname, 'w') as f:
        f.write(f"# val (float32) (len={len(vec)})\n")
        for v in vec:
            f.write(f"{v:.6f}\n")


def save_x(fname, vec):
    with open(fname, 'w') as f:
        f.write(f"{len(vec)}\n")
        f.write(f"# dense vector (len={len(vec)})\n")
        for v in vec:
            f.write(f"{v:.6f}\n")

# -------------------------------------------------------------------
# Save global CSR files
# -------------------------------------------------------------------
save_row_ptr("row_ptr.txt", NROWS, NCOLS, nnz, row_ptr)
save_col_idx("col_idx.txt", cols)
save_val("val.txt", vals)
save_x("x.txt", x)
print(f"Global CSR: rows={NROWS}, cols={NCOLS}, nnz={nnz}")

# -------------------------------------------------------------------
# Greedy partitioning
# -------------------------------------------------------------------
def greedy_split(rptr, pad, parts):
    nnz_row = np.diff(rptr)
    pad_nnz = ((nnz_row + pad - 1) // pad) * pad
    total   = pad_nnz.sum()
    ideal   = total / parts
    ends, acc = [], 0
    for i, p in enumerate(pad_nnz):
        if len(ends) < parts - 1 and acc + p > ideal:
            ends.append(i)
            acc = 0
        acc += p
    ends.append(len(nnz_row) - 1)
    return ends

ends = greedy_split(row_ptr, PAD, P)

# -------------------------------------------------------------------
# Save partition files
# -------------------------------------------------------------------
row_start, nz_start = 0, 0
for pid, row_end in enumerate(ends):
    # local CSR slice
    rptr_p = row_ptr[row_start:row_end+2] - row_ptr[row_start]
    nnz_p  = int(rptr_p[-1])
    cols_p = cols[nz_start:nz_start+nnz_p]
    vals_p = vals[nz_start:nz_start+nnz_p]

    # write partition files
    save_row_ptr(f"row_ptr_{pid}.txt", row_end-row_start+1, NCOLS, nnz_p, rptr_p)
    save_col_idx(f"col_idx_{pid}.txt", cols_p)
    save_val(f"val_{pid}.txt", vals_p)
    with open(f"region_{pid}.txt", 'w') as f:
        f.write(f"# region rows (start end)\n")
        f.write(f"{row_start} {row_end}\n")

    print(f"Partition {pid}: rows {row_start}-{row_end}, nnz={nnz_p}")
    row_start += (row_end - row_start + 1)
    nz_start += nnz_p

