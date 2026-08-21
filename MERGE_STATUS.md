# Status: merging upstream/master into the SYCL branch

Written 2026-08-21 ~14:20 CDT, near the end of a 6 h PBS allocation
(`12473547`, node `x1921c1s0b0n0`). Resume from here.

## Where things stand

**Step 1 — DONE.** `fix/sycl-create-tasks-barrier` merged into `sycl` as a clean
fast-forward (12 commits). `sycl` == that branch, nothing to reconcile.

**Step 2 — IN PROGRESS.** `upstream/master` merged into branch
`sycl-merge-upstream`, in a git worktree at
`/home/abagusetty/gpu4pyscf-testing/sycl_merge` (shared Lustre, so it survives a
node change; `MERGE_HEAD` and the index are intact and the merge can simply be
continued).

Started at 43 conflicted files / 634 hunks. At time of writing 2 files still
contain conflict markers:

    gpu4pyscf/lib/pbc/contract_int3c2e.cu    7 hunks
    gpu4pyscf/lib/pbc/unrolled_rys_k.cu     29 hunks

Everything else is resolved on disk but **not yet staged or committed.**

## Resume

```bash
cd /home/abagusetty/gpu4pyscf-testing/sycl_merge
git status                      # MERGE_HEAD present, 36 unmerged paths
grep -rln '^<<<<<<<' gpu4pyscf/ # what is left
```

Read `MERGE_RECIPE.md` in that worktree first — it is the resolution rule the
whole merge followed.

## The rule used throughout

**upstream/master is ground truth.** This branch merges *into* upstream, so
upstream owns logic, kernel signatures and file layout; the SYCL port is a
transformation re-applied on top. Take theirs, then re-apply the SYCL overlay.
Never keep our side merely because it is ours.

## Before building — required verification

Conflict-marker-clean is NOT sufficient. Git silently dropped an upstream
addition **outside** any conflict marker: upstream added
`dim3 threads(nsq_per_block, gout_stride);` before the dispatch switch in six
`gvhf-rys/unrolled_*.cu` files and the 3-way merge kept our side, which never
had it. Left alone the CUDA build fails on an undeclared identifier — and we
cannot build CUDA here, so it would have escaped.

So: for every file touched by the merge, diff the resolved file against
`upstream/master` and confirm the only remaining deltas are the intended SYCL
overlay. Do this before spending 45 min on a build.

## Then

1. `git add -A && git commit` the merge.
2. Rebuild (~45 min; note the ECP AOT compile alone is ~47 min single-threaded).
3. Re-run the suite and compare against `test_baselines/pre_merge_baseline.txt`.

## Test protocol (matters — get this right or the comparison is meaningless)

intel/llvm#22943 is a *probabilistic* deadlock and concurrency widens its
window enormously. Measured: `test_df_int3c2e.py` times out at 3000 s when run
6-way but finishes in 6.68 s alone. **Even 2-way is unsafe** — it produced two
false failures in the baseline (`test_scf_jk::test_q_cond`, and
`test_ucdft::test_energy` where SCF convergence shifted to 1.4e-7 against a
1e-7 delta); both files are perfectly green run alone.

Protocol: sweep 2-way for throughput, then **requalify every non-green file
sequentially** before counting it. Apply identically pre- and post-merge.

## Open items flagged by the resolvers, not yet verified

Low confidence, worth a look or a smoke test:

1. `gvhf-rys/create_tasks.cu` — `_fill_ejk_tasks` / `_fill_sr_ejk_tasks` keep
   `JKEnergy jk` **by value** where upstream has `JKEnergy &jk`. Read-only in
   both, so semantically equal, but it is a deliberate divergence.
2. `gvhf-rys/rys_contract_k.cu` / `rys_contract_jk.cu` — upstream dropped the
   `template<int OFFSET>` kernel for a `__constant__ c_gxyz_offset[256]`
   re-uploaded before each of 3 launches. Mapped onto the existing
   `s_rys_gxyz_offset` device global. Correctness depends on the queue being
   in-order (it is documented as such). This is a **new write-during-flight
   pattern** — smoke-test it first.
3. `gvhf-rys/fill_int3c2e.cu` — upstream converted this to a persistent-worker
   kernel (1-D grid of `workers`, `while(1)` + `atomicAdd(head,1)`,
   `break`/`continue`). Barrier uniformity was checked by hand, not by test.

Already checked and fine: `RYS_init_rysj_constant()` now takes no argument and
`scf/jk.py:708` already calls it with none; `RYS_build_jk_init`/`RYS_build_k_init`
were deleted upstream and have zero callers; `fill_int3c2e`'s head counter is
already covered by `int3c2e_bdiv.py`'s `workers * POOL_SIZE + 1`, byte-identical
to upstream.

A genuine pre-existing bug was found and fixed during resolution: the SYCL
launch of `rys_vjk_ip1_kernel` used `blocks(1, npairs_ij)` where CUDA uses
`<<<workers, ...>>>`. With the pool sized `workers*QUEUE_DEPTH`, any
`npairs_ij > workers` ran off the pool and stomped the head counter.

## Deliberate resolutions worth remembering

- `gto/mole.py` — took upstream. Ours mixed a device operand with a host one
  (`recontract_bas` is numpy), which dpnp rejects; upstream builds on host and
  uploads once. Upstream is both ground truth and a latent-bug fix.
- `scf/jk.py` — took upstream's `TILE=12`, `QUEUE_DEPTH=262144`. Safe: the host
  only *allocates*, while the device strides by the C-side `QUEUE_DEPTH`
  (65536) and carves `head` at `pool + workers*QUEUE_DEPTH_C`. Upstream's value
  merely over-allocates (470 MB vs 117 MB). A comment now says so, so nobody
  "fixes" the C side to match.
- `lib/pbc/decompress.cu` and `lib/gint/nr_fill_ao_int3c2e_general.cu` —
  accepted upstream's deletions (zero remaining references; the surviving
  decompress is `gvhf-rys/decompress.cu`).
- `requirements.txt` — restored upstream's; our branch had deleted it.

## MUST FIX before the merged tree will run: `dpnp_helper.unpack_sparse`

Upstream removed the C symbol `unpack_sparse` from `lib/cupy_helper/unpack.cu`,
replacing it with `decompress_and_fill` / `decompress_and_transpose` taking
*pair addresses* instead of separate `row`/`col`. The merged `unpack.cu` has
been ported to the new API.

`gpu4pyscf/lib/dpnp_helper.py:426` still calls `libdpnp_helper.unpack_sparse`,
which no longer exists — that call will fail at runtime.

Upstream's fix, in `lib/cupy_helper.py`, is to keep a deprecated one-line
`unpack_sparse` wrapper and move the work into a new `fill_symmetric`:

```python
def unpack_sparse(cderi_sparse, row, col, p0, p1, nao, out=None, stream=None):
    warnings.warn('unpack_sparse is deprecated. Use fill_symmetric instead',
                  DeprecationWarning, stacklevel=2)
    return fill_symmetric(cderi_sparse, row*nao+col, nao, p0, p1, out, stream)

def fill_symmetric(a, pair_addresses, nao, p0=0, p1=None, out=None, stream=None):
    ...  # row-major  -> libcupy_helper.decompress_and_fill(...)
         # col-major  -> libcupy_helper.decompress_and_transpose(...)
```

Port that into `dpnp_helper.py` (dpnp/`libdpnp_helper` equivalents). Note it is
not a rename: the argument list and the row/column-major branch are both new.
Deliberately NOT attempted at the end of this allocation, because it could not
be built or tested here and a wrong port is worse than a known TODO.

## Confidence notes on the resolution itself

Two of the resolvers reported *silent* auto-merge artifacts — content wrong in
ways that leave no conflict marker:

- upstream's `dim3 threads(...)` line dropped in six `gvhf-rys/unrolled_*.cu`
  (would break only the CUDA build, which cannot be exercised here);
- a vestigial duplicate macro block in both `gvhf-md/unrolled_md_j*.cu`, and
  stale `Rt_buf`/`nf3ijkl` double-buffering left behind in
  `gvhf-md/contract_int3c2e.cu`.

One resolver also reported that a bulk regex it used initially ate three whole
kernel functions in `unrolled_md_j*.cu`; it caught this itself with a
kernel-count vs dispatcher-case-count cross-check and rebuilt the file from the
three merge stages. **Re-run that cross-check independently** (kernel
definitions vs dispatcher cases: 12/12 and 11/11 were the expected counts).

Nothing here has been compiled. Treat the whole merge as unverified until it
builds and the suite is compared against `test_baselines/pre_merge_baseline.txt`.

## Exact resume state (recorded at end of allocation 12473547)

```
worktree   /home/abagusetty/gpu4pyscf-testing/sycl_merge
branch     sycl-merge-upstream
HEAD       8c0ccae   (== sycl)
MERGE_HEAD 5025fc4   (upstream/master)
unmerged   36 paths staged as unmerged; all but 2 already resolved on disk
still conflicted on disk:
    gpu4pyscf/lib/pbc/contract_int3c2e.cu    7 hunks
    gpu4pyscf/lib/pbc/unrolled_rys_k.cu     29 hunks
backup     sycl_merge_backup_1418.tar.gz  (resolved sources, 2.5 MB, excludes .git)
```

Independently re-verified before the allocation ended: kernel definitions vs
dispatcher cases are 12/12 in `unrolled_md_j.cu` and 11/11 in
`unrolled_md_j_4dm.cu`, so the resolver's self-reported regex mishap was
recovered correctly.

Next session, in order:
1. finish the 2 pbc files;
2. global diff-vs-upstream to catch silently dropped upstream content;
3. port `dpnp_helper.unpack_sparse` to `decompress_and_fill` (see above);
4. `git add -A && git commit`;
5. rebuild (~45 min, ECP AOT alone ~47 min);
6. sweep 2-way, requalify non-green files sequentially, diff against
   `test_baselines/pre_merge_baseline.txt`.
