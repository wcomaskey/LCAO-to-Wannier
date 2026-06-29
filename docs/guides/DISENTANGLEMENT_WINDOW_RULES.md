# Disentanglement Window Rules & Consistency Checks

This guide explains the energy-window rules Wannier90 enforces during
disentanglement, the **k-resolved frozen-window pitfall** that auto-generated
`.win` files can hit, and the consistency checks this package runs to catch it
before you launch `wannier90.x`.

## Background: the two windows

When `num_bands > num_wann` (disentanglement), Wannier90 uses two energy windows,
defined in the `.win` **relative to the Fermi energy**:

- **Outer / disentanglement window** `dis_win_min … dis_win_max`
  The pool of bands Wannier90 is allowed to mix to extract the optimal
  `num_wann`-dimensional subspace.
- **Frozen / inner window** `dis_froz_min … dis_froz_max`
  Bands inside are **forced** into the Wannier subspace unchanged (preserved
  exactly). Used to pin the bands near `E_F` you care about most.

## The rules (all are k-resolved)

Wannier90 checks these **at every k-point**, not on average:

1. **Frozen ⊆ outer.**
2. **Outer window must contain at least `num_wann` bands at every k** — otherwise
   there are too few states to disentangle into.
3. **The frozen window must contain at most `num_wann` bands at every k.** You
   cannot force more bands into the subspace than you have Wannier functions.
   Violating this aborts the run with:

   ```
   dis_windows: More states in the frozen window than target WFs
   ```

Rule 3 is the easy one to break automatically.

## The pitfall: steep bands and interlopers

A frozen window is an **energy interval**. A natural automatic choice is to size
it to span the "frontier" target bands and set `num_wann` to the number of those
bands. But two effects break rule 3:

- **Interloper bands.** Other selected bands (not in the frontier set) disperse
  *into* the frozen energy range at some k-points — most often steep/dispersive
  bands bunching up at the zone center (Γ). They are counted by Wannier90 even
  though they were not "frontier."
- **Per-band averaging.** Classifying a band as frontier vs. support by its
  *k-averaged* energy hides the fact that it crosses into the window at specific
  k-points.

So the count of bands inside the frozen window must be evaluated as the
**maximum over all k-points**. A window that looks fine "on average" can hold far
more than `num_wann` bands at a single k.

> **Worked example.** A 2D Sc slab (SHRINK 3 3 1 → 9 k-points), non-SOC,
> spin-polarized. Stage 1 chose `num_wann = 51` (alpha) and a frozen window
> `[-6.95, 5.10] eV`. At the zone center that window actually contained **76**
> bands (51 frontier + 25 interlopers). wannier90 aborted. The down-spin channel
> failed independently with 69 bands at a *different* worst k-point — underscoring
> that this is genuinely k-resolved and per-spin.

## The fix: k-resolved frozen-window clamp

Band selection now **uses the full BZ sample to validate the window**. After the
candidate frozen window is built from the frontier bands, it is shrunk until the
per-k rule holds (`_clamp_frozen_window_kresolved` in
`lcao_wannier/projectability.py`):

- For every k-point it counts the selected bands inside `[froz_min, froz_max]`.
- While any k exceeds `num_wann`, it trims whichever edge most reduces the
  worst-case count, keeping the Fermi-level region.
- The outer window is unchanged and remains valid (the clamped frozen window is a
  subset).

Stage 1 prints when it acts:

```
  [frozen-window] clamped to per-k BZ rule: [-6.950, 5.100] -> [-6.950, 1.602] (<= 51 bands/k)
```

This makes Stage 1 emit **valid `.win` files by construction**. The frozen window
may end up narrower than the full frontier span — that is correct: you cannot
freeze more bands than you have Wannier functions, so only the region that is
genuinely isolated to `num_wann` bands is frozen.

## Consistency checks

Even with the clamp, every Stage 1 and Stage 2 run validates the written `.win`
against the rules using the per-k eigenvalues and prints a report:

```
======================================================================
  Stage 1 consistency check: Sc_alpha
======================================================================
  Frozen window: up to 51 bands/k (worst k=1)
  Outer window:  at least 316 bands/k (worst k=1)
  STATUS: ✓ PASS — satisfies Wannier90 disentanglement rules
======================================================================
```

A failing setup reports the worst k-point and a suggested `dis_froz_max`. The
check is **non-fatal** — files are still written so you can inspect or override.

### Checking an existing seedname

```bash
python3 scripts/check_win.py path/to/seedname   # reads seedname.win + seedname.eig
```

Exit code 0 = PASS, 1 = FAIL. Implemented in `lcao_wannier/wannier_checks.py`
(`check_disentanglement_rules`, `check_seed_windows`), which you can also call
programmatically with your own per-k eigenvalues.

## If a setup still fails

- **Lower `dis_froz_max`** (or raise `dis_froz_min`) to the suggested value — the
  report tells you the bound.
- **Increase `num_wann`** if you genuinely want to capture more bands near `E_F`.
- **Widen the outer window** if rule 2 fails (too few bands to disentangle into).

## Related flags

- `--spin {alpha,beta,both}` — collinear spin-polarized runs. The window rules are
  checked **per channel**; the up and down channels generally have different
  `num_wann` and frozen windows.
- `--memory low`, `--no-prune`, `--prune-threshold` — memory controls; see the
  CHANGELOG. They do not affect the windows or the checks.
