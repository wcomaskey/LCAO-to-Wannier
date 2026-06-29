#!/usr/bin/env python3
"""Unit checks for the spread-minimization window assist."""
import numpy as np
from lcao_wannier.window_assist import spread_minimizing_windows, manifold_windows


def make(nk=4, jitter=0.05):
    # 5 projectable bands (-3..+1) + 3 diffuse bands (+2..+4)
    centers = np.array([-3., -2., -1., 0., 1., 2., 3., 4.])
    proj = np.array([0.95, 0.95, 0.95, 0.95, 0.95, 0.30, 0.25, 0.20])
    rng = np.linspace(-jitter, jitter, nk)
    eig = np.array([centers + d for d in rng])  # (nk, nb)
    return eig, proj


def _slices(eig, E):
    e_lo, e_hi = eig.min(0), eig.max(0)
    return bool(np.any((e_lo < E) & (e_hi > E)))


def test_basic():
    eig, proj = make()
    # edges in the gaps (-1.5 between -2/-1, +0.5 between 0/1) -> no snapping
    r = spread_minimizing_windows(eig, proj, num_wann=5, min_froz=(-1.5, 0.5),
                                  proj_floor=0.9, win_padding=0.5)
    assert abs(r.dis_froz[0] - (-1.5)) < 1e-6 and abs(r.dis_froz[1] - 0.5) < 1e-6, r.dis_froz
    assert r.dis_win[1] < 1.9, r.dis_win          # outer excludes diffuse (>=2)
    assert r.num_wann == 5, r.num_wann            # target holds 2 bands <= 5
    print("  basic: froz", tuple(round(x, 2) for x in r.dis_froz),
          "win", tuple(round(x, 2) for x in r.dis_win), "nw", r.num_wann, "OK")


def test_min_froz_forces_growth():
    eig, proj = make()
    # [-1.5, 2.5] (clean edges) holds bands -1,0,1,2 => 4 > num_wann 3 -> grow
    r = spread_minimizing_windows(eig, proj, num_wann=3, min_froz=(-1.5, 2.5),
                                  proj_floor=0.9, win_padding=0.5)
    assert r.num_wann >= 4, r.num_wann
    assert any("raising num_wann" in n for n in r.notes), r.notes
    print("  min-froz forces growth: nw 3 ->", r.num_wann, "OK")


def test_snap_clean_cut():
    eig, proj = make()
    # [-1.0, 0.0]: BOTH edges land mid-band (bands centered at -1 and 0) -> must snap
    r = spread_minimizing_windows(eig, proj, num_wann=5, min_froz=(-1.0, 0.0))
    assert not _slices(eig, r.dis_froz[0]), f"lower edge {r.dis_froz[0]} slices a band"
    assert not _slices(eig, r.dis_froz[1]), f"upper edge {r.dis_froz[1]} slices a band"
    assert any("snapped frozen edges" in n for n in r.notes), r.notes
    print(f"  snap: froz {tuple(round(x,3) for x in r.dis_froz)} are clean cuts OK")


def test_frozen_count_le_numwann():
    eig, proj = make()
    r = spread_minimizing_windows(eig, proj, num_wann=5, min_froz=(-1.5, 0.5))
    cnt = int(((eig >= r.dis_froz[0]) & (eig <= r.dis_froz[1])).sum(axis=1).max())
    assert cnt <= r.num_wann, (cnt, r.num_wann)
    omin = int(((eig >= r.dis_win[0]) & (eig <= r.dis_win[1])).sum(axis=1).min())
    assert omin >= r.num_wann, (omin, r.num_wann)
    print(f"  frozen {cnt}/k <= num_wann {r.num_wann}; outer >= num_wann  OK")


def test_manifold_interest_count():
    eig, _ = make()
    # user window [-1.5, 0.5] -> bands centered -1 and 0 pass through it
    r = manifold_windows(eig, (-1.5, 0.5))
    assert r['num_wann'] == 2, r['num_wann']
    assert abs(r['dis_froz'][0] + 1.5) < 1e-6 and abs(r['dis_froz'][1] - 0.5) < 1e-6
    print("  manifold interest count: nw", r['num_wann'], "froz",
          tuple(round(x, 2) for x in r['dis_froz']), "OK")


def test_manifold_steep_band():
    # band 0 sits at 0 eV at k0 and dips to -5 eV at k3 (steep); band 1 stays ~3 eV.
    eig = np.array([[0.0, 3.0], [-2.0, 3.1], [-4.0, 3.0], [-5.0, 2.9]])
    r = manifold_windows(eig, (-1.0, 1.0))
    assert r['num_wann'] == 1, r['num_wann']            # only band 0 enters [-1,1]
    assert r['dis_win'][0] <= -5.0 + 1e-9, r['dis_win']  # outer follows it to -5
    assert r['dis_win'][1] < 2.0, r['dis_win']           # high band excluded
    print("  manifold steep band: dis_win",
          tuple(round(x, 2) for x in r['dis_win']), "OK")


def test_manifold_no_cascade():
    # dense overlapping ladder: only the band touching the window counts (no chain)
    centers = np.array([-3., -2., -1., 0., 1., 2., 3., 4.])
    eig = np.array([centers, centers + 0.1])
    r = manifold_windows(eig, (-0.4, 0.4))
    assert r['num_wann'] == 1, r['num_wann']
    print("  manifold no-cascade: nw", r['num_wann'], "OK")


def test_manifold_kresolved_no_false_connect():
    # band 0 enters the window (0 eV) and dips to -5; band 1 shares the energy
    # range but stays ~2 eV away at EVERY k. Energy-overlap would wrongly connect
    # them (the deep-band false positive); the k-resolved test must not.
    eig = np.array([[0., -2.], [-5., -7.], [0., -2.], [-5., -7.]])
    r = manifold_windows(eig, (-1.0, 1.0))
    assert r['num_wann'] == 1, r['num_wann']
    print("  manifold k-resolved no-false-connect: nw", r['num_wann'], "OK")


def test_manifold_kresolved_crossing_connects():
    # two bands that actually cross (meet at k1) DO connect -> both frozen.
    eig = np.array([[0., -4.], [-2., -2.], [-4., 0.], [-2., -2.]])
    r = manifold_windows(eig, (-1.0, 1.0))
    assert r['num_wann'] == 2, r['num_wann']
    print("  manifold k-resolved crossing-connects: nw", r['num_wann'], "OK")


if __name__ == '__main__':
    print("Window-assist unit checks")
    print("=" * 50)
    test_basic()
    test_min_froz_forces_growth()
    test_snap_clean_cut()
    test_frozen_count_le_numwann()
    test_manifold_interest_count()
    test_manifold_steep_band()
    test_manifold_no_cascade()
    test_manifold_kresolved_no_false_connect()
    test_manifold_kresolved_crossing_connects()
    print("=" * 50)
    print("ALL PASSED")
