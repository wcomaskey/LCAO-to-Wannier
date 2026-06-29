import sys
import numpy as np
from lcao_wannier.basis_parser import parse_basis_shells, get_atom_list
from lcao_wannier.valence_config import build_target_mask, get_valence_l

path = sys.argv[1]
with open(path) as f:
    lines = f.readlines()
shells, nao = parse_basis_shells(lines, num_atoms=2)
atoms = get_atom_list(shells)
print("atoms:", atoms, " spatial AOs:", nao, " n_shells:", len(shells))
print("valence_l(Bi) standard:", get_valence_l('Bi', extended=False))

mask = build_target_mask(shells, extended=False, include_tm_p=False,
                         has_soc=True, verbose=False)
print("target mask total (SOC):", int(np.sum(mask)), "/", len(mask))

# Show shells with (atom, l, radial-index-within-l) and whether targeted.
# mask is SOC-doubled; use the spatial half.
half = len(mask) // 2
mask_sp = mask[:half]
L = {0: 's', 1: 'p', 2: 'd', 3: 'f'}
print("\n idx  atom  l  ao_range            targeted?")
ao = 0
per_atom_l_count = {}
for sh in shells:
    a = getattr(sh, 'atom_index', getattr(sh, 'atom', '?'))
    l = getattr(sh, 'l', getattr(sh, 'angular_momentum', '?'))
    norb = 2 * l + 1 if isinstance(l, int) else getattr(sh, 'num_orbitals', 1)
    key = (a, l)
    per_atom_l_count[key] = per_atom_l_count.get(key, 0) + 1
    nth = per_atom_l_count[key]
    tgt = bool(np.any(mask_sp[ao:ao + norb]))
    print(f"  atom{a}  l={l}({L.get(l,'?')})  radial#{nth}  ao[{ao}:{ao+norb}]   target={tgt}")
    ao += norb
