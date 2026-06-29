import re, numpy as np, sys
seed = sys.argv[1]
p = {}
for line in open(seed + '.win'):
    m = re.match(r'\s*(\w+)\s*=\s*([-\d.eE+]+)', line)
    if m and m.group(1).startswith(('dis_', 'num_')):
        p[m.group(1)] = m.group(2)
print("parsed win:", p)
per_k = {}
for line in open(seed + '.eig'):
    a = line.split()
    if len(a) >= 3:
        per_k.setdefault(int(a[1]), []).append(float(a[2]))
ks = sorted(per_k)
e1 = np.array(per_k[ks[0]])
print(f"k=1: {len(e1)} bands, energy range [{e1.min():.3f}, {e1.max():.3f}]")
print(f"first 5 energies k=1: {e1[:5]}")
