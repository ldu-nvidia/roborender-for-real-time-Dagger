#!/usr/bin/env python3
"""Find references to fast_roborender scripts that no longer exist at the referenced path.

Catches exactly the class of breakage the reorganization introduced: a launcher naming a
sibling script by its pre-move path.
"""
import os, re, subprocess, sys

ROOT = os.path.realpath(
    os.environ.get("ROOT", os.path.join(os.path.dirname(__file__), "..", ".."))
)
os.chdir(ROOT)
SKIP = ('GR00T-VisualSim2Real', 'IsaacLab', 'roborender-acceleration', 'Sim2RealGen_Video_Model',
        'logs_archive', 'videogen_dataset', 'teacher_fullres', 'image3f_reproduction',
        'step_ab_out', '.git', 'transfer', 'archive')

# every script we own, by basename -> actual path(s)
owned = {}
for dp, dns, fns in os.walk('.'):
    dns[:] = [d for d in dns if d not in SKIP and not d.startswith('image3f_queue')]
    for f in fns:
        if f.endswith(('.py', '.sh', '.sbatch')):
            owned.setdefault(f, []).append(os.path.normpath(os.path.join(dp, f)))

APPLY = '--apply' in sys.argv
problems = []
for dp, dns, fns in os.walk('.'):
    dns[:] = [d for d in dns if d not in SKIP and not d.startswith('image3f_queue')]
    for f in fns:
        if not f.endswith(('.py', '.sh', '.sbatch')):
            continue
        path = os.path.normpath(os.path.join(dp, f))
        text = open(path, errors='ignore').read()
        for m in re.finditer(
            r'(/[^\s"\']+/fast_roborender/)([\w./-]+\.(?:py|sh|sbatch))', text
        ):
            ref = m.group(2)
            if os.path.exists(ref):
                continue
            base = os.path.basename(ref)
            actual = owned.get(base, [])
            problems.append((path, m.group(0), actual))
        # bare-name invocations of our scripts (python foo.py / sbatch foo.sbatch)
        for m in re.finditer(r'(?:python[0-9.]*\s+|sbatch\s+(?:--[\w=./:-]+\s+)*)([\w-]+\.(?:py|sbatch))', text):
            ref = m.group(1)
            if os.path.exists(ref) or ref not in owned:
                continue
            problems.append((path, m.group(0), owned[ref]))

if not problems:
    print('no stale references')
    sys.exit(0)

print(f'{len(problems)} stale reference(s):\n')
fixes = {}
for path, ref, actual in problems:
    tgt = f'{ROOT}/{actual[0]}' if actual else '??? (file not found anywhere)'
    print(f'  {path}\n      {ref}\n   -> {tgt}\n')
    if actual and ref.startswith('/'):
        fixes.setdefault(path, []).append((ref, f'{ROOT}/{actual[0]}'))
    elif actual:
        fixes.setdefault(path, []).append(
            (ref, ref.replace(os.path.basename(ref), f'{ROOT}/{actual[0]}')))

if APPLY:
    for path, subs in fixes.items():
        t = open(path).read()
        for old, new in subs:
            t = t.replace(old, new)
        open(path, 'w').write(t)
        print(f'fixed {path}')
else:
    print('(dry run — rerun with --apply)')
sys.exit(1)
