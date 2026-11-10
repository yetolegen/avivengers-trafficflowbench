import json, glob, os
cands = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)
print('BUNDLES', cands)
B = cands[0]
for root, d, fs in os.walk(B):
    for f in fs:
        p = os.path.join(root, f); print(p, os.path.getsize(p))
for n in ['config.json', 'split_definitions.json', 'manifest.json']:
    p = os.path.join(B, n)
    if os.path.exists(p):
        print('=====', n); print(open(p).read()[:6000])
src = open(os.path.join(B, 't2lib.py')).read()
i = src.index('def clim_rows')
print('===== T2LIB TAIL'); print(src[i:])
