import json
from pathlib import Path
ROOT = Path(r'C:\Users\tuleg\avivengers')
nb = json.loads((ROOT / 'backup' / 'task-4_v1_original.ipynb').read_text(encoding='utf-8'))
def src(i):
    return ''.join(nb['cells'][i]['source'])
def put(i, text):
    nb['cells'][i]['source'] = text.splitlines(True)
    if nb['cells'][i]['cell_type'] == 'code':
        nb['cells'][i]['outputs'] = []
        nb['cells'][i]['execution_count'] = None
put(0, """# task 4

update nov 3

before: we used the organizers' baseline file, which is fit to the train months' link counts.
now: we use the file fit to the counts of the month being scored (validation and private each use their own counts, released with the data). the link-count score goes from about 0.965 to 0.9999 on validation.

same format as before: odme.csv, 70,708 rows. internet on.
""")
c2 = src(2)
assert 'baseline_submission.csv' in c2
c2 = c2.replace('baseline_submission.csv', 'public_reference.csv')
check = '''for split in ["validation", "private"]:
    for p in sorted(x.name for x in (REL / "task4").iterdir() if x.is_dir()):
        assert (REL / "task4" / p / split / "synthetic_link_counts.csv").exists(), (split, p)

'''
put(2, check + c2)
put(3, 'S_link, own-month counts (validation and private)')
put(4, '''for split in ["validation", "private"]:
    run("python", REPO / "src/task4/score_task4.py",
        "--submission", Path(f"/tmp/task4_{split}/public_reference.csv"),
        "--release-root", REL, "--split", split)''')
(ROOT / 'notebooks' / 'task4_v2.ipynb').write_text(json.dumps(nb), encoding='utf-8')
print('built', [len(src(i)) for i in range(len(nb['cells']))])
