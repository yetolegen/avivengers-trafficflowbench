import ast, hashlib, io, json, re, tokenize
from pathlib import Path
ROOT = Path(r'C:\Users\tuleg\avivengers')
dec = (ROOT / 'backup' / 'mansur_cell1_decoded.py').read_text(encoding='utf-8')
m = re.search(r"MODULE_SRC = r'''(.*?)'''\n", dec, re.S)
src = m.group(1)
assert len(src) == 75015, len(src)
assert hashlib.sha256(src.encode()).hexdigest() == 'f88501e566d15d4e160ab26939be39e73fdf3bd63695505437265402be4ba1d1'
def strip_comments(code):
    toks = list(tokenize.generate_tokens(io.StringIO(code).readline))
    lines = code.split('\n')
    for t in reversed(toks):
        if t.type == tokenize.COMMENT:
            r, c = t.start
            line = lines[r - 1]
            new = line[:c].rstrip()
            if new:
                lines[r - 1] = new
            else:
                lines[r - 1] = None
    return '\n'.join(l for l in lines if l is not None)
def doc_nodes(tree):
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            b = node.body[0]
            if isinstance(b, ast.Expr) and isinstance(b.value, ast.Constant) and isinstance(b.value.value, str):
                found.append((node, b))
    return found
def strip_docstrings(code):
    tree = ast.parse(code)
    lines = code.split('\n')
    edits = []
    for parent, b in doc_nodes(tree):
        only = len(parent.body) == 1
        edits.append((b.lineno, b.end_lineno, only, b.col_offset))
    for start, end, only, col in sorted(edits, reverse=True):
        lines[start - 1:end] = [' ' * col + 'pass'] if only else []
    return '\n'.join(lines)
def without_docs(tree):
    for parent, b in doc_nodes(tree):
        parent.body = parent.body[1:] or [ast.Pass()]
    return tree
clean = strip_docstrings(strip_comments(src))
assert ast.dump(without_docs(ast.parse(src))) == ast.dump(ast.parse(clean))
assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(clean).readline))
assert not doc_nodes(ast.parse(clean))
defs = (ROOT / 'notebooks' / '07_t2_struct_defs.py').read_text(encoding='utf-8')
main = (ROOT / 'notebooks' / '11_t2_final_main.py').read_text(encoding='utf-8')
for name, code in (('defs', defs), ('main', main)):
    ast.parse(code)
    assert not any(t.type == tokenize.COMMENT for t in tokenize.generate_tokens(io.StringIO(code).readline)), name
notes = """# task 2

## Update notes, 2026-11-03

This version replaces the previous task 2 final (v3). The earlier version stays in this notebook's version history.

**What changed**

- Onset training windows now follow the shipped train windows. In 39 of the 40 shipped onset windows the first queued cell appears at exactly T+30, with no queue in the 60-minute history and none at T+5 to T+25. Every shipped ongoing window has its first queue at T+5. The old onset windows had their first queue anywhere from T+5 to T+30, so the model was trained and tuned on a different kind of window from the ones it is scored on.
- History is read from the unmasked train layer and the reading at T from the masked layer, as in the released window_history.
- 41 added features per link: distance to the nearest queued link upstream and downstream, how fast that front moved over 15 and 30 minutes, number of queued links and queue clusters, the bottleneck links where queues historically start with their current state, time since the link last queued, and a recovery flag.
- Final prediction is the average of two LightGBM models (the original 109 features, and those plus the 41 new ones). Cutoffs are 0.30 for windows with no queue in the history and 0.25 for the rest.
- Trained on all 273 train days. No validation or private labels exist and none are used.

**Evidence** (proxy labels from the train layers, not the official scorer; 80 shipped train windows from days not used for training)

| model | overall | onset | ongoing |
|---|---|---|---|
| previous version | 0.650 | 0.436 | 0.863 |
| same features, new windows | 0.862 | 0.877 | 0.847 |
| with structure features | 0.882 | 0.905 | 0.860 |

**Output**

- queue.csv (174,000 rows), queue_validation.csv and queue_private.csv (87,000 rows each), columns window_id, timestamp, link_id, queue_pred.
- task2_models/ holds both models and their settings.
- Runs in about 30 minutes on 4 CPUs, no internet needed.
"""
notes = """# task 2

update nov 3

redid task 2. the old version (v3) is still in the version history if we need it back.

**what i changed**

- the onset windows we trained on were wrong. the train data ships 40 onset windows and in 39 of them the first jam shows up exactly at T+30 (nothing before that). ours had the first jam at random times, so the model learned the wrong thing. now the onset windows are built the same way as theirs
- history now comes from the unmasked data, like the real windows
- added 41 new features: how far the nearest jam is up and down the road, how fast it moves, where jams usually start, how long since the link was last jammed
- final answer is the average of 2 lightgbm models (old features, and old + new features). cutoffs are 0.30 for windows with no jam yet and 0.25 for the rest
- trained on all the train days, nothing from validation or private

**scores** (80 shipped train windows, our own labels, not the real scorer)

| | overall | onset | ongoing |
|---|---|---|---|
| old | 0.650 | 0.436 | 0.863 |
| new | 0.882 | 0.905 | 0.860 |

**output**

queue.csv, queue_validation.csv, queue_private.csv, same format as before

takes about 30 min, no internet needed
"""
def code_cell(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(True)}
cells = [
    {'cell_type': 'markdown', 'metadata': {}, 'source': notes.splitlines(True)},
    code_cell('%%writefile /kaggle/working/t2lib.py\n' + clean),
    code_cell(defs),
    code_cell(main),
]
nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
out = ROOT / 'notebooks' / 'task2_final_v4.ipynb'
out.write_text(json.dumps(nb), encoding='utf-8')
(ROOT / 'notebooks' / 't2lib_clean.py').write_text(clean, encoding='utf-8')
print('built', out, 'cells', len(cells), 'size', out.stat().st_size, 'clean lines', clean.count('\n'))
