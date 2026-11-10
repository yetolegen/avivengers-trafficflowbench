import glob, json, pandas as pd
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40); pd.set_option('display.max_rows', 80)
B = glob.glob('/kaggle/input/**/task2_bundle', recursive=True)[0]
print(open(B + '/categorical_mapping.json').read())
cfg = json.load(open(B + '/config.json'))
print({k: v for k, v in cfg.items() if k not in ('features', 'ref_features', 'model_features')})
for n in ['eval_summary', 'tune_comparison', 'tune_thresholds_v2', 'window_counts', 'prediction_shares', 'selection_stats', 'eval_horizon']:
    d = pd.read_csv(f'{B}/metrics/{n}.csv'); print('=====', n, d.shape); print(d.head(40).to_string())
c = pd.read_csv(B + '/metrics/generated_windows_catalog.csv'); print('===== catalog', c.shape); print(c.head(5).to_string()); print(c.nunique())
r = pd.read_csv(B + '/metrics/released_window_audit.csv'); print('===== released_window_audit', r.shape); print(r.head(8).to_string())
