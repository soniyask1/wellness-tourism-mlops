"""Clean, group-stratify the holdout, and materialize CSV workflow artifacts."""
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from tourism_project.model_building.common import (DATA, SPLITS, REPORTS, FEATURES,
    POST_CONTACT, TARGET, clean_data, profile_groups, dump_json, sha256)

def main():
    raw = pd.read_csv(DATA); d = clean_data(raw)
    x, y = d[FEATURES], d[TARGET]; groups = profile_groups(x)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    train, test = next(splitter.split(x, y, groups))
    assert not set(groups.iloc[train]) & set(groups.iloc[test])
    SPLITS.mkdir(parents=True, exist_ok=True)
    for name, idx in [('train', train), ('test', test)]:
        x.iloc[idx].to_csv(SPLITS / f'X_{name}.csv', index=False)
        y.iloc[idx].to_csv(SPLITS / f'y_{name}.csv', index=False)
        groups.iloc[idx].rename('profile_group').to_csv(SPLITS / f'groups_{name}.csv', index=False)
    report = {'raw_rows': len(raw), 'clean_rows': len(d),
              'exact_repeats_removed_after_cleaning': len(raw)-len(d),
              'train_rows': len(train), 'test_rows': len(test),
              'train_positive_rate': float(y.iloc[train].mean()),
              'test_positive_rate': float(y.iloc[test].mean()),
              'unique_profile_groups': int(groups.nunique()), 'profile_overlap': 0,
              'excluded_interaction_columns': POST_CONTACT, 'features': FEATURES,
              'split_seed': 42, 'split_method': 'first fold of 5-fold StratifiedGroupKFold',
              'source_sha256': sha256(DATA),
              'split_sha256': {p.name: sha256(p) for p in sorted(SPLITS.glob('*.csv'))}}
    dump_json(REPORTS / 'preparation.json', report)
    print(f'Cleaned {len(raw):,} -> {len(d):,} records; removed {len(raw)-len(d)} repeats.')
    print(f'Train: {len(train):,} ({y.iloc[train].mean():.2%} buyers); test: {len(test):,} ({y.iloc[test].mean():.2%} buyers)')
    print(f'Pre-contact features: {len(FEATURES)}; train/test matching-profile overlap: 0')
    print('Saved six CSV files in artifacts/splits for upload-artifact/download-artifact.')

if __name__ == '__main__': main()
