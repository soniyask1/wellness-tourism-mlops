"""Register the repository CSV after schema/domain checks; exit nonzero on errors."""
import argparse
import pandas as pd
from tourism_project.model_building.common import DATA, EXPECTED, REPORTS, dump_json, sha256

def validate(d):
    missing = sorted(set(EXPECTED) - set(d.columns))
    extra = sorted(c for c in set(d.columns) - set(EXPECTED) if not c.startswith('Unnamed:'))
    if missing or extra:
        raise ValueError(f'Schema mismatch; missing={missing}, unexpected={extra}')
    if d.empty or d['CustomerID'].isna().any() or d['CustomerID'].duplicated().any():
        raise ValueError('Dataset empty or invalid/duplicate CustomerID')
    if d['ProdTaken'].isna().any() or set(d['ProdTaken'].unique()) != {0, 1}:
        raise ValueError('ProdTaken must contain both 0 and 1, without nulls')
    domains = {'CityTier': {1,2,3}, 'Passport': {0,1}, 'OwnCar': {0,1},
               'PreferredPropertyStar': {3,4,5}, 'PitchSatisfactionScore': {1,2,3,4,5}}
    for c, allowed in domains.items():
        if not set(d[c].dropna()).issubset(allowed):
            raise ValueError(f'Invalid domain for {c}')
    for c in ['Age', 'DurationOfPitch', 'NumberOfPersonVisiting', 'NumberOfFollowups',
              'NumberOfTrips', 'NumberOfChildrenVisiting', 'MonthlyIncome']:
        values = pd.to_numeric(d[c], errors='raise').dropna()
        if (values < 0).any():
            raise ValueError(f'Negative value in {c}')
    return True

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--data', default=str(DATA))
    args = parser.parse_args(); d = pd.read_csv(args.data); validate(d)
    report = {'path': str(DATA.relative_to(DATA.parents[2])), 'sha256': sha256(args.data),
              'rows': len(d), 'columns': len(d.columns),
              'missing_by_column': d.isna().sum().to_dict(),
              'target_counts': d.ProdTaken.value_counts().sort_index().to_dict(),
              'duplicate_customer_ids': int(d.CustomerID.duplicated().sum()),
              'numeric_summary': d.select_dtypes('number').describe().round(3).to_dict()}
    dump_json(REPORTS / 'registration.json', report)
    print(f'VALIDATED: {len(d):,} rows, {len(d.columns)} columns; missing cells={d.isna().sum().sum()}')
    print('Target counts:', report['target_counts']); print('Dataset SHA256:', report['sha256'])

if __name__ == '__main__': main()
