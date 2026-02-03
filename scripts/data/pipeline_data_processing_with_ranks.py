"""
Pipeline 1: Data Processing
============================
This module handles all data loading, preprocessing, feature engineering, and dataset preparation
for the election modeling project.
"""

import pandas as pd
import os
import numpy as np
import re
import tqdm
import json
from shapely.geometry import shape
from datetime import datetime
from config.data_processing_config import CONFIG
config = CONFIG

def load_electoral_data(processor):
    """Load and process electoral data from parquet files"""
    print("Loading electoral data...")
    print(f"Vote statistics (targets) will be {' '.join(processor.config['vote_variables'])}")

    folder_path = os.path.join(processor.config['data_path'], "elections")
    processor.dfs = pd.DataFrame(columns=['codecommune'])
    processor.election_included = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.parquet') and file not in processor.config['elections_to_exclude']:
                relative_path = os.path.relpath(root, folder_path)
                election_type = relative_path.split(os.sep)[0]
                year = relative_path.split(os.sep)[1] if len(relative_path.split(os.sep)) > 1 else None
                processor.election_included.append((election_type, year))
                df = pd.read_parquet(os.path.join(root, file))
                X = df[['codecommune', 'inscrits']+processor.config['vote_variables']]
                for var in processor.config['vote_variables']:
                    X = X.rename(columns={
                        var: f'pvote{var}{year}-{election_type}'})
                X = X.rename(columns={
                    'inscrits': f'inscrits{year}-{election_type}'
                })
                processor.dfs = pd.merge(processor.dfs, X, on='codecommune', how='outer')

    codes_to_remove = ['26383', '69380', '.', '07350', '51700', '51701', '51702', '51703', '51704', '51705', '51706', '51707', '51708', '51709', '51710', '51711', '51712', '51713']
    processor.dfs = processor.dfs[~processor.dfs['codecommune'].isin(codes_to_remove)]
    print(f"Electoral data loaded: {processor.dfs.shape}")

def load_socioeconomic_data(processor, cache_file="../data/cache/dfc_cached.csv"):
    """Load and process socio-economic data from parquet files"""
    print("Loading socio-economic data...")
    if processor.config['use_cached'] and os.path.exists(cache_file):
        print(f"Loading cached data from {cache_file}")
        processor.dfc = pd.read_parquet(cache_file)
    else: 
        processor.dfc = pd.DataFrame(columns=['codecommune'])
        for root, dirs, files in os.walk(processor.config['data_path']):
            for file in files:
                if file.endswith('.parquet') and 'communes' in file:
                    if 'code' in file:
                        continue
                    print(f"Processing: {file[:-len('.parquet')]}")
                    df = pd.read_parquet(os.path.join(root, file))
                    if bool(re.search(r'(\d{4})$', file[:-len('.parquet')])):
                        df.columns = df.columns.map(
                            lambda col: (
                                re.sub(r'(\d{4})(?!$)', '', col) + re.search(r'(\d{4})(?!$)', col).group(1)
                                if re.search(r'(\d{4})(?!$)', col) and not re.search(r'(\d{4})$', col)
                                else col
                            )
                        )
                    if file == "menagescommunes.parquet":
                        exclude_columns = ['dep', 'nomdep', 'codecommune', 'nomcommune']
                        columns_to_rename = [col for col in df.columns if col not in exclude_columns]
                        df = df.rename(columns={col: f"{col}_1975" for col in columns_to_rename})
                    if file == "terrescommunes.parquet":
                        exclude_columns = ['dep', 'nomdep', 'codecommune', 'nomcommune', 'plm', 'nomcanton']
                        columns_to_rename = [col for col in df.columns if col not in exclude_columns]
                        df = df.rename(columns={col: f"{col}_1968" for col in columns_to_rename})
                    features_with_years = [col for col in df.columns if re.search(r'(\d{4})$', col)]
                    selected_columns = ['codecommune'] + [f'{feature}' for feature in features_with_years]
                    Y = df[selected_columns]
                    Y.columns = [f"{file[:-len('.parquet')]}/{col}" if col != 'codecommune' else col for col in Y.columns]
                    processor.dfc = pd.merge(processor.dfc, Y, on='codecommune', how='outer')
        communes_socio_eco_data = set(processor.dfc['codecommune'].dropna().unique())
        communes_election_data = set(processor.dfs['codecommune'].dropna().unique())
        to_remove = communes_socio_eco_data - communes_election_data
        processor.dfc = processor.dfc[~processor.dfc['codecommune'].isin(to_remove)]
        if not os.path.exists(cache_file):
            processor.dfc.to_parquet(cache_file, index=False)
            print(f"Cached data saved to {cache_file}")
    print(f"Socio-economic data loaded: {processor.dfc.shape}")
    fl = list(set([col[:-4] for col in processor.dfc.columns if re.search(r'\d{4}$', col)]))
    print(f'Features available ({len(fl)})')

def add_gini(df, columns=None):
    if columns is None:
        columns = df.columns
    df2 = df.set_index('tconst')
    total = df2.pop('total_ethnicities')
    result = 1 - ((df2** 2 ).div(total**2, axis=0)).sum(axis=1)
    result.name = 'gini'

def _find_feat_and_year(feature):
    year = re.search(r'(\d{4})$', feature)
    if year is not None:
        year = year.group(1)
    else: 
        return None, None
    feat = re.sub(r'(\d{4})$', '', feature)
    return feat, year

def apply_feature_engineering(processor):
    """Apply feature engineering to create rank and delta features"""
    print("Applying feature engineering...")
    features_all = list(set([col for col in processor.dfc.columns if re.search(r'\d{4}$', col)]))
    new_columns = {}
    for feature in tqdm.tqdm(features_all, desc="Creating augmentation features"):
        previous_feature = _find_previous_feature(processor, feature, features_all)
        feat, year = _find_feat_and_year(feature)
        AUG = {}
        for aug in ['rank', 'delta', 'lag', 'pct_change',  'winsor', 'deltawinsor', 'lagwinsor', 'pct_changewinsor']:
            if aug in processor.config["features_aug"]:
                AUG[f"create_{aug}"] = True
            else:
                AUG[f"create_{aug}"] = False
        if AUG['create_rank']:
            if feature != 'codecommune':
                rank_feature = f"{feat}_rank{year}"
                new_columns[rank_feature] = processor.dfc[feature].rank(pct=True)
        if AUG['create_winsor']:
            if feature != 'codecommune':
                winsor_feature = f"{feat}_winsor{year}"
                if not isinstance(processor.dfc[feature].iloc[0], str):
                    new_columns[winsor_feature] = processor.dfc[feature].clip(lower=processor.dfc[feature].quantile(0.01), upper=processor.dfc[feature].quantile(0.99))
                else:
                    new_columns[winsor_feature] = np.nan
        if AUG['create_delta']:
            if previous_feature is not None:
                delta_feature = f"{feat}_delta{year}"
                new_columns[delta_feature] = processor.dfc[feature] - processor.dfc[previous_feature]
            else:
                new_columns[delta_feature] = np.nan
        if AUG['create_lag']:
            lagged_feature = f"{feat}_lag{year}"
            if previous_feature is not None:
                new_columns[lagged_feature] = processor.dfc[previous_feature]
            else:
                new_columns[lagged_feature] = np.nan
        if AUG['create_pct_change']:
            pct_change_feature = f"{feat}_pctchange{year}"
            if previous_feature is not None:
                new_columns[pct_change_feature] = np.where(
                    processor.dfc[previous_feature] != 0,
                    (processor.dfc[feature] - processor.dfc[previous_feature]) / processor.dfc[previous_feature],
                    np.nan 
                )
            else:
                new_columns[pct_change_feature] = np.nan
    if AUG["create_deltawinsor"] or  AUG["create_lagwinsor"] or AUG["create_pct_changewinsor"]:
        for feature in tqdm.tqdm(features_all, desc="Creating augmentation features"):
            previous_feature = _find_previous_feature(processor, feature, features_all)
            feat, year = _find_feat_and_year(feature)
            if AUG['create_deltawinsor']:
                if previous_feature is not None:
                    delta_feature = f"{feat}_deltawinsor{year}"
                    featprev, yearprev = _find_feat_and_year(previous_feature)
                    winsor_feature = f"{feat}_winsor{year}"
                    winsor_previous_feature = f"{featprev}_winsor{yearprev}"
                    try:
                        new_columns[delta_feature] = new_columns[winsor_feature] - new_columns[winsor_previous_feature]
                    except KeyError:
                        breakpoint()
                else:
                    new_columns[delta_feature] = np.nan
            if AUG['create_lagwinsor']:
                lagged_feature = f"{feat}_lagwinsor{year}"
                if previous_feature is not None:
                    featprev, yearprev = _find_feat_and_year(previous_feature)
                    winsor_previous_feature = f"{featprev}_winsor{yearprev}"
                    new_columns[lagged_feature] = new_columns[winsor_previous_feature]
                else:
                    new_columns[lagged_feature] = np.nan
            if AUG['create_pct_changewinsor']:
                pct_change_feature = f"{feat}_pctchangewinsor{year}"
                if previous_feature is not None:
                    winsor_feature = f"{feat}_winsor{year}"
                    featprev, yearprev = _find_feat_and_year(previous_feature)
                    winsor_previous_feature = f"{featprev}_winsor{yearprev}"
                    new_columns[pct_change_feature] = np.where(
                        new_columns[winsor_previous_feature] != 0,
                        (new_columns[winsor_feature] - new_columns[winsor_previous_feature]) / new_columns[winsor_previous_feature],
                        np.nan 
                    )
                else:
                    new_columns[pct_change_feature] = np.nan
    if 'raw' in processor.config["features_aug"]:
        processor.dfc = pd.concat([processor.dfc, pd.DataFrame(new_columns)], axis=1)
    else:
        processor.dfc = pd.concat([processor.dfc['codecommune'], pd.DataFrame(new_columns)], axis=1)
    print(f"Feature engineering completed: {processor.dfc.shape}")

def apply_quality_filter(processor):
    """Apply a quality filter to remove low-quality features."""
    missing_values = processor.global_dataset.isna().sum()
    high_missing = missing_values[missing_values > processor.global_dataset.shape[0] / 2].index
    processor.global_dataset.drop(columns=high_missing, inplace=True)
    print(f"Quality filter applied. The following features are removed (missing for more than half of the rows): {high_missing.tolist()}")
    print(processor.global_dataset.shape)

def add_geographical_data(processor):
    """Add geographical coordinates from GeoJSON file"""
    print("Adding geographical data...")
    if not os.path.exists(processor.config['geojson_path']):
        print(f"Warning: GeoJSON file not found at {processor.config['geojson_path']}. Skipping geographical data.")
        return
    with open(processor.config['geojson_path']) as f:
        communes_geojson = json.load(f)
    geo_data = []
    for feature in tqdm.tqdm(communes_geojson["features"], desc="Processing geographical data"):
        commune_geom = shape(feature["geometry"])
        lat = float(commune_geom.centroid.x)
        long = float(commune_geom.centroid.y)
        codecommune = feature["properties"]["code"]
        geo_data.append({'codecommune': codecommune, 'lat': lat, 'long': long})
        if codecommune=='13055':
            for i in range(1, 16+1):
                if i < 10:
                    geo_data.append({'codecommune': f'1320{i}', 'lat': lat, 'long': long})
                else:
                    geo_data.append({'codecommune': f'132{i}', 'lat': lat, 'long': long})
        if codecommune=='69123':
            for i in range(1, 9+1):
                geo_data.append({'codecommune': f'6938{i}', 'lat': lat, 'long': long})
        if codecommune=='75056':
            for i in range(1, 20+1):
                if i < 10:
                    geo_data.append({'codecommune': f'7510{i}', 'lat': lat, 'long': long})
                else:
                    geo_data.append({'codecommune': f'751{i}', 'lat': lat, 'long': long})
    geo_df = pd.DataFrame(geo_data)
    processor.dfc = pd.merge(processor.dfc, geo_df, on='codecommune', how='left')
    missing = processor.dfc[processor.dfc['lat'].isna()]['codecommune']
    missing_deps = missing.str[:2]
    geo_mapping = (
        geo_df.groupby(geo_df['codecommune'].str[:2])
        .first()[['lat', 'long']]
        .to_dict(orient='index')
    )
    lat_long_updates = missing_deps.map(lambda dep: geo_mapping.get(dep, {'lat': None, 'long': None}))
    processor.dfc.loc[processor.dfc['lat'].isna(), ['lat', 'long']] = pd.DataFrame(lat_long_updates.tolist(), index=missing.index)
    print("Geographical data added successfully")

def create_dataset_common(processor, year, election_type):
    """Create dataset for a specific election year and type"""
    features_year_list = set([col[:-4] for col in processor.dfc.columns if re.search(r'\d{4}$', col)])
    year_pattern = r'(\d{4})$'
    columns_to_keep = [col for col in processor.dfc.columns 
                      if not re.search(year_pattern, col)
                      or int(re.search(year_pattern, col).group(1)) == year]
    dataset = processor.dfc[columns_to_keep]
    processor.feature_backfill_map = {} 
    new_columns = {}
    dataset = dataset.copy()
    max_past_year = 5
    missing_year_features = features_year_list - set([re.sub(rf'{year}$', '', col) for col in dataset.columns])
    for feature in missing_year_features:
        for year_offset in range(1, max_past_year+1):
            previous_year = year - year_offset
            if previous_year < year-max_past_year:
                break
            if f'{feature}{previous_year}' in processor.dfc.columns:
                new_columns[f'{feature}{year}'] = processor.dfc[f'{feature}{previous_year}']
                processor.feature_backfill_map[f'{feature}{year}'] = f'{feature}{previous_year}'
                break
    if new_columns:
        new_columns_df = pd.DataFrame(new_columns, index=processor.dfc.index)
        dataset = pd.concat([dataset, new_columns_df], axis=1)
    election = f'{year}-{election_type}'
    previous_election = _find_previous_election(processor, election)
    target_cols = ['codecommune', f'inscrits{election}'] + [f'pvote{var}{year}-{election_type}' for var in processor.config['vote_variables']]
    for var in processor.config['vote_variables']:
        if previous_election is not None:
            target_cols.append(f'pvote{var}{previous_election}')
            previous_previous_election = _find_previous_election(processor, previous_election)
            if previous_previous_election is not None:
                target_cols.append(f'pvote{var}{previous_previous_election}')
    Y = processor.dfs[target_cols].copy()
    if year==1946 and election_type=='referendum':
        Y.dropna(subset=['codecommune'], inplace=True)
    print('Drop commune that have no election results for the given vote statistics')
    for var in processor.config['vote_variables']:
        rows_before = Y.shape[0]
        Y.dropna(subset=[f'pvote{var}{election}'], inplace=True)
        rows_after = Y.shape[0]
        communes_dropped = rows_before - rows_after
        print(f"Number of communes dropped for {var}: {communes_dropped}")
    dataset_common = pd.merge(Y, dataset, on='codecommune', how='left', validate='one_to_one')
    dataset_common.rename(columns={f'inscrits{election}': f'inscrits{year}'}, inplace=True)
    for var in processor.config['vote_variables']:
        dataset_common.rename(columns={f'pvote{var}{election}': f'pvote{var}{year}'}, inplace=True)
        if previous_election is not None:
            dataset_common.rename(columns={f'pvote{var}{previous_election}': f'pvoteprevious{var}{year}'}, inplace=True)
            previous_previous_election = _find_previous_election(processor, previous_election)
            if previous_previous_election is not None:
                dataset_common.rename(columns={f'pvote{var}{previous_previous_election}': f'pvotepreviousprevious{var}{year}'}, inplace=True)
    return dataset_common

def create_global_dataset(processor):
    """Create the global dataset for modeling"""
    print("Creating global dataset...")
    if processor.config['one_election']:
        processor.global_dataset = create_dataset_common(processor, processor.config['target_year'], processor.config['target_type'])
        processor.global_dataset.rename(columns=lambda col: re.sub(r'(\d{4})$', '', col), inplace=True)
        processor.global_dataset['annee'] = _map_election_year(processor, float(processor.config['target_year']))
        processor.global_dataset['type'] = _map_election_type(processor, processor.config['target_type'], processor.config['encoding_type'])
        election_included = f"{processor.config['target_type']}{processor.config['target_year']}"
    else:
        first_dataset = True
        election_included = []
        for election_type in processor.config['include_elections']:
            year_pattern = rf'pvote[a-zA-Z]*?(\d{{4}}[A-Za-z]*)-' + re.escape(election_type)
            relevant_years = list(set(
                    int(re.search(year_pattern, col).group(1))
                    for col in processor.dfs.columns if re.search(year_pattern, col)
                ))
            for year in relevant_years:
                if year >= processor.config['min_year']:
                    print(f"Processing {election_type} {year}")
                    dataset_common = create_dataset_common(processor, year, election_type)
                    dataset_common.rename(columns=lambda col: re.sub(r'(\d{4})$', '', col), inplace=True)
                    dataset_common['annee'] = _map_election_year(processor, year)
                    dataset_common['type'] = _map_election_type(processor, election_type, processor.config['encoding_type'])
                    election_included.append(f'{election_type}{year}')
                    if first_dataset:
                        processor.global_dataset = dataset_common
                        first_dataset = False
                    else:
                        processor.global_dataset = pd.concat([processor.global_dataset, dataset_common], axis=0, ignore_index=True)
    print(f"Global dataset created: {processor.global_dataset.shape}")
    return election_included

def save_processed_data(processor):
    """Save all processed datasets and configuration file"""
    os.makedirs(processor.config['output_dir'], exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if processor.global_dataset is not None:
        filename = f"data_{'_'.join(processor.config['vote_variables'])}_{processor.config['min_year']}_{'_'.join(processor.config.get('include_elections', []))}_{timestamp}.parquet"
        processor.global_dataset.to_parquet(os.path.join(processor.config['output_dir'], filename), index=False)
    print(f"All processed data saved to {processor.config['output_dir']}")

def _find_previous_feature(processor, feature, features_all):
    """Find the previous year's feature for a given feature"""
    year = re.search(r'(\d{4})$', feature) 
    if year is not None:
        year = year.group(1)
    else: 
        return None
    feat = re.sub(r'(\d{4})$', '', feature)
    year = int(year)
    years = [int(re.search(r'(\d{4})$', col).group(1)) for col in features_all 
            if re.sub(r'(\d{4})$', '', col) == feat and re.search(r'(\d{4})$', col)]
    previous_years = [y for y in years if y < year]
    if len(previous_years) > 0:
        previous_year = max(previous_years)
        return f'{feat}{previous_year}'
    else:
        return None

def _find_previous_election(processor, election):
    """Find the previous election of the same type"""
    year, election_type = election.split('-')
    year = int(year)
    relevant_columns = [col for col in processor.dfs.columns if election_type in col]
    years = [int(re.search(r'(\d{4})', col).group(1)) for col in relevant_columns 
            if re.search(r'(\d{4})', col)]
    previous_years = [y for y in years if y < year]
    if previous_years:
        previous_year = max(previous_years)
        return f"{previous_year}-{election_type}"
    return None

def _map_election_year(processor, year):
    """Encode with gini index (placeholder)"""
    return year

def _map_election_type(processor, election_type, encoding_logic='no'):
    """Map election type to numerical encoding"""
    l = encoding_logic.split('_')
    choice = l[0]
    if choice == 'average_vote':
        var = l[1]
        print(f"Using average vote {var} mapping")
        mapping = {
            'presidentiel': processor.dfs[[col for col in processor.dfs.columns if 'presidentiel' in col and f'pvote{var}'in col]].mean().mean()
            if any('presidentiel' in col and f'pvote{var}' in col for col in processor.dfs.columns) else None,
            'legislative': processor.dfs[[col for col in processor.dfs.columns if 'legislative' in col and f'pvote{var}' in col]].mean().mean()
            if any('legislative' in col and f'pvote{var}' in col for col in processor.dfs.columns) else None,
            'referendum': processor.dfs[[col for col in processor.dfs.columns if 'referendum' in col and f'pvote{var}' in col]].mean().mean()
            if any('referendum' in col and f'pvote{var}' in col for col in processor.dfs.columns) else None
        }
    else:
        mapping = {
                'presidentiel': 0,
                'legislative': 1,
                'referendum': 2
            }
        if choice == 'no':
            print('Using no mapping')
        else:
            print("mapping not recognized, using no mapping")
    return mapping.get(election_type, -1)

import pdb

def main():
    """Main function to run the data processing pipeline"""
    # Initialize processor
    processor = ElectionDataProcessor()

    # Load election data
    processor.load_electoral_data()

    # Load socioeconomic data
    processor.load_socioeconomic_data()


    # Geo data
    processor.add_geographical_data()
    
    # Create global dataset
    election_included = processor.create_global_dataset()

    # split by year/type and save intermediate datasets
    

    processor.save_processed_data()

    if processor.config["quality_filter"]:
        processor.apply_quality_filter()
    else:
        print('Skipping quality filter')

    pdb.set_trace()

    # Feature engineering - set by config
    if not processor.config["features_aug"] == ['raw']:
        processor.apply_feature_engineering()
    else:
        print('Skipping feature augmentation')



    # Save processed data
    processor.save_processed_data()

    print(f"Data processing pipeline completed!")
    print(f"Election include: {election_included}")
   
    return processor.global_dataset

if __name__ == "__main__":
    dataset = main()
