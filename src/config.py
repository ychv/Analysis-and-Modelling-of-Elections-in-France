### Global config
random_seed = 42


### Paths
full_dataset_path = r"data/data_merged_20250922.parquet"


### Data config
data_year_filter = 1990
missing_data_thresh = 0.22
type_pres, type_legis = 0, 1

selected_num_features = [
    "pibcommunes/pibtot_pctchange",
    "capitalimmobiliercommunes/capitalratio_pctchange",
    "capitalimmobiliercommunes/capitalratioagglo_pctchange",
    "long",
    "popcommunes/peragglo",
    "capitalimmobiliercommunes/capitalimmo",
    "pibcommunes/pibtot",
    "lat",
    "proprietairescommunes/nlogement",
    "popcommunes/percommu",
    "agesexcommunes/popf",
    "agesexcommunes/poph",
    "diplomescommunes/supf_rank",
    "popcommuneselecteurs/electeurs_rank",
    "capitalimmobiliercommunes/capitalimmoagglo_delta",
    "diplomescommunes/suph_rank",
    "cspcommunes/cadr_rank",
    "pibcommunes/pibtot_delta",
    "cspcommunes/capi_rank",
]

vote_features = [
    "inscrits",
    "pvoteppar",
    "pvotepvoteG",
    "pvotepvoteC",
    "pvotepvoteD",
    "pvotepvoteCG",
    "pvotepvoteCD",
    "pvotepreviousppar",
    "pvotepreviouspreviousppar",
    "pvotepreviouspvoteG",
    "pvotepreviouspreviouspvoteG",
    "pvotepreviouspvoteC",
    "pvotepreviouspreviouspvoteC",
    "pvotepreviouspvoteD",
    "pvotepreviouspreviouspvoteD",
    "pvotepreviouspvoteCG",
    "pvotepreviouspreviouspvoteCG",
    "pvotepreviouspvoteCD",
    "pvotepreviouspreviouspvoteCD",
]

cols_to_keep = ["codecommune, annee"]


### Model config
