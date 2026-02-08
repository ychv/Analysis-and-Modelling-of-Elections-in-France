import pandas as pd

path = "data/datasets/data_merged_20250922.parquet"

ids = ["annee","codecommune"]
vote = ["pvotepvoteD","pvoteppar","pvotepreviouspvoteD","pvotepreviousppar"]
immo = ["capitalimmobiliercommunes/capitalratio","capitalimmobiliercommunes/percap","capitalimmobiliercommunes/prixbien","capitalimmobiliercommunes/prixm2","capitalimmobiliercommunes/propappartement"]
rev = ["revcommunes/revratio","revcommunes/perrev"]
pib = ["pibcommunes/pibratio"]
chom = ["cspcommunes/pchom"]

col = ids + vote + immo + rev + pib + chom

col = ["annee","codecommune",
       "pvotepvoteD","pvoteppar","pvotepreviouspvoteD","pvotepreviousppar",
       "capitalimmobiliercommunes/capitalratio","capitalimmobiliercommunes/percap","capitalimmobiliercommunes/prixbien","capitalimmobiliercommunes/prixm2","capitalimmobiliercommunes/propappartement",
       "capitalimmobiliercommunes/capitalratio_pctchange","capitalimmobiliercommunes/percap_pctchange","capitalimmobiliercommunes/prixbien_pctchange","capitalimmobiliercommunes/prixm2_pctchange","capitalimmobiliercommunes/propappartement_pctchange",
       "revcommunes/revratio","revcommunes/perrev",
       "revcommunes/revratio_pctchange","revcommunes/perrev_pctchange",
       "pibcommunes/pibratio",
       "pibcommunes/pibratio_pctchange",
       "cspcommunes/pchom",
       "cspcommunes/pchom_pctchange"]

col = ["annee","codecommune","inscrits","lat","long"]

sel = ([("annee",">=",2007),("type","==",1)])

# data2007 = data_loader(path,2007)
# data2007 = filter_large_parquet(path,col,filter=sel)
data = pd.read_parquet(path,engine="pyarrow",columns=col,filters=sel)

# annees = sorted(pd.unique(data['annee']),reverse=True)
# lags = [annees[0] - x for x in annees]

# data = data.sort_values(['commune_id', 'annee'])

# data['annee_prec'] = data.groupby('commune_id')['annee'].shift(1)


# data = data2007[col]
# print(data2007.shape)

data.to_csv("declassement_suite.csv")

# data["deltaED"] = data['pvotepvoteD'] - data["pvotepreviouspvoteD"]
# data = data2007[data2007["deltaED"] > 0]



