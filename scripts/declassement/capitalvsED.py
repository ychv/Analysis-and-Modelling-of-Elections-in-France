import pandas as pd
from src.data.data_handler import DataHandler
from src.model.models_rf import RegressorWrapper

path = "../../data/datasets/data_merged_20250922.parquet"

ids = ["annee","codecommune","inscrits","lat","long"]
vote = ["pvotepvoteD","pvoteppar","pvotepreviouspvoteD","pvotepreviousppar","pvotepvoteG","pvotepreviouspvoteG","pvotepvoteCG","pvotepreviouspvoteCG","pvotepvoteCD","pvotepreviouspvoteCD","pvotepvoteC","pvotepreviouspvoteC"]
immo = ["capitalimmobiliercommunes/capitalratio","capitalimmobiliercommunes/percap","capitalimmobiliercommunes/prixbien","capitalimmobiliercommunes/prixm2","capitalimmobiliercommunes/propappartement","capitalimmobiliercommunes/capitalratio_pctchange","capitalimmobiliercommunes/percap_pctchange","capitalimmobiliercommunes/prixbien_pctchange","capitalimmobiliercommunes/prixm2_pctchange","capitalimmobiliercommunes/propappartement_pctchange"]
rev = ["revcommunes/revratio","revcommunes/perrev","revcommunes/revratio_pctchange","revcommunes/perrev_pctchange"]
pib = ["pibcommunes/pibratio","pibcommunes/pibratio_pctchange"]
chom = ["cspcommunes/pchom","cspcommunes/pchom_pctchange"]

col = ids + vote + immo + rev + pib + chom


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

# data.to_csv("declassement_suite.csv")

# data["deltaED"] = data['pvotepvoteD'] - data["pvotepreviouspvoteD"]
# data = data2007[data2007["deltaED"] > 0]

