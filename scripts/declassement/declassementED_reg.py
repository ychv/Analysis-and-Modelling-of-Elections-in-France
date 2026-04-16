import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.metrics import r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split
from src.data.data_handler import DataHandler
from src.model.models_rf import RegressorWrapper
from src.config import random_seed, vote_features,data_year_filter,missing_data_thresh,type_pres,type_legis,cols_to_keep,full_dataset_path

# path = "../../data/datasets/data_merged_20250922.parquet"
path = full_dataset_path

ids = ["annee","codecommune"]
geo = ["lat","long"]
immo = ["capitalimmobiliercommunes/capitalratio","capitalimmobiliercommunes/percap","capitalimmobiliercommunes/prixbien","capitalimmobiliercommunes/prixm2","capitalimmobiliercommunes/propappartement","capitalimmobiliercommunes/capitalratio_pctchange","capitalimmobiliercommunes/percap_pctchange","capitalimmobiliercommunes/prixbien_pctchange","capitalimmobiliercommunes/prixm2_pctchange","capitalimmobiliercommunes/propappartement_pctchange"]
rev = ["revcommunes/revratio","revcommunes/perrev","revcommunes/revratio_pctchange","revcommunes/perrev_pctchange"]
pib = ["pibcommunes/pibratio","pibcommunes/pibratio_pctchange"]
chom = ["cspcommunes/pchom","cspcommunes/pchom_pctchange"]

col = ids + vote_features + immo + rev + pib + chom + geo


# sel = ([("annee",">=",2007),("type","==",1)])

# data2007 = filter_large_parquet(path,col,filter=sel)
# data = pd.read_parquet(path,engine="pyarrow",columns=col,filters=sel)

dataload = DataHandler(path,random_seed)
dataload.load_data(data_year_filter,2022,type_legis,col,missing_data_thresh)
dataload.aggregate_dep_inscrits()
data = dataload.agg_df

# data["dep"] = data["dep"].astype(str)

########################### Departement split ##################################
 
dep_GE = ['08',"51","10","52","55","54","57","88","67","68"]
dep_N = ['62','59','02','80','60']
dep_PACA = ['05','04','06','84','13','83']

data_GE = data[data['dep'].isin(dep_GE)]
data_N = data[data['dep'].isin(dep_N)]
data_PACA = data[data['dep'].isin(dep_PACA)]

def get_Xy(datadep):
    y = datadep["pvotepvoteD"]
    X = datadep.drop(columns=vote_features #+ ['dep','annee']
                     )
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=random_seed)
    return X,y,X_train, X_test, y_train, y_test

X_GE, y_GE, X_GE_train, X_GE_test, y_GE_train, y_GE_test = get_Xy(data_GE)
X_N, y_N, X_N_train, X_N_test, y_N_train, y_N_test = get_Xy(data_N)
X_PACA, y_PACA, X_PACA_train, X_PACA_test, y_PACA_train, y_PACA_test = get_Xy(data_PACA)

############################### Regression with Random Forest #####################################
print('######################### Random Forest #########################\n')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('-------------------- Grand Est --------------------\n')
    reg_GE = RegressorWrapper(base=RandomForestRegressor,n_estimators=100)
    reg_GE.fit(X_GE.to_numpy(),y_GE.to_numpy())
    perm_GE = reg_GE.compute_permutation_importance(X_GE,y_GE,random_seed,1,'r2')
    print(perm_GE)

    print('-------------------- Nord --------------------\n')
    reg_N = RegressorWrapper(base=RandomForestRegressor,n_estimators=100)
    reg_N.fit(X_N.to_numpy(),y_N.to_numpy())
    perm_N = reg_N.compute_permutation_importance(X_N,y_N,random_seed,1,'r2')
    print(perm_N)

    print('-------------------- PACA --------------------\n')
    reg_PACA = RegressorWrapper(base=RandomForestRegressor,n_estimators=100)
    reg_PACA.fit(X_PACA.to_numpy(),y_PACA.to_numpy())
    perm_PACA = reg_PACA.compute_permutation_importance(X_PACA,y_PACA,random_seed,1,'r2')
    print(perm_PACA)

    print('-------------------- Nord contre Grand Est --------------------\n')
    yNvsGEpred = reg_GE.predict(X_N.to_numpy())
    print(reg_GE.compute_permutation_importance(X_N,y_N,random_seed,1,'r2'))

    print('-------------------- PACA contre Grand Est --------------------\n')
    yPACAvsGEpred = reg_GE.predict(X_PACA.to_numpy())
    print(reg_GE.compute_permutation_importance(X_PACA,y_PACA,random_seed,1,'r2'))

############################### Regression with Lasso #####################################
print('######################### LASSO ##############################\n')
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print('-------------------- Grand Est --------------------\n')
    reg_GE = LassoCV()
    reg_GE.fit(X_GE.to_numpy(),y_GE.to_numpy())
    coeff_GE = reg_GE.coef_
    print(coeff_GE)

    print('-------------------- Nord --------------------\n')
    reg_N = LassoCV()
    reg_N.fit(X_N.to_numpy(),y_N.to_numpy())
    coeff_N = reg_N.coef_
    print(coeff_N)

    print('-------------------- PACA --------------------\n')
    reg_PACA = LassoCV()
    reg_PACA.fit(X_PACA.to_numpy(),y_PACA.to_numpy())
    coeff_PACA = reg_PACA.coef_
    print(coeff_PACA)