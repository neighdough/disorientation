import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA
from configparser import ConfigParser
from sqlalchemy import create_engine
import os
import sys
from collections import OrderedDict
import matplotlib.pyplot as plt
home = os.getenv("HOME")
sys.path.append(os.path.join(home, "dev"))
from disorientation import analysis

cnx_dir = os.getenv("CONNECTION_INFO")
parser = ConfigParser()
parser.read(os.path.join(cnx_dir, "db_conn.ini"))
psql_params = {k:v for k,v in parser._sections["disorientation"].items()}
psql_string = "postgresql://{user}:{password}@{host}:{port}/{dbname}"
engine = create_engine(psql_string.format(**psql_params))
pd.set_option('display.width', 180)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.max_rows', 125)
df = pd.read_sql("select * from all_features", engine)
df.fillna(0, inplace=True)
x_vars = [col for col in df.columns if col not in 
            ['numpermit', 'numdemo', 'geoid10', 'wkb_geometry', 
             'scale_const', 'scale_demo', 'net']]
min_max_scale = lambda x: (x-x.min())/(x.max() - x.min())
std_scale = lambda x: (x-x.mean())/float(x.std())
df['scale_const'] = min_max_scale(df.numpermit)
df['scale_demo'] = min_max_scale(df.numdemo)
#permit column is actually net construction, but needs to be named permit
#to run correctly in fact_heatmap function
df['net'] = df.scale_const - df.scale_demo
RANDOM_STATE = 1416
yname = "net"#"scale_const"#"net"
y = df[yname]
features = analysis.select_features(yname, True, .25)
#X = df[features["index"]]
#X = min_max_scale(X)
#X = df[x_vars]
# 30% examples in test data
x_train, x_test, y_train, y_test = train_test_split(X,
                                         y, 
                                         test_size = 0.3, 
                                         random_state = RANDOM_STATE)

rfr = RandomForestRegressor(max_features=None, warm_start=True,
        oob_score=True, random_state=RANDOM_STATE)

from scipy.stats import spearmanr, pearsonr
min_estimators = 10
max_estimators = 200
rfr_error = OrderedDict()
for i in range(min_estimators, max_estimators + 1):
    rfr.set_params(n_estimators=i)
    rfr.fit(X, y)
    oob = rfr.oob_score_
    y_pred = rfr.oob_prediction_
    sp = spearmanr(y, y_pred)
    pe = pearsonr(y, y_pred)
    feat_imp = rfr.feature_importances_
    rfr_error[i] = {'error':oob, 
                'spearman': sp, 
                'pearson': pe, 
                'feat_imp': feat_imp}
    print(i, '\n\toob: ', oob, '\n\tspearman: ', sp.correlation)
    print('\tpearson: ', pe[0])
    print()

pca = PCA(n_components=1)
X = df[x_vars]
X_new = pca.fit_transform(X)
y_pos_idx = y[y >= 0].index
y_neg_idx = y[y < 0].index
plt.figure(figsize=(8,8))
plt.scatter(X_new[y_pos_idx], y_pred[y_pos_idx], c='Green', s=100)
plt.scatter(X_new[y_neg_idx], y_pred[y_neg_idx], c='Purple', s=100)
plt.ylabel('Net Construction', fontsize=20)