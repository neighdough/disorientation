import sys
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor 
from sklearn.model_selection import train_test_split, cross_val_score
from collections import OrderedDict
from configparser import ConfigParser
from sqlalchemy import create_engine
import click
import json

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

def add_missing(df):
    idx_cols = ["month", "year", "name"]
    idx = list(itertools.product(*[range(1,13),
                                   range(2002,2017),
                                   df.name.unique()]))
    df.set_index(idx_cols, inplace=True)
    df = df.reindex(idx)
    df.reset_index(inplace=True)
    df.permit.fillna(0, inplace=True)
    return df

#-----------------------------------------------------------------------------
#--------------------------- Urban Index -------------------------------------
#-----------------------------------------------------------------------------
sql =("select w.geoid10, numpermit, numdemo, ninter/sqmiland inter_density,"
        "totpop/sqmiland popdensity, hhinc, "
        #"pct_wh,pct_bl, "
        "pct_bl + pct_ai + pct_as + "
        "pct_nh + pct_ot + pct_2m pct_nonwh, "
        "pct_pov_tot, pct_to14 + pct_15to19 pct_u19,"
        "pct_20to24, pct_25to34, pct_35to49, pct_50to66, "
        "pct_67up, hsng_density, pct_comm, age_comm, "
        "pct_dev, pct_vac, park_dist, park_pcap, gwy_sqmi, "
        "age_bldg, mdnhprice,mdngrrent, pct_afford, "
        "pct_hu_vcnt, affhsgreen, foreclose,pct_own, "
        "pct_rent, pct_mf, age_sf, mdn_yr_lived, "
        "strtsdw_pct, bic_index,"
        "b08303002 + b08303003 + b08303004 tt_less15,"
        "b08303005 + b08303006 + b08303007 tt_15to29,"
        "b08303008 + b08303009 + b08303010 + b08303011 "
        "+ b08303012 + b08303013 tt30more,"
        "b08301002 tm_caralone, b08301010 tm_transit, "
        "b08301018 tm_bicycle, b08301019 tm_walk, mmcnxpsmi, "
        "transit_access, bic_sqmi, rider_sqmi, vmt_per_hh_ami, "
        "walkscore, autos_per_hh_ami, pct_canopy, "
        "green_bldgs_sqmi, pct_chgprop, avg_hours, "
        "emp_ovrll_ndx, pct_labor_force, emp_ndx, pct_unemp, "
        "pct_commercial, pct_arts, pct_health, pct_other, "
        "pct_pubadmin, pct_util, pct_mining, pct_ag, "
        "pct_food, pct_retail, pct_wholesale, pct_manuf, "
        "pct_construction, pct_waste_mgmt, pct_ed, pct_info, "
        "pct_transport, pct_finance, pct_realestate, "
        "pct_prof_services, pct_mgmt,pct_lowinc_job, "
        "pct_b15003016 pct_no_dip, pct_b15003017 pct_dip, "
        "pct_b15003018 pct_ged, pct_b15003019 pct_uni_1yr, "
        "pct_b15003020 pct_uni_no_deg, pct_b15003021 pct_assoc, "
        "pct_b15003022 pct_bach, pct_b15003023 pct_mast, "
        "pct_b15003024 pct_prof_deg, pct_b15003025 pct_phd, "
        "elem_dist, middle_dist, high_dist, "
        "pvt_dist, chldcntr_dist, cmgrdn_dist, frmrmkt_dist, "
        "library_dist, commcenter_dist,pct_medicaid, "
        "bpinc_pcap, hosp_dist, pol_dist, fire_dist, "
        "os_sqmi, pct_imp, wetland_sqmi, brnfld_sqmi, "
        "mata_route_sqmi, mata_stop_sqmi "
    "from (select count(s.fid) ninter, t.wkb_geometry, geoid10 "
            "from tiger_tract_2010 t, streets_carto_intersections s "
            "where st_intersects(s.wkb_geometry, t.wkb_geometry) "
            "group by geoid10, t.wkb_geometry) bg, "
            "(select geoid10, "
            "count(distinct case when const_type = 'new' "
                "then permit end) numpermit, "
            "count(distinct case when const_type = 'demo' "
                "then permit end) numdemo "
            "from permits p, tiger_tract_2010 t "
            "where st_within(p.wkb_geometry, t.wkb_geometry) "
            "group by t.geoid10) p, "
            "wwl_2017_tract w "
    "where w.geoid10 = bg.geoid10 "
    "and w.geoid10 = p.geoid10;") 

#df = pd.read_sql(sql, engine)
#X = df[x_vars]
#y_net = df.net
#X_pos = df[df.net > 0][x_vars]
#y_pos = df[df.net > 0]['net']
#X_neg = df[df.net < 0][x_vars]
#y_neg = df[df.net < 0]['net']

def corr_matrix(df):
    """Create correlation matrix and generate heatmap

    df (pandas dataframe): Pandas dataframe containing all of the features
        to be used for matrix
    """
    from string import ascii_letters
    import numpy as np
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    X = df[x_vars]
    sns.set(style="white")

    # Compute the correlation matrix
    corr = X.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=(11, 9))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5})    
    ax.set_xticklables(corr.index, fontsize=6)
    ax.set_yticklabels(corr.index, fontsize=6)
    plt.savefig("./disorientation/figs/corr_matrix.png")

def remove_correlates(X, coeff=.8):
    """Identify which features have highest correlation coefficient. The method
    returns a list of column names that should be excluded from an analysis,
    but also updates a table in the project database

    Args:
        X (pandas dataframe): Pandas dataframe containing only predictor variables
        coeff (float): threshold to identify values that should be removed

    """

    corr = X.corr()
    #correlates = set()
    correlates = list()
    for col_idx in range(len(corr.columns)):
        for row_idx in range(col_idx):
            corr_coeff = corr.iloc[col_idx, row_idx]
            if  abs(corr_coeff) > coeff:
                d = {"variable": corr.columns[col_idx],
                     "corr_variable": corr.index[row_idx],
                     "corr_coeff": corr_coeff}
                correlates.append(d)
    df_corr = pd.read_json(json.dumps(correlates), orient="records")
    df_corr.to_sql("correlation_values", con=engine, if_exists="replace", 
        index=False)
    return df_corr.variable.to_list()

def correlated_feature_list(df):
    try:
        df_corr = pd.read_sql("select * from correlation_values", engine)
        return df_corr.variable.to_list()
    except:
        corr_feats = remove_correlates(df)
        return corr_feats


def create_feature_scores(df, yname="net", n_features=1, create_plot=False):
    """Select most important features using recursive feature elimination (RFE)
        in conjunction with random forest regression and then plot the accuracy
        of the fit.

    References:

        Title: Feature Ranking RFE, Random Forest, Linear Models
        Author: Arthur Tok
        Date: June 18, 2018
        Code version: 80
        Availability: https://bit.ly/37ngDg8

        Title: Selecting good features – Part IV: stability selection, RFE and everything side by side
        Author: Ando Saabas
        Date: December 20, 2014
        Availability: https://bit.ly/2SGCuLx


    """
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor 
    from sklearn.feature_selection import RFE

    corr_features = correlated_feature_list(df)
    X = df[[col for col in x_vars if col not in corr_features]]
    cols = X.columns
    feature_rank = {}
    y = df[yname]
    accuracy = []

    def rank_features(ranks, names, order=1):
        minmax = MinMaxScaler()
        feature_rank = minmax.fit_transform(order*np.array([ranks]).T).T[0]
        feature_rank = map(lambda x: round(x,2), feature_rank)
        return dict(zip(names, feature_rank))

    #********************* Recursive Feature Elimination ***********    
    #RFE with Linear Regression
    lr = LinearRegression(normalize=True)
    lr.fit(X,y)
    rfe = RFE(lr, n_features_to_select=n_features, verbose=3)
    rfe.fit(X,y)
    feature_rank["rfe-lr"] = rank_features(list(map(float, rfe.ranking_)), cols,
                        order=-1)
    accuracy.append(["rfe-lr", rfe.score(X,y)])    

    
    #RFE with Random Forest Regression
    rfr = RandomForestRegressor(max_features="sqrt", random_state=RANDOM_STATE)
    rfr.fit(X,y)
    rfe = RFE(rfr, n_features_to_select=n_features, verbose=3)
    rfe.fit(X,y)
    feature_rank["rfe-rfr"] = rank_features(list(map(float, rfe.ranking_)), cols,
                        order=-1)
    accuracy.append(["rfe-rfr", rfe.score(X,y)])

    #************************* Regression *****************************
    #Linear Regression alone
    lr = LinearRegression(normalize=True)
    lr.fit(X,y)
    feature_rank["lr"] = rank_features(np.abs(lr.coef_), cols)
    #Ridge Regression
    ridge = Ridge(alpha=7)
    ridge.fit(X,y)
    feature_rank["ridge"] = rank_features(np.abs(ridge.coef_), cols)
    accuracy.append(["ridge", ridge.score(X,y)])

    #Lasso
    lasso = Lasso(alpha=.05)
    lasso.fit(X,y)
    feature_rank["lasso"] = rank_features(np.abs(lasso.coef_), cols)
    accuracy.append(["lasso", lasso.score(X,y)])

    #Random Forest Regression alone
    rfr = RandomForestRegressor(max_features="sqrt", random_state=RANDOM_STATE)
    rfr.fit(X,y)
    feature_rank["rfr"] = rank_features(rfr.feature_importances_, cols)
    accuracy.append(["rfr", rfr.score(X,y)])

    r = {}
    for col in cols:
        r[col] = round(np.mean([feature_rank[method][col] 
                    for method in feature_rank.keys()]),2)
    methods = sorted(feature_rank.keys())
    feature_rank["mean"] = r
    df_feature_rank = pd.DataFrame.from_dict(feature_rank)
    df_feature_rank.to_sql("feature_rank_{}".format(yname),engine,
                            if_exists='replace')
    sort_feat_rank = df_feature_rank.sort_values("mean", ascending=False)
    sort_feat_rank["colnames"] = sort_feat_rank.index
    #plot feature rankings
    if create_plot:
        f = sns.catplot(x="mean", y="colnames", data=sort_feat_rank, kind="bar",
                        palette="coolwarm", height=22)
        f.set_yticklabels(sort_feat_rank.colnames,fontsize=10)
        f.set_xlabels("Mean Feature Importance")
        f.set_ylabels("Column Name")
        f.fig.tight_layout(pad=6.)
        f.fig.suptitle("Mean Feature Importance for {}".format(yname))
        plt.savefig("./disorientation/figs/bar_feat_ranking_{}.png".format(yname))
    return accuracy

def select_features(yname="net", index_only=False, min_score=.0):
    """Select the feature rank table (feature_rank_<yname>) from postgres

    Args:
        yname (str): the suffix for which postgres table should be selected. 
            Accepted values are net, scale_const, or scale_demo. Defaults to 'net'
        index_only (bool): False if only the index column containing the column
            names should be returned, True if all columns from the table should
            be returned. Defaults to False
        min_score (float): the minimum mean importance score that should be returned.
            Defaults to .0 for all values.
    
    Returns:

    """
    cols = "index" if index_only else "*"
    params = {"yname": yname, "cols": cols, "mean": min_score}
    sql = "select {cols} from feature_rank_{yname} where mean >= {mean}"
    df = pd.read_sql(sql.format(**params), engine)
    return df


def scatter_plot(df, y="net"):
    """Generates scatter plot matrix for all predictor variables against a y-value
    such as net construction, total construction or toal demolition.
    """
    corr_cols = correlated_feature_list(df)
    cols = sorted([col for col in x_vars if col not in corr_cols])
    nrows, ncols = 10,10
    f, axes = plt.subplots(nrows, ncols, sharex=False, sharey=True,
                    tight_layout=True, figsize=(24,24))
    var_pos = 0
    def plot(var_pos, row, col):
        #if y-value is for net construction, add two plots, one for net-poitive
        #construction, the other for net negative
        if y == "net":
            df[df.net < 0].plot.scatter(x=cols[var_pos], y=y, marker="<",
                ax=axes[row,col],color="Purple")
            df[df.net >= 0].plot.scatter(x=cols[var_pos], y=y, marker=">",
                ax=axes[row,col], color="Green")       
        else:
            color = lambda x: "Green" if "const" in x else "Purple"
            df.plot.scatter(x=cols[var_pos], y=y, marker=">",
                ax=axes[row,col], color=color(y))
    for row in range(nrows):
        for col in range(ncols):
            if var_pos < len(cols):
                plot(var_pos, row, col)
            var_pos += 1
    plt.savefig("./disorientation/figs/scatter_plot_all_feats_{}.png".format(y))
    plt.close()

def num_trees(df, create_plot=False,feat_score=.25, yname="net"):

    features = select_features(index_only=True, min_score=feat_score, yname=yname)
    #determinte number of trees in forest
    ensemble_clfs = [
        ("RFR, max_features='sqrt'|red|-",
            RandomForestRegressor(warm_start=True, oob_score=True,
                                max_features="sqrt",
                                random_state=RANDOM_STATE
                                )),
        ("RFR, max_features='log2'|green|-",
            RandomForestRegressor(warm_start=True, max_features='log2',
                                oob_score=True,
                                random_state=RANDOM_STATE
                                )),
        ("RFR, max_features=None|blue|-",
            RandomForestRegressor(warm_start=True, max_features=None,
                                oob_score=True,
                                random_state=RANDOM_STATE
                                ))]
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    min_estimators = 15
    max_estimators = 500
    X = df[features["index"]]
    y_net = df[yname]
    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y_net)
            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        label, color, linestyle = label.split('|')
        plt.plot(xs, ys, label=label, color=color,
                linestyle=linestyle)

    if create_plot:
        plt.xlim(min_estimators, max_estimators)
        plt.xlabel("n_estimators")
        plt.ylabel("OOB error rate")
        #plt.legend(bbox_to_anchor=(0, 1.1, 1., .102), loc="upper center", ncol=2)
        plt.legend(ncol=2)
        title = ("Estimator at Feature Mean of {0} with {1} Features\n"
                 "for Column '{2}'")
        plt.title(title.format(feat_score, features.shape[0], yname), pad=10)
        plt.tight_layout()
        min_score_format = int(feat_score*100)
        plt.savefig("./disorientation/figs/rfr_accuracy_{0}_{1}.png".format(yname,min_score_format))
        plt.close()


@click.command()
@click.option("--correlation", "-c", is_flag=True)
def main(correlation):
    X = df[x_vars]
    #scaling function
    #Net Construction PD
    y = df.net
    if correlation:
        corr_matrix(df)

if __name__=="__main__":
    main()