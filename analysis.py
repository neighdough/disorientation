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

min_max_scale = lambda x: (x-x.min())/(x.max() - x.min())
std_scale = lambda x: (x-x.mean())/float(x.std())
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

df = pd.read_sql(sql, engine)
x_vars = [col for col in df.columns if col not in 
            ['numpermit', 'numdemo', 'geoid10', 'wkb_geometry', 
             'scale_const', 'scale_demo', 'net']]
#X = df[x_vars]
y_net = df.net
X_pos = df[df.net > 0][x_vars]
y_pos = df[df.net > 0]['net']
X_neg = df[df.net < 0][x_vars]
y_neg = df[df.net < 0]['net']

def corr_matrix(df):
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
                d = {"var": corr.columns[col_idx],
                     "corr_var": corr.index[row_idx],
                     "corr_coeff": corr_coeff}
                correlates.append(d)
    df_corr = pd.read_json(json.dumps(correlates), orient="records")
    df_corr.to_sql("correlation_values", con=engine, if_exists="replace", 
        index=False)
    return df_corr.var.tolist()



def select_features(df):
    from sklearn.feature_selection import RFE
    from sklearn.ensemble import RandomForestRegressor 

    #scaling function
    #Net Construction PD
    df['scale_const'] = min_max_scale(df.numpermit)
    df['scale_demo'] = min_max_scale(df.numdemo)
    #permit column is actually net construction, but needs to be named permit
    #to run correctly in fact_heatmap function
    df['net'] = df.scale_const - df.scale_demo
    X = df[x_vars]
    y = df.net
    rf = RandomForestRegressor()
    rfe = RFE(rf, n_features_to_select=1)
    rfe.fit(X,y)


@click.command()
@click.option("--correlation", "-c", is_flag=True)
def main(correlation):
    df = pd.read_sql("select * from all_features")
    df.fillna(0, inplace=True)
    X = df[x_vars]
    if correlation:
        corr_matrix(df)

if __name__=="__main__":
