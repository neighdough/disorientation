#! /usr/bin/python

import pandas as pd
import numpy as np
import os

os.chdir('/home/nate/dropbox-caeser/Data/DPD_LYDIA/memphis_30')
col = lambda x: [col.lower() for col in x.columns] 
const = pd.read_csv('./data/new_construction_permits.csv')
contr = pd.read_csv('./data/contractor_permits.csv')
perm = pd.read_csv('./data/permits.csv')
const.columns = col(const)
contr.columns = col(contr)
perm.columns = col(perm)
col_diff = lambda x, y: set(x.columns).difference(set(y.columns))
drop_cols = ['loc_name', 'status', 'score', 'match_type', 
                'match_addr', 'side', 'addr_type', 'arc_street']

for df in [const, contr, perm]:
    df.drop(drop_cols, axis=1, inplace=True)
    df['sub_type'] = df['sub_type'].str.lower()
    df['const_type'] = df['const_type'].str.lower()

perm['year'] = perm.issued.str.split('/').str[0]
comb = const.append(contr, ignore_index=True)
comb.year = comb.issued.str[:4]
comb = comb.append(perm, ignore_index=True)
comb['dup'] = comb.duplicated([col for col in comb.columns])
uni = comb[comb.dup == False]

uni.replace({'descriptio':{'\r\n\r\n':' ', '\r\n':' '},
             'fraction':{'MEMP':'Memphis','CNTY':'Memphis','LKLD':'Lakeland',
                         'ARLI':'Arlington', 'MILL':'Millington',np.nan:'Memphis',
                         'BART':'Bartlett','CNY':'Memphis', 'CMTY':'Memphis',
                         'COLL':'Collierville','GTWN':'Germantown', 
                         '`':'Memphis'}}, inplace=True, regex=True)
#replace didn't work when run as part of first replace statement
#regex for 4 or more whitespace chars followed by any number of non-whitespace
#chars followed by any character except line endings
uni.address.replace('\s{4,}[\S]+.*','',inplace=True, regex=True)
uni['state'] = 'TN'

#New Construction
uni['month'] = uni.issued.str[-5:7]
new_cons = uni[uni.const_type == 'new']
demos = uni[uni.const_type == 'demo']