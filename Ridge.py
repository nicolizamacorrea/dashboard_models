#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import math
import lightgbm as lgb
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from explainerdashboard.explainers import *
from explainerdashboard.dashboards import *
from explainerdashboard.datasets import *


X_avg=['FE_RESIN_LOAD','TA04_L2_5_L_SIC0726_PV','TA04_L2_5_L_MIC0730_PV','TA04_L2_5_WIC0728_PV' ,'TA04_L2_5_GREC_DENS_PV','FE_ESPESOR_GRECON','FE_CALCULATED_DENSITY_MAT',
'FE_SAWDUST_RATIO', 'TA04_L2_4_H_WI0386_PV','FE_EXTERNAL_CHIP_RATIO', 'FE_PRESSURE_1'  ,'FE_PRESSURE_2_3',
'FE_PRESSURE_4_5','FE_PRESSURE_6_7','FE_PRESSURE_8_9','FE_PRESSURE_10_11','FE_PRESSURE_12_13','FE_PRESSURE_14_15',
'FE_PRESSURE_16_17','FE_THICKNESS_20','FE_THICKNESS_22','FE_THICKNESS_25','FE_THICKNESS_29','FE_THICKNESS_33','FE_THICKNESS_37']



model = pickle.load(open('Ridge.pkl', 'rb'))
df_test = pd.read_csv('df_test.csv', sep=',')

X_test = df_test[X_avg]
y_test= df_test['IB_avg']

explainer = RegressionExplainer(model, X_test, y_test, idxs=X_avg)


db = ExplainerDashboard(explainer, title="Ridge IB avg",
                        model_summary=True,  # you can switch off individual tabs
                        contributions=True,
                        shap_dependence=True,
                        shap_interaction=True,
                        decision_trees=True)
db.run(port=8051)



# In[ ]:




