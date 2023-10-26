#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:59:01 2023

@author: diego
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from utils import compute_statistics,compute_statistics_for_ratio_features,compute_statistics_for_nominal_features,check_for_model,save_variable,find_pairs_of_features_based_on_correlation,visualize_data,drop_redundant_features,label_encoder_for_quick_training
from seaborn import heatmap
from sklearn.ensemble import RandomForestRegressor 
import category_encoders as ce
from joblib import dump, load


    
df = pd.read_csv('loan_data_training.csv')
number_of_features=8

###### ZERO or LOW VARIANCE ATTRIBUTES ##############

## low variance
df=df.drop(['FLAG_DOCUMENT_2',
            'FLAG_DOCUMENT_4',
            'FLAG_DOCUMENT_5',
            'FLAG_DOCUMENT_6',
            'FLAG_DOCUMENT_7',
            'FLAG_DOCUMENT_8',
            'FLAG_DOCUMENT_9',
            'FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11',
            'FLAG_DOCUMENT_13',
            'FLAG_DOCUMENT_14',
            'FLAG_DOCUMENT_15',
            'FLAG_DOCUMENT_16',
            'FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_18',
            'FLAG_DOCUMENT_19',
            'FLAG_DOCUMENT_20',
            'FLAG_DOCUMENT_21',
            'FLAG_CONT_MOBILE',
            'FLAG_EMAIL',
            'NAME_CONTRACT_TYPE',
            'REG_REGION_NOT_WORK_REGION',
            'REG_REGION_NOT_LIVE_REGION',
            'AMT_REQ_CREDIT_BUREAU_DAY',
            'AMT_REQ_CREDIT_BUREAU_WEEK',
            'AMT_REQ_CREDIT_BUREAU_HOUR'],axis=1)

## zero variance
df=df.drop(['FLAG_MOBIL',
            'FLAG_DOCUMENT_12'], axis=1)


## irrelevant
df=df.drop(['SK_ID_CURR'],axis=1)



#Count missing values
nan_count = df.isna().sum()

print(nan_count)




######################### MISSING VALUES

## Reaplace nan with mode for categorical or discrete variables
df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0], inplace=True)
df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0], inplace=True)
df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(df['AMT_REQ_CREDIT_BUREAU_QRT'].mode()[0], inplace=True)

df['NAME_TYPE_SUITE'].fillna(df['NAME_TYPE_SUITE'].mode()[0], inplace=True)

df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

## Reaplace nan with mean for ratio or interval
df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].mean(), inplace=True)
df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].mean(), inplace=True)

##Use random forest to predict missing values
train_data = df.dropna().copy()
test_data = df[df['EXT_SOURCE_3'].isnull()].copy()

train_data,test_data=label_encoder_for_quick_training(test_data, train_data, df)
X_train= train_data.drop('EXT_SOURCE_3',axis=1)
y_train= train_data['EXT_SOURCE_3']
X_test= test_data.drop('EXT_SOURCE_3',axis=1)



model_file_name='random_forest_for_missing_values_ext_source_3.joblib'
if(check_for_model(model_file_name)):
    model=load(model_file_name)
    print('model found!')
else:
    model= RandomForestRegressor()
    model.fit(X_train,y_train)
    dump(model, model_file_name)   




predicted_values= model.predict(X_test)

df.loc[df['EXT_SOURCE_3'].isnull(),'EXT_SOURCE_3']=predicted_values



# Drop the rest of instances with nan
df=df.dropna()
nan_count = df.isna().sum()
print(nan_count)



########################################## OUTLIERS
#REPLACE OUTLIER OF DAYS EMPLOYED


train_data=df[df['DAYS_EMPLOYED'] <=0 ].copy()
test_data = df[df['DAYS_EMPLOYED'] > 0 ].copy()



train_data,test_data=label_encoder_for_quick_training(test_data, train_data, df)

X_train= train_data.drop('DAYS_EMPLOYED',axis=1)
y_train= train_data['DAYS_EMPLOYED']
X_test= test_data.drop('DAYS_EMPLOYED',axis=1)



model_file_name='random_forest_for_outlier_correction_days_employed.joblib'
if(check_for_model(model_file_name)):
    model=load(model_file_name)
    print('model found!')
else:
    model= RandomForestRegressor()
    model.fit(X_train,y_train)
    dump(model, model_file_name)  
    
    
predicted_values= model.predict(X_test)


df.loc[df['DAYS_EMPLOYED'] > 0,'DAYS_EMPLOYED']=predicted_values


#### FILTER OUTLIERS

#Filter outliers with AMT_INCOME_TOTAL greater than $800.000
df=df[df['AMT_INCOME_TOTAL'] < 800000]




# for f in df.columns:
#     if  len(df[f].unique())>30 and f!='ORGANIZATION_TYPE': 
#         df[f].plot(kind = 'box').set_title(f)
#         plt.grid()
#         plt.show()







############## ENCODING CATEGORICAL VARIABLES


df_new=df.copy()
cols=df_new.columns
num_cols = df_new._get_numeric_data().columns
categorical=set(cols)-set(num_cols)
df_new=df_new.reset_index()
df_new=df_new.drop(['index'],axis=1)
for attribute in categorical:
    if(attribute=='NAME_EDUCATION_TYPE' or attribute=='WEEKDAY_APPR_PROCESS_START'):
        encoder = LabelEncoder()
        df_new[attribute] = encoder.fit_transform(df_new[attribute])
        save_variable(f'{attribute}_label_encoder_11.pkl',encoder)
    elif(len(df_new[attribute].unique())>10):
        encoder = ce.CountEncoder()
        df_new[attribute] = encoder.fit_transform(df_new[attribute])
        save_variable(f'{attribute}_count_encoder_11.pkl',encoder)
    else:
        # Initialize the OneHotEncoder
        encoder = OneHotEncoder(sparse=False, drop='first')
        
        # Fit and transform the data
        encoded_data = encoder.fit_transform(df_new[[attribute]])
        
        # Convert the encoded data back to a DataFrame for better visualization
        encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names([attribute]))
        
        
        df_new = pd.concat([df_new, encoded], axis=1)
        df_new=df_new.drop([attribute],axis=1)
        save_variable(f'{attribute}_onehot_encoder_11.pkl',encoder)



############## REDUNDANT FEATURE ANALYSIS

correlation_matrix=df_new.corr()


heatmap(np.array(correlation_matrix),cmap="YlGnBu")
plt.title('Correlation Matrix')
plt.show()

###### FIND HIGH CORRELATED VARIABLES #################
threshold=0.7

high_correlated_attributes=find_pairs_of_features_based_on_correlation(threshold,
                                                                       correlation_matrix,
                                                                       True)
###### FIND LOW CORRELATED VARIABLES #################
threshold=0.1

low_correlated_attributes=find_pairs_of_features_based_on_correlation(threshold,
                                                                       correlation_matrix,
                                                                       False)
###################### plot High Correlated variables


visualize_data(high_correlated_attributes,
               df_new,
               correlation_matrix,
               ['No Difficulties','Difficulties'],
               len(high_correlated_attributes))

###################### plot Low Correlated variables


visualize_data(low_correlated_attributes,
               df_new,
               correlation_matrix,
               ['No Difficulties','Difficulties'],
               10)




selected_features,correlation_with_the_target=drop_redundant_features(correlation_matrix,
                                                                      high_correlated_attributes,
                                                                      number_of_features)


# A='DAYS_BIRTH'
# B='EXT_SOURCE_2'
# correlation_value=correlation_matrix.loc[A, B]
# plot_attributes(df_new[A],
#                df_new[B],
#                 A,
#                 B,
#                 correlation_value,
#                 df_new['TARGET'],
#                 ['No Difficulties','Difficulties'],
#                 )



# f1=correlation_with_the_target.index[0]
# f2=correlation_with_the_target.index[1]
# f3=correlation_with_the_target.index[2]


# plot_features_3D(f1,f2,f3,df_new,40,130)


# import seaborn as sns

# pp = sns.pairplot(df_new[selected_features], height=1.8, aspect=1.8,
#               plot_kws=dict(edgecolor="k", linewidth=0.5),
#               diag_kind="kde", diag_kws=dict(fill=True),hue="TARGET")

# fig = pp.fig
# fig.subplots_adjust(top=0.93, wspace=0.3)
# t = fig.suptitle('Import data Pairwise Plots', fontsize=14) 
    




selected_features.append('TARGET')
dataset=df_new[selected_features]




statistical_description={}


ordinal_features=['REGION_RATING_CLIENT_W_CITY',]

ratio_variables=['AMT_REQ_CREDIT_BUREAU_YEAR',
                 'OBS_30_CNT_SOCIAL_CIRCLE',
                 'DEF_30_CNT_SOCIAL_CIRCLE',
                 'EXT_SOURCE_2',
                 'EXT_SOURCE_3',
                 'DAYS_BIRTH',
                 'DAYS_REGISTRATION',
                 'DAYS_ID_PUBLISH',
                 'DAYS_LAST_PHONE_CHANGE',
                 'AMT_GOODS_PRICE',
                 'DAYS_EMPLOYED',
                 'CNT_CHILDREN',
                 'AMT_INCOME_TOTAL',
                 'AMT_CREDIT',
                 'AMT_ANNUITY',
                 'AMT_GOODS_PRICE']

nominal=['FLAG_DOCUMENT_3',
         'REG_CITY_NOT_WORK_CITY',
         'REG_CITY_NOT_LIVE_CITY',
         'FLAG_WORK_PHONE',
         'FLAG_EMAIL',
         'NAME_TYPE_SUITE',
         'NAME_INCOME_TYPE',
         'NAME_EDUCATION_TYPE',
         'NAME_FAMILY_STATUS',
         'NAME_HOUSING_TYPE',
         'NAME_CONTRACT_TYPE',
         'CODE_GENDER',
         'FLAG_OWN_CAR',
         'FLAG_OWN_REALTY',
         'TARGET']

for f in df_new.columns:
    if  len(df_new[f].unique())>30 and f!='ORGANIZATION_TYPE': 
        df_new[f].plot(kind = 'box').set_title(f)
        plt.grid()
        plt.show()




def label_function(val):
    return f'{val / 100 * len(df):.0f}\n{val:.0f}%'

for feature in selected_features:
    try:
        if len(dataset[feature].unique())>50:
            bins=int(len(dataset[feature].unique())/50)
        else:
            bins=int(len(dataset[feature].unique()))
            
        
        
        ordinal_features_match = [feature for item in ordinal_features if feature.startswith(item)]
        ratio_variables_match = [feature for item in ratio_variables if feature.startswith(item)]
        nominal_match = [feature for item in nominal if feature.startswith(item)]
        
        if(len(nominal_match)>=1):
            fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
            dataset.groupby(feature).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8},
                                  colors=['tomato', 'gold', 'skyblue','green'], ax=ax1)
            
            dataset[feature].value_counts().plot(kind='bar',title=feature,ax=ax2)
            ax1.set_ylabel('', size=8)
            ax2.set_ylabel('Frequency', size=8)
            plt.tight_layout()
            plt.show()
        else:
            dataset[feature].plot(kind='hist', bins=bins,title=feature,align='mid')
            plt.show()
        
        if (len(ordinal_features_match)>=1) or (len(ratio_variables_match)>=1):
            statistical_description=compute_statistics(dataset,
                                                       feature,
                                                       statistical_description,
                                                       correlation_with_the_target)
        if (len(nominal_match)>=1):
            statistical_description=compute_statistics_for_nominal_features(dataset,
                                                                            feature,
                                                                            statistical_description,
                                                                            correlation_with_the_target)
        if (len(ratio_variables_match)>=1):
            statistical_description=compute_statistics_for_ratio_features(dataset,
                                                                          feature,
                                                                          statistical_description)

        
               
    except:
        print('error')
        #dataset.groupby(['NAME_INCOME_TYPE']).sum().plot(kind='pie', y='NAME_INCOME_TYPE', autopct='%1.0f%%')

        # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        # dataset.groupby(feature).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8},
        #                       colors=['tomato', 'gold', 'skyblue','green'], ax=ax1)
        
        # dataset[feature].value_counts().plot(kind='bar',title=feature,ax=ax2)
        # ax1.set_ylabel('', size=8)
        # ax2.set_ylabel('Frequency', size=8)
        # plt.tight_layout()
        # plt.show()
        
        
        # statistical_description=compute_statistics_for_nominal_features(dataset,
        #                                                                 feature,
        #                                                                 statistical_description,
        #                                                                 correlation_with_the_target)

        
df_new[selected_features].to_csv('DATA_PROCESSED_11.csv', index=False)


# Specify the filename where you want to save the variable
filename = 'selected_features_11.pkl'

# Save selected features

save_variable(filename,selected_features)

       
       