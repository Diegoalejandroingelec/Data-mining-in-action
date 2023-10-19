#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 19:59:01 2023

@author: diego
"""
import os
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from utils import save_variable,find_pairs_of_features_based_on_correlation,visualize_data,drop_redundant_features,plot_attributes
from seaborn import heatmap
from sklearn.ensemble import RandomForestRegressor 
import category_encoders as ce
from sklearn import preprocessing
from joblib import dump, load

def check_for_model(file_name):
    if os.path.exists(file_name):
        return True
    else:
        return False
    
def label_function(val):
    return f'{val / 100 * len(df):.0f}\n{val:.0f}%'

def compute_statistics_for_nominal_features(dataset,feature,statistical_description,correlation_with_the_target):
    ### The most common value within this attribute
    mode=dataset[feature].mode()[0]
    ### Entropy of a variable is the measure of uncertainty if it is 0 it means that the variable is deterministic
    entropy=stats.entropy(dataset[feature].replace(dataset[feature].unique(),list(range(len(dataset[feature].unique())))))
    ### Chi-square test p-value that is less than or equal to your significance level indicates there is sufficient evidence to conclude that the observed distribution is not the same as the expected distribution
    
    p=chi_square_test(dataset,feature)
    ### Correlation with TARGET
    corr=correlation_with_the_target[feature]
    
    statistical_description[feature]={'Mode':mode,
                                      'Entropy':entropy,
                                      'Chi_square_test_p':p,
                                      'Correlation':corr}
    return statistical_description


def compute_statistics(dataset,feature,statistical_description,correlation_with_the_target):

    min_val= dataset[feature].min()
    max_value= dataset[feature].max()
    median=dataset[feature].median()
    
    #A percentile rank indicates the percentage of data points in a dataset that are less than or equal to a particular value.
    #For example, the 25th percentile  represents the value below which 25% of the data points fall
    P25=dataset[feature].quantile(0.25)
    P50=dataset[feature].quantile(0.5)
    P75=dataset[feature].quantile(0.75)
    
    corr=correlation_with_the_target[feature]
    ### The most common value within this attribute
    mode=dataset[feature].mode()[0]
    
    statistical_description[feature]={'Median':median,
                                      'Min_value':min_val,
                                      'Max_value':max_value,
                                      'Correlation':corr,
                                      'Mode':mode,
                                      'P25':P25,
                                      'P50':P50,
                                      'P75':P75}
    return statistical_description

def compute_statistics_for_ratio_features(dataset,feature,statistical_description):
    mean= dataset[feature].mean()
    std= dataset[feature].std()
    statistical_description[feature]['Mean']=mean
    statistical_description[feature]['Standard_deviation']=std
    return statistical_description
    
def chi_square_test(dataset,attribute,attribute2='TARGET'):
    # Perform the chi-square test
    
    observed = pd.crosstab(dataset[attribute], dataset[attribute2])
    print(f'TARGET VS {attribute}')
    print(observed)
    chi2, p, dof, expected = stats.chi2_contingency(observed)
    
    # Interpret the results
   
    print(f"P-Value: {p}")
    
    # significance level 
    alpha = 0.05
    
    if p < alpha:
        print(f"There is a significant association between {attribute} and TARGET.")
    else:
        print(f"There is no significant association between {attribute} and TARGET")
        
    return p

    

def plot_features_3D(f1,f2,f3,dataset,ag1=None, ag2=None):

    dataset=dataset.sort_values(by=['TARGET'])
    
    fig = plt.figure(figsize=(12,8))
    
    ax = fig.add_subplot(111, projection='3d')
    
    color_dict = {0:'red', 1:'green'}
    
    names = dataset['TARGET'].unique()
    
    for s in names:
        if s == 1:
            l='No difficulties'
        else:
            l='Difficulties'
        data = dataset.loc[dataset['TARGET'] == s]
        sc = ax.scatter(data[f1], data[f2], data[f3], s=25,
        c=[color_dict[i] for i in data['TARGET']], marker='x',  label=l)
        plt.legend(loc=2, bbox_to_anchor=(1.05, 1))
    if(ag1 and ag2):
        ax.view_init(ag1, ag2)   
    ax.set_xlabel(f1, rotation=150)
    ax.set_ylabel(f2)
    ax.set_zlabel(f3, rotation=60)
    
    
    plt.show()    


def label_encoder_for_quick_training(test_data, train_data, df):
    
    le = preprocessing.LabelEncoder()
    for column_name in train_data.columns:
        if train_data[column_name].dtype == object:
            le.fit(df[column_name])
            train_data[column_name] = le.transform(train_data[column_name])
            test_data[column_name]= le.transform(test_data[column_name])                
        else:
            pass
        
    return train_data,test_data
    
    
df = pd.read_csv('loan_data_training.csv')


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
df=df.drop(['FLAG_MOBIL', 'FLAG_DOCUMENT_12'], axis=1)


## irrelevant
df=df.drop(['SK_ID_CURR'],axis=1)


nan_count = df.isna().sum()

print(nan_count)




######################### Missing Values


df['AMT_REQ_CREDIT_BUREAU_MON'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0], inplace=True)
df['AMT_REQ_CREDIT_BUREAU_YEAR'].fillna(df['AMT_REQ_CREDIT_BUREAU_YEAR'].mode()[0], inplace=True)
df['AMT_REQ_CREDIT_BUREAU_QRT'].fillna(df['AMT_REQ_CREDIT_BUREAU_QRT'].mode()[0], inplace=True)

df['NAME_TYPE_SUITE'].fillna(df['NAME_TYPE_SUITE'].mode()[0], inplace=True)

df['OBS_30_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['DEF_30_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_30_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['OBS_60_CNT_SOCIAL_CIRCLE'].fillna(df['OBS_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)
df['DEF_60_CNT_SOCIAL_CIRCLE'].fillna(df['DEF_60_CNT_SOCIAL_CIRCLE'].mode()[0], inplace=True)

df['EXT_SOURCE_2'].fillna(df['EXT_SOURCE_2'].mean(), inplace=True)
df['AMT_GOODS_PRICE'].fillna(df['AMT_GOODS_PRICE'].mean(), inplace=True)


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


# FILTER OUTLIERS

#Filter outliers with AMT_INCOME_TOTAL greater than $800.000
df=df[df['AMT_INCOME_TOTAL'] < 800000]




for f in df.columns:
    if  len(df[f].unique())>30 and f!='ORGANIZATION_TYPE': 
        df[f].plot(kind = 'box').set_title(f)
        plt.grid()
        plt.show()







############## Encoding categorical variables


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

# for f in df_new.columns:
#     if  len(df_new[f].unique())<30: 
#         df_new.groupby(f).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8})

#         plt.title(f)
#         plt.grid()
#         plt.show()

############## Redundant Features Analysis 

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



number_of_features=8
selected_features,correlation_with_the_target=drop_redundant_features(correlation_matrix,
                                          high_correlated_attributes,number_of_features)


A='DAYS_BIRTH'
B='EXT_SOURCE_2'
correlation_value=correlation_matrix.loc[A, B]
plot_attributes(df_new[A],
               df_new[B],
                A,
                B,
                correlation_value,
                df_new['TARGET'],
                ['No Difficulties','Difficulties'],
                )



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

#nan_count = dataset.isna().sum()

#print(nan_count)

#dataset=dataset.dropna()


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
                 'DAYS_EMPLOYED']
nominal=['FLAG_DOCUMENT_3','REG_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY','FLAG_WORK_PHONE','FLAG_EMAIL']

for f in df_new.columns:
    if  len(df_new[f].unique())>30 and f!='ORGANIZATION_TYPE': 
        df_new[f].plot(kind = 'box').set_title(f)
        plt.grid()
        plt.show()





categorical=[]
for feature in selected_features:
    try:
        if len(dataset[feature].unique())>50:
            bins=int(len(dataset[feature].unique())/10)
        else:
            bins=int(len(dataset[feature].unique()))
            
        if(feature in nominal):
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
        
        ordinal_features_match = [item for item in ordinal_features if item.startswith(feature)]
        ratio_variables_match = [item for item in ratio_variables if item.startswith(feature)]
        nominal_match = [item for item in nominal if item.startswith(feature)]
        
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
        #dataset.groupby(['NAME_INCOME_TYPE']).sum().plot(kind='pie', y='NAME_INCOME_TYPE', autopct='%1.0f%%')

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 5))
        dataset.groupby(feature).size().plot(kind='pie', autopct=label_function, textprops={'fontsize': 8},
                              colors=['tomato', 'gold', 'skyblue','green'], ax=ax1)
        
        dataset[feature].value_counts().plot(kind='bar',title=feature,ax=ax2)
        ax1.set_ylabel('', size=8)
        ax2.set_ylabel('Frequency', size=8)
        plt.tight_layout()
        plt.show()
        
        
        statistical_description=compute_statistics_for_nominal_features(dataset,
                                                                        feature,
                                                                        statistical_description,
                                                                        correlation_with_the_target)

        
df_new[selected_features].to_csv('DATA_PROCESSED_11.csv', index=False)


# Specify the filename where you want to save the variable
filename = 'selected_features_11.pkl'

# Save selected features

save_variable(filename,selected_features)

       
       