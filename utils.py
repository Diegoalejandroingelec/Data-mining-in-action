#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 11:52:08 2023

@author: diego
"""
#Import useful Libraries
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pickle

def save_variable(filename,variable):
    with open(filename, 'wb') as file:
        pickle.dump(variable, file)


def load_variable(filename):
    # Load selected features
    with open(filename, 'rb') as file:
        variable = pickle.load(file)
    return variable  


       
#Function that plots two features of the dataset
def plot_attributes(attr1_values,attr2_values,name1,name2,correlation_value,values,classes):
    
    #Define the color for each class
    colors = ListedColormap(['green','red'] )
    
    #Plot the two specified features
    scatter=plt.scatter(attr1_values,attr2_values,s=10,c=values, cmap=colors)
    #add title, labels and legend
    plt.title('{} vs {} corr={:.3f}'.format(name1,name2,correlation_value),fontsize = 8, fontweight ='bold')
    plt.xlabel(name1)
    plt.ylabel(name2)
    plt.legend(handles=scatter.legend_elements()[0], labels=classes,fontsize=8)
    #Show plot
    plt.show()

# Function for simple dimensionality reduction
# It works based on the correlation between variables. If two variables are highly correlated
# that means that with just one variable it is possible to explain the other, consecuently it is redundant to have both features.
# This function selects the feature that is more correlated with tha labels for the prediction task.
def drop_redundant_features(correlation_matrix,high_correlated_attributes,number_of_features=-1):

    #Obtain the correlation of all features with labels and sort it in descending order 
    correlation_with_the_target=correlation_matrix['TARGET'].abs().sort_values(ascending=False)
    #Iterates over an array that contains pairs of high correlated variables
    for item in high_correlated_attributes:
        #for each pair of variables it gives the format of list because it is saved as a frozenset
        attributes=list(item)
        try:
            #Get the correlation value of both features with the labels
            c1=correlation_with_the_target[attributes[0]]
            c2=correlation_with_the_target[attributes[1]]
            #Delete attribute related with c2 (the one that is less correlated with the labels)
            if(c1>=c2):
                try:
                    #Try to drop the feature if it exists in the dataset
                    correlation_with_the_target=correlation_with_the_target.drop([attributes[1]],axis=0)
                except:
                    #Do nothing if the feature was alreade deleted
                    print("")
            #Delete attribute related with c1 (the one that is less correlated with the labels)
            else:
                try:
                    #Try to drop the feature if it exists in the dataset
                    correlation_with_the_target=correlation_with_the_target.drop([attributes[0]],axis=0)
                except:
                    #Do nothing if the feature was alreade deleted
                    print("")
        except:
            #Do nothing if the feature was alreade deleted
            print("")
            
    #Drop labels from the features selected        
    correlation_with_the_target=correlation_with_the_target.drop(['TARGET'],axis=0)
    #Create a list with the names of the selected features reducing the complexity of the dataset
    if(number_of_features==-1):
        selected_features=list(correlation_with_the_target.index)
    else:
        selected_features=list(correlation_with_the_target[0:number_of_features].index)
    
    return selected_features,correlation_with_the_target

#Function to visualize the data. It receives as input:
# A List "attribute_names": which contains pairs of feature names to visualize
# A dataframe "data": which contains the data of each feature
# A correlation matrix "correlation_matrix": to obtain the correlation between features
# A list "classes": which contains the names of each class
# An integer number "pairs": Useful to limit the number of plots in case that attribute_names is too large

def visualize_data(attribute_names,data,correlation_matrix,classes,pairs=10):
    #iterates over attribute_names until pairs
    for item in attribute_names[0:pairs]:
        #Get the names of each feature
        attributes=list(item)
        attr1_values=data[attributes[0]]
        attr2_values=data[attributes[1]]
        #Get the correlation value between those two variables
        correlation_value=correlation_matrix.loc[attributes[0], attributes[1]]
        
        #Plot all the information
        plot_attributes(attr1_values,
                        attr2_values,
                        attributes[0],
                        attributes[1],
                        correlation_value,
                        data['TARGET'],
                        classes
                        )
 
#Function that returns a list of pairs of variables that are low or high correlated based on a threshold.It receives as input:
# threshold: float number which defines the correlation value. it must be between 0 and 1 because 
# it compares the absolut value of the correlation
# correlation_matrix: correlation matrix of the dataset
# high_correlation: boolean. If it is true the function returns the pairs of variables that are high correlated based on the threshold 
# otherwise it returns the pairs of variables that are low correlated based on the threshold
def find_pairs_of_features_based_on_correlation(threshold,correlation_matrix,high_correlation):
    #find high or low correlated variables
    if(high_correlation):
        correlation=np.abs(correlation_matrix)>threshold
    else:
        correlation=np.abs(correlation_matrix)<threshold
        
    #Get the name of the variables
    pairs_of_correlated_attributes = correlation.rename_axis(index='index', columns='col').stack().loc[lambda x: x==True]
    
    #Delete names from pairs_of_correlated_attributes of values within the diagonal wich is the correlation of that variable with itself
    #for example [{'feature1','feature1'},...] will be deleted
    correlated_attributes=[]
    for i in range(len(pairs_of_correlated_attributes)):
        if pairs_of_correlated_attributes[[i]].index[0][0] != pairs_of_correlated_attributes[[i]].index[0][1]:
            correlated_attributes.append(pairs_of_correlated_attributes[[i]].index[0])
    
    
    #Delete redundant pairs for example [{'feature1','feature2'}, {'feature2','feature1'}, ...] is redundant, so just one pair will be kept
    correlated_attributes=list(set([frozenset(x) for x in correlated_attributes]))
    #Returns the list of correlated variables
    return correlated_attributes

#Function that receives loss values and metric values per epoch to plot the curves
def plot_loss_metric_curves(loss,metric,training,epochNo,metric_name):
    
    fig, (ax1, ax2) = plt.subplots(1, 2)


    #plot epochs vs Loss for training set
    ax1.plot(list(range(epochNo)),loss[0],label="training")
    #plot epochs vs Loss
    ax1.plot(list(range(epochNo)),loss[1],label="testing")
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss (mean_squared_error)')
    ax1.set_title('loss vs Epochs')
    ax1.legend()
    
    #plot epochs vs metric for training set
    ax2.plot(list(range(epochNo)),metric[0],label="training")
    #plot epochs vs metric for testing set
    ax2.plot(list(range(epochNo)),metric[1],label="testing")
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel(f'{metric_name}')
    ax2.set_title(f'{metric_name} vs Epochs')
    ax2.legend()
    fig.tight_layout(h_pad=10)
    plt.show()    