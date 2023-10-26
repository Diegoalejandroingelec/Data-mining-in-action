#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 22:25:48 2023

@author: diego
"""

import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing
from utils import plot_loss_metric_curves,save_variable,load_variable

#Import metrics to evaluate model performance
from sklearn import metrics
#Import confusion matrix to visualize classification results
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

#Import train_test_split to divide the dataset in training and testing
from sklearn.model_selection import train_test_split

from tensorflow.keras import regularizers

# Import all machine learning algorithms and libraries
from sklearn.svm import SVC


# Import tensorflow to implement neural networks 
import tensorflow as tf

from joblib import dump, load

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from tensorflow.keras.layers import Dense, Dropout, BatchNormalization,PReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import KFold

def find_best_threshold(y_test,prediction,model_file,model_name):
    fpr, tpr, thresholds = metrics.roc_curve(y_test,  prediction)
    auc = metrics.roc_auc_score(y_test, prediction)
    
    
    plt.plot([0,1],[0,1],'r--',label='Random Classifier')
    plt.plot(fpr,tpr,label=f"{model_name}, auc="+str(auc))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc=4)
    plt.title('ROC curve')
    plt.grid()
    plt.show()
    
    
    
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    save_variable(f'optimal_threshold_{model_file}',optimal_threshold)
    return optimal_threshold,fpr, tpr,auc
    
    
def create_neural_network(input_dim,
                          amount_neurons_hidden_layers,
                          output_dim):
    
    
    
    
    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=(input_dim,)))
    
    # Increase the number of units in the initial layers
    
    for amount_neurons in amount_neurons_hidden_layers:
        model.add(Dense(amount_neurons,
                        kernel_regularizer=regularizers.l2(0.001),
                        kernel_initializer=tf.initializers.HeNormal()))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Dropout(0.5))
    
    
    
    model.add(Dense(output_dim,
                    activation='sigmoid',
                    kernel_initializer='glorot_uniform'))
    return model 


#1 Training
#2 Prediction for unknown dataset   
#3 Testing
mode_=int(input("Choose mode from the following: 1- Training 2- Prediction for unknown dataset   3- Testing \n your choice is: "))

n_folds=11
#Choose model
model_option = int(input("Choose one model from the following: 1- Support vector machine classifier 2- Decision Tree  3- Random Forest 4- Deep Neural Network \n your choice is: "))
df_new = pd.read_csv('DATA_PROCESSED.csv')


labels= np.array(df_new['TARGET'])
data=df_new.drop(['TARGET'],axis=1)       
print(data.shape)



scaler = preprocessing.StandardScaler()
data=scaler.fit_transform(data)
save_variable('scaler.pkl', scaler)


kf = KFold(n_splits=n_folds,random_state=42,shuffle=True)


data_frame=pd.DataFrame(np.concatenate((data,np.expand_dims(labels, axis=1)), axis=1))

# Split the dataset in train 80% and test 20%. random_state=42 to ensure replicability 
X_train_total, X_val, y_train_total, y_val = train_test_split(data_frame.iloc[:,0:data.shape[1]],
                                                              data_frame.iloc[:,data.shape[1]],
                                                              test_size=0.2,
                                                              random_state=42) 

if(mode_==1):

    X_dataframe=pd.concat([X_train_total, y_train_total], axis=1)

    
    for fold_, (train_index, test_index) in enumerate(kf.split(X_dataframe)):
        print(f"Fold {fold_}:")
        X_train = X_dataframe.iloc[train_index,0:data.shape[1]]
        X_test = X_dataframe.iloc[test_index,0:data.shape[1]]
        y_train = X_dataframe.iloc[train_index,data.shape[1]]
        y_test = X_dataframe.iloc[test_index,data.shape[1]]
        

    
    
          
    
        
    

        if (model_option == 1):
            model_file=f'svm_trained_{fold_}.joblib'
            # Call the contructor SVC (Support Vector Machine for Classification) to create the model
            #with a large C, the model will focus on minimizing the classification error, even if this implies a small margin, which might generate overfitting.
            #Conversely, with a small C, the algorithm will prioritize maximizing the margin, producing a more robust model that might generalize better for unseen data
            
            # A small gamma results in a smoother decision boundary, while a large gamma leads to a more localized and wiggly decision boundary
            model = SVC(C=1.0,
                        kernel='rbf',
                        gamma='scale',
                        probability=True,
                        verbose=True,
                        max_iter=-1)
            # Train the model using the method fit() using just the training dataset
            model.fit(X_train, y_train)
            dump(model, model_file) 
        elif(model_option == 2):
            model_file=f'decision_tree_trained_{fold_}.joblib'
            # Call the constructor DecisionTreeClassifier() to create the model
            #The algorithm selects the best possible feature and threshold to split the data based on the chosen criterion
            #min_samples_split: This parameter specifies the minimum number of samples required to perform a split at a node
            #min_samples_leaf: This parameter defines the minimum number of samples required in a leaf node
            model = DecisionTreeClassifier(criterion='entropy',
                                           splitter='best',
                                           max_depth=20,
                                           min_samples_split=2,
                                           min_samples_leaf=1)
            # Train the model using the method fit() using just the training dataset
            model.fit(X_train, y_train)
            dump(model, model_file) 
        elif(model_option == 3):
            model_file=f'random_forest_trained_{fold_}.joblib'
            # Call the constructor RandomForestClassifier() to create the model
            model = RandomForestClassifier(n_estimators=80,
                                           criterion='gini',
                                           max_depth=20,
                                           min_samples_split=2,
                                           min_samples_leaf=1)
            # Train the model using the method fit() using just the training dataset
            model.fit(X_train, y_train)
            dump(model, model_file)    
        elif(model_option == 4): 
            
            #Define the name of the file which will contain the best weights
            model_file = f'neural_network_best_weights_fold_{fold_}'
            # Number of epochs to train the model
            epochNo = 80
            # Number of instances that will be analized at the same time by the model
            batchSize=64
            
            
            model=create_neural_network(input_dim=X_train.shape[1],
                                      amount_neurons_hidden_layers=[256,256,256,256,256,256],
                                      output_dim=1)
            
            model.summary()
            # Use Adam optimizer with an adjustable learning rate
            optimizer = Adam(learning_rate=0.001)
            model.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                          optimizer=optimizer,
                          metrics=['accuracy'])
            
            # Implement learning rate reduction  callbacks
            reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                          factor=0.1,
                                          patience=5,
                                          min_lr=0.00001)
        
            
        
            
            #Define callback function to check after each epoch the accuracy in the testing set and save the best weights
            model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=model_file,
                                                                            save_weights_only=True,
                                                                            monitor='val_accuracy',
                                                                            mode='max',
                                                                            save_best_only=True)
            
            #Train the model using the training dataset. It also evaluates the accuracy and loss for the testing set after each epoch
            #It does not update the weights using the testing dataset.
            history=model.fit(X_train,
                      y_train,
                      epochs=epochNo,
                      batch_size=batchSize,
                      verbose=1,
                      validation_data=(X_test, y_test),
                      callbacks=[model_checkpoint_callback, reduce_lr])
        
        
        
        
        
        
                                      
            # Plot the training curves
            plot_loss_metric_curves([history.history['loss'],history.history['val_loss']],
                                    [history.history['accuracy'],history.history['val_accuracy']],
                                      True,
                                      epochNo,
                                      'accuracy')
        
        
        
        
            
        
        
        
        
        
        #Predict the values using the testing dataset   
        if(model_option==4):
            # Load the best model
            model.load_weights(model_file)
            #####PREDICTIONS FOR DNN
            prediction = model.predict(X_test)
        else:
            #####PREDICTIONS FOR THE REST OF THE MODELS
            model = load(model_file) 
            prediction=model.predict_proba(X_test)
            prediction=prediction[:,1].reshape(-1,1)
        
        
        fpr, tpr, thresholds = metrics.roc_curve(y_test,  prediction)
        auc = metrics.roc_auc_score(y_test, prediction)
        
        
        plt.plot([0,1],[0,1],'r--',label='Random Clasifier')
        plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc=4)
        plt.title('ROC curves comparison')
        plt.grid()
        plt.show()
        
        
        
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        save_variable(f'optimal_threshold_{model_file}',optimal_threshold)
        y_test=np.array(y_test)
        
        binarized_prediction = preprocessing.Binarizer(threshold=optimal_threshold).transform(prediction).astype(int)[:,0]
        
        # Calculate accuracy of the model using the testing dataset
        # (tp+tn)/ (tp+tn+fp+fn) where tp is the number of true positives, fp the number of false positives, fn the number of false negatives
        # tn the number of true negatives
        accuracy_scores=[]
        
        accuracy_scores.append( metrics.accuracy_score(y_test, binarized_prediction))
        
        
        # Calculate precision of the model using the testing dataset
        # The precision is tp / (tp + fp) where tp is the number of true positives and fp the number of false positives
        # average='weighted' Calculate metrics for each label, and find their average weighted to account for label imbalance
        precision_test=[]
        
        precision_test.append(metrics.precision_score(y_test,binarized_prediction))
        
        # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.
        recall_test=[]
        recall_test.append(metrics.recall_score(y_test,binarized_prediction))
        
        # # F1 = 2 * (precision * recall) / (precision + recall)
        f1_test=[]
        f1_test.append(metrics.f1_score(y_test,binarized_prediction))
        
        # #Print the performance metrics
        print(f'Accuracy = {np.mean(accuracy_scores)}\n Precision = {np.mean(precision_test)}\n Recall = {np.mean(recall_test)}\n F1_score = {np.mean(f1_test)}\n')
        
        #Plot Confusion matrix
        fig,ax = plt.subplots(figsize=(5,4),dpi = 100)
        cm = confusion_matrix(y_test,binarized_prediction)
        cmp = ConfusionMatrixDisplay(cm)
        cmp.plot(ax = ax)
        plt.xticks(rotation=90)
        plt.show()
        
        
if(mode_==3 or mode_==1):
    print('VALIDATION DATASET')
    predictions=[]
    for fold_ in range(n_folds):

        if(model_option==4):
            model_name='Deep Neural Network'
            model=create_neural_network(input_dim=X_val.shape[1],
                                      amount_neurons_hidden_layers=[256,256,256,256,256,256],
                                      output_dim=1)
            model_file=f'neural_network_best_weights_fold_{fold_}'
            # Load the best model
            model.load_weights(model_file)
            #####PREDICTIONS FOR DNN
            prediction = model.predict(X_val)
        else:
            if (model_option == 1):
                model_name='Support Vector Machine'
                model_file=f'svm_trained_{fold_}.joblib'
            elif(model_option == 2):
                model_name='Decision Tree'
                model_file=f'decision_tree_trained_{fold_}.joblib'
            elif(model_option == 3): 
                model_name='Random Forest'
                model_file=f'random_forest_trained_{fold_}.joblib'
            #####PREDICTIONS FOR THE REST OF THE MODELS
            model = load(model_file) 
            prediction=model.predict_proba(X_val)
            prediction=prediction[:,1].reshape(-1,1)
            
            

        
        optimal_threshold=load_variable(f'optimal_threshold_{model_file}')
        binarized_prediction = preprocessing.Binarizer(threshold=optimal_threshold).transform(prediction).astype(int)[:,0]
        
        print('F1_SCORE: ', metrics.f1_score(y_val, binarized_prediction))

        
        predictions.append(binarized_prediction)
    
    
    predictions=np.array(predictions)
    
   
    
    #final_prediction= preprocessing.Binarizer(threshold=int(n_folds/2)).transform(np.expand_dims(np.sum(predictions, axis=0), axis=1)).astype(int)[:,0]
    
    final_prediction_probabilities=np.sum(predictions, axis=0)/n_folds
    
    T,fpr,tpr,auc=find_best_threshold(y_val,final_prediction_probabilities,
                          "_".join(model_file.split('_')[:-2])+'_final_model',model_name)
    
    
    final_prediction=preprocessing.Binarizer(threshold=T).transform(final_prediction_probabilities.reshape(-1,1)).astype(int)[:,0]
    
    
    
    # Calculate accuracy of the model using the testing dataset
    # (tp+tn)/ (tp+tn+fp+fn) where tp is the number of true positives, fp the number of false positives, fn the number of false negatives
    # tn the number of true negatives

    
    accuracy_score=  metrics.accuracy_score(y_val, final_prediction)
    
    
    # Calculate precision of the model using the testing dataset
    # The precision is tp / (tp + fp) where tp is the number of true positives and fp the number of false positives
    # average='weighted' Calculate metrics for each label, and find their average weighted to account for label imbalance

    
    precision_test=metrics.precision_score(y_val, final_prediction)
    
    # The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives.

    recall_test=metrics.recall_score(y_val, final_prediction)
    
    # # F1 = 2 * (precision * recall) / (precision + recall)

    f1_test=metrics.f1_score(y_val, final_prediction)
    
    # #Print the performance metrics
    print(f'Accuracy = {accuracy_score}\n Precision = {precision_test}\n Recall = {recall_test}\n F1_score = {f1_test}\n')
    
    #Plot Confusion matrix
    fig,ax = plt.subplots(figsize=(5,4),dpi = 100)
    cm = confusion_matrix(y_val, final_prediction)
    cmp = ConfusionMatrixDisplay(cm)
    cmp.plot(ax = ax)
    plt.title(model_name)
    plt.show()
        
        
        
        
    
    # prediction_svm=final_prediction_probabilities
    # auc_svm=metrics.roc_auc_score(y_val, prediction_svm)
    # fpr_SVM=fpr
    # tpr_SVM=tpr
    

    # prediction_dt=final_prediction_probabilities
    # auc_dt = metrics.roc_auc_score(y_val, prediction_dt)
    # fpr_DT=fpr
    # tpr_DT=tpr
    

    # prediction_rf=final_prediction_probabilities
    # auc_rf = metrics.roc_auc_score(y_val, prediction_rf)
    # fpr_RF=fpr
    # tpr_RF=tpr
    

    # prediction_dnn=final_prediction_probabilities
    # auc_dnn = metrics.roc_auc_score(y_val, prediction_dnn)
    # fpr_DNN=fpr
    # tpr_DNN=tpr
    

    
   
   
    
    # plt.plot([0,1],[0,1],'r--',label='Random Classifier')
    # plt.plot(fpr_SVM,tpr_SVM,label=" Support Vector Machine" )
    # plt.plot(fpr_DT,tpr_DT,label="Decision Tree")
    # plt.plot(fpr_RF,tpr_RF,label="Random forest")
    # plt.plot(fpr_DNN,tpr_DNN,label=" Deep Neural Network")
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.legend(loc=4)
    # plt.title('ROC curves comparison')
    # plt.grid()
    # plt.show()
        
    
if(mode_==2):
    
    selected_features=load_variable('selected_features.pkl')    
    df = pd.read_csv('loan_data_unknown.csv')
    #df=df[selected_features]
    
    cols=df.columns
    num_cols = df._get_numeric_data().columns
    categorical=set(cols)-set(num_cols)
    
    
    # Replace NaN values with the mode in each column
    for column in df.columns:
        mode_value = df[column].mode()[0]  # Get the mode of the column
        df[column].fillna(mode_value, inplace=True)  # Replace NaN with the mode
    
    for attribute in categorical:
        try:
            if(attribute=='NAME_EDUCATION_TYPE' or attribute=='WEEKDAY_APPR_PROCESS_START'):
                
                encoder=load_variable(f'{attribute}_label_encoder.pkl')
                
                df[attribute] = encoder.transform(df[attribute])
                
            elif(len(df[attribute].unique())>10):
                encoder = load_variable(f'{attribute}_count_encoder.pkl')
                df[attribute] = encoder.transform(df[attribute])
                
            else:
                # Initialize the OneHotEncoder
                encoder = load_variable(f'{attribute}_onehot_encoder.pkl')
                
                # Fit and transform the data
                encoded_data = encoder.transform(df[[attribute]])
                
                # Convert the encoded data back to a DataFrame for better visualization
                encoded = pd.DataFrame(encoded_data, columns=encoder.get_feature_names([attribute]))
                
                
                df = pd.concat([df, encoded], axis=1)
                df=df.drop([attribute],axis=1)
        except:
            #For variables that were deleted in the preprocessing step
            print('file name was not found')
            
    selected_features=selected_features[0:-1]      
    ids=df['SK_ID_CURR']
    df=df[selected_features]
    
    scaler = load_variable('scaler.pkl')
    df=scaler.transform(df)
    
    
      
    
    predictions=[]
    for fold_ in range(n_folds):

        if(model_option==4):
            model=create_neural_network(input_dim=X_val.shape[1],
                                      amount_neurons_hidden_layers=[256,256,256,256,256,256],
                                      output_dim=1)
            model_file=f'neural_network_best_weights_fold_{fold_}'
            # Load the best model
            model.load_weights(model_file)
            #####PREDICTIONS FOR DNN
            prediction = model.predict(df)
        else:
            if (model_option == 1):
                model_file=f'svm_trained_{fold_}.joblib'
            elif(model_option == 2):
                model_file=f'decision_tree_trained_{fold_}.joblib'
            elif(model_option == 3): 
                model_file=f'random_forest_trained_{fold_}.joblib'
            #####PREDICTIONS FOR THE REST OF THE MODELS
            model = load(model_file) 
            prediction=model.predict_proba(df)
            prediction=prediction[:,1].reshape(-1,1)
    
        optimal_threshold=load_variable(f'optimal_threshold_{model_file}')
        binarized_prediction = preprocessing.Binarizer(threshold=optimal_threshold).transform(prediction).astype(int)[:,0]
        
        predictions.append(binarized_prediction)
   
   # final_prediction= preprocessing.Binarizer(threshold=int(n_folds/2)).transform(np.expand_dims(np.sum(predictions, axis=0), axis=1)).astype(int)[:,0]
    final_prediction_probabilities=np.sum(predictions, axis=0)/n_folds
    
    T=load_variable(f"optimal_threshold_{'_'.join(model_file.split('_')[:-2])+'_final_model'}")
    
    final_prediction=preprocessing.Binarizer(threshold=T).transform(final_prediction_probabilities.reshape(-1,1)).astype(int)[:,0]


    
    result=pd.DataFrame({'SK_ID_CURR': ids,
                         'TARGET': final_prediction
                        })
    result.to_csv('result.csv',index=False)