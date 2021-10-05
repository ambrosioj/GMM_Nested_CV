#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 20:28:19 2021

@author: ambrosioj
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
import numpy as np
import scipy.io
import timeit
import pickle

#=============================================================================

def nested_cv(k_outer_cv, k_inner_cv, input_data, output_data, parameter_grid):
            
    # Arrays to store scores
    train_scores_accuracy_avg = np.zeros(k_outer_cv)
    train_scores_accuracy_std = np.zeros(k_outer_cv)
    train_scores_fmacro_avg = np.zeros(k_outer_cv)
    train_scores_fmacro_std = np.zeros(k_outer_cv)
    train_scores_fmicro_avg = np.zeros(k_outer_cv)
    train_scores_fmicro_std = np.zeros(k_outer_cv)
    test_scores_accuracy_avg = np.zeros(k_outer_cv)
    test_scores_accuracy_std = np.zeros(k_outer_cv)
    test_scores_fmacro_avg = np.zeros(k_outer_cv)
    test_scores_fmacro_std = np.zeros(k_outer_cv)
    test_scores_fmicro_avg = np.zeros(k_outer_cv)
    test_scores_fmicro_std = np.zeros(k_outer_cv)
    best_params = {"kernel":[], "C":[], "gamma":[], "degree":[], "coef0":[]}
    result_dict = {}

    for i in range(k_outer_cv):
    
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits = k_inner_cv, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits = k_outer_cv , shuffle=True, random_state=i)
        
        clf = GridSearchCV(estimator=svm, param_grid=parameter_grid, \
                           cv=inner_cv, refit=True)
        clf.fit(input_data, classes)        
        best_params["kernel"].append(clf.best_params_["kernel"])
        best_params["C"].append(clf.best_params_["C"])
        if "rbf" in clf.best_params_["kernel"]:
            best_params["gamma"].append(clf.best_params_["gamma"])
            best_params["degree"].append(None)
            best_params["coef0"].append(None)
        elif "poly" in clf.best_params_["kernel"]:
            best_params["gamma"].append(clf.best_params_["gamma"])
            best_params["degree"].append(clf.best_params_["degree"])
            best_params["coef0"].append(clf.best_params_["coef0"])
        elif "linear" in clf.best_params_["kernel"]:
            best_params["gamma"].append(None)
            best_params["degree"].append(None)
            best_params["coef0"].append(None)
    
        # Nested CV with parameter optimization
        scoring = {'acc': 'accuracy',
                   'fmacro': 'f1_macro',
                   'fmicro': 'f1_micro'}
    
        nested_score = cross_validate(clf, X=input_data, y=output_data, \
                        scoring=scoring, cv=outer_cv, return_train_score=True)
        train_scores_accuracy_avg[i] = nested_score["train_acc"].mean()
        train_scores_accuracy_std[i] = nested_score["train_acc"].std()
        train_scores_fmacro_avg[i] = nested_score["train_fmacro"].mean()
        train_scores_fmacro_std[i] = nested_score["train_fmacro"].std()
        train_scores_fmicro_avg[i] = nested_score["train_fmicro"].mean()
        train_scores_fmicro_std[i] = nested_score["train_fmicro"].std()
        test_scores_accuracy_avg[i] = nested_score["test_acc"].mean()
        test_scores_accuracy_std[i] = nested_score["test_acc"].std()
        test_scores_fmacro_avg[i] = nested_score["test_fmacro"].mean()
        test_scores_fmacro_std[i] = nested_score["test_fmacro"].std()
        test_scores_fmicro_avg[i] = nested_score["test_fmicro"].mean()
        test_scores_fmicro_std[i] = nested_score["test_fmicro"].std()
    
    result_dict["train_accuracy_avg"] = train_scores_accuracy_avg
    result_dict["train_accuracy_std"] = train_scores_accuracy_std
    result_dict["train_fmacro_avg"] = train_scores_fmacro_avg
    result_dict["train_fmacro_std"] = train_scores_fmacro_std
    result_dict["train_fmicro_avg"] = train_scores_fmicro_avg
    result_dict["train_fmicro_std"] = train_scores_fmicro_std
    
    result_dict["test_accuracy_avg"] = test_scores_accuracy_avg
    result_dict["test_accuracy_std"] = test_scores_accuracy_std
    result_dict["test_fmacro_avg"] = test_scores_fmacro_avg
    result_dict["test_fmacro_std"] = test_scores_fmacro_std
    result_dict["test_fmicro_avg"] = test_scores_fmicro_avg
    result_dict["test_fmicro_std"] = test_scores_fmicro_std  
    result_dict["best_params"] = best_params
    
    return result_dict

#=============================================================================

tic = timeit.default_timer()
file_1 = scipy.io.loadmat('../Data/dados_excel.mat')
file_2 = scipy.io.loadmat('../Data/features.mat')

classes = file_1["class"]
j_g = file_1["J_g"]
j_l = file_1["J_l"]
alfa1 = file_2["a1"]
alfa2 = file_2["a2"]
alfa3 = file_2["a3"]
mu1 = file_2["mu1"]
mu2 = file_2["mu2"]
mu3 = file_2["mu3"]
sigma1 = file_2["sigma1"]
sigma2 = file_2["sigma2"]
sigma3 = file_2["sigma3"]

k_inner_cv = 5
k_outer_cv = 10   
    
svm = SVC()

p_grid = [{'kernel': ['rbf'],'C': [1, 10, 100, 1000], \
           'gamma': [1/3, 1/5, 1/7,1e-3, 1e-4]},
          {'kernel': ['linear'], 'C': [1, 10, 100, 1000]},
          {'kernel': ['poly'], 'C': [1, 10, 100, 1000], 'gamma': [1e-1, 1, 10]\
           , 'degree': [1, 2, 3, 4], 'coef0': [0, 1e-1, 1, 10]}]



input_vector_3_features = np.concatenate((alfa1, alfa2, alfa3, mu1, mu2, mu3,\
                                         sigma1, sigma2, sigma3), axis=1)
    
input_vector_2_features = np.concatenate((alfa1, alfa2, alfa3, mu1, mu2, mu3)\
                                         , axis=1) 
     
input_vector_1_features = np.concatenate((mu1, mu2, mu3), axis=1)

    
results_3_features = nested_cv(k_outer_cv, k_inner_cv, input_vector_3_features,\
                               classes, p_grid)
results_2_features = nested_cv(k_outer_cv, k_inner_cv, input_vector_2_features,\
                               classes, p_grid)
results_1_features = nested_cv(k_outer_cv, k_inner_cv, input_vector_1_features,\
                               classes, p_grid)

write_file_3 = open('../Data/results_3_features.pkl', "wb")
pickle.dump(results_3_features, write_file_3)
write_file_3.close()

write_file_2 = open('../Data/results_2_features.pkl', "wb")
pickle.dump(results_2_features, write_file_2)
write_file_2.close()

write_file_1 = open('../Data/results_1_features.pkl', "wb")
pickle.dump(results_1_features, write_file_1)
write_file_1.close()
    
toc = timeit.default_timer()
execution_time = str(toc-tic)

print(f"Time Elapsed: {execution_time} seconds")
