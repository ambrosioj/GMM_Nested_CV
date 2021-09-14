#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 14 20:28:19 2021

@author: ambrosioj
"""

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_validate, KFold
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import scipy.io

def nested_cv(number_trials, input_data, output_data, parameter_grid):
            
    # Arrays to store scores
    train_scores_accuracy_avg = np.zeros(number_trials)
    train_scores_accuracy_std = np.zeros(number_trials)
    train_scores_fmacro_avg = np.zeros(number_trials)
    train_scores_fmacro_std = np.zeros(number_trials)
    train_scores_fmicro_avg = np.zeros(number_trials)
    train_scores_fmicro_std = np.zeros(number_trials)
    test_scores_accuracy_avg = np.zeros(number_trials)
    test_scores_accuracy_std = np.zeros(number_trials)
    test_scores_fmacro_avg = np.zeros(number_trials)
    test_scores_fmacro_std = np.zeros(number_trials)
    test_scores_fmicro_avg = np.zeros(number_trials)
    test_scores_fmicro_std = np.zeros(number_trials)
    best_params = {"C":[],"kernel":[],"gamma":[]}
    result_dict = {}

    for i in range(number_trials):
    
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "GroupKFold", "LeaveOneOut", "LeaveOneGroupOut", etc.
        inner_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=i)
        
        clf = GridSearchCV(estimator=svm, param_grid=parameter_grid, cv=inner_cv)
        clf.fit(input_vector, classes)
        best_params["C"].append(clf.best_params_["C"])
        best_params["kernel"].append(clf.best_params_["kernel"])
        if "gamma" in clf.best_params_:
            best_params["gamma"].append(clf.best_params_["gamma"])
        else:
            best_params["gamma"].append(0)
    
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
    return result_dict

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

NUM_TRIALS = 10    
    
svm = SVC()

p_grid = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]},
          {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]

input_vector = np.concatenate((alfa1, alfa2, alfa3, mu1, mu2, mu3,\
                               sigma1, sigma2, sigma3), axis=1)
    
teste_1 = nested_cv(NUM_TRIALS, input_vector, classes,p_grid)