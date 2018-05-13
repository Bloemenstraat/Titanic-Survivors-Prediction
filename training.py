#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 13 23:24:28 2018

@author: therapie
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def train(data, result):
    x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.3)
    
    logistic = LogisticRegression()
    logistic.fit(x_train, y_train)
    print(logistic.score(x_test, y_test))
    
    