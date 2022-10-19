#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 27 12:40:12 2022

@author: lauritzstorch
"""


import sys
import os
import numpy as np


# Set Path
PATH = '/Users/lauritzstorch/MMM Dropbox/Lauritz Storch/Mac/Desktop/Data Analytics II Assignments/Self_Studies'
os.chdir(PATH)

import function_file as fct

# define the name for the output file
OUTPUT_NAME = 'self_study_output'

# save the console output in parallel to a txt file
orig_stdout = sys.stdout
sys.stdout = fct.Output(path=PATH, name=OUTPUT_NAME)







# Data Generating processes pre define
xlow = 0            # lower bound for uniform distr in DGPs 1-3
xup = 1             # upper bound for uniform distr in DGPs 1-3
u_sd = 1            # noise variance
n = 1000            # number of observations
n_sim = 50         # number of simulations
n_red = 5           # number of observations reduced
n_sim_red = 20      # number of simulations reduced
d_0 = 0             # first stage intersection
b_0 = 0             # second stage intersection
d_1 = 0.3           # first stage coeff
b_1 = 1             # second stage coeff
b_2 = 0.8           # second stage coeff
d_2 = 0.4           # first stage coeff     
np.random.seed(1)   # random seed
weak_d = 0.0001     # weak instrument for second DGP
random_n = 2        # set seed


##### Run Simulations

raw_data = fct.simulation(n_sim, d_0, d_1, d_2, b_0, b_1, b_2, weak_d, xlow, xup, u_sd, n,
                          random_n, n_red, n_sim_red)


##### Get variables from raw data

data = fct.split_data(raw_data)

# all gdps
LIML = data[0]
TSLS = data[1]

# splitted gdps
LIML_gpd1 = data[2]
LIML_gpd2 = data[3]
LIML_gpd3 = data[4]

TSLS_gpd1 = data[5]
TSLS_gpd2 = data[6]
TSLS_gpd3 = data[7]



##### Inferences   

# bias
bias_GDP1_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[0][0]
bias_GDP2_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[0][1]
bias_GDP3_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[0][2]

bias_LIML = [bias_GDP1_LIML, bias_GDP2_LIML, bias_GDP3_LIML]
#bias_LIML

bias_GDP1_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[1][0]
bias_GDP2_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[1][1]
bias_GDP3_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[1][2]

bias_TSLS = [bias_GDP1_TSLS, bias_GDP2_TSLS, bias_GDP3_TSLS]
#bias_TSLS

# variance
variance_GDP1_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[2][0]
variance_GDP2_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[2][1]
variance_GDP3_LIML = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[2][2]

variance_LIML = [variance_GDP1_LIML, variance_GDP2_LIML, variance_GDP3_LIML]
#variance_LIML

variance_GDP1_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[3][0]
variance_GDP2_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[3][1]
variance_GDP3_TSLS = fct.bias_var(LIML,TSLS, b_0, b_1, b_2, d_0, d_1, d_2)[3][2]

variance_TSLS = [variance_GDP1_TSLS, variance_GDP2_TSLS, variance_GDP3_TSLS]
#variance_TSLS

# mean square error
MSE_GDP1_LIML = pow(bias_GDP1_LIML ,2) + variance_GDP1_LIML
MSE_GDP2_LIML = pow(bias_GDP2_LIML ,2) + variance_GDP2_LIML
MSE_GDP3_LIML = pow(bias_GDP3_LIML ,2) + variance_GDP3_LIML

MSE_LIML = [MSE_GDP1_LIML, MSE_GDP2_LIML, MSE_GDP3_LIML]
#MSE_LIML

MSE_GDP1_TSLS = pow(bias_GDP1_TSLS ,2) + variance_GDP1_TSLS
MSE_GDP2_TSLS = pow(bias_GDP2_TSLS ,2) + variance_GDP2_TSLS
MSE_GDP3_TSLS = pow(bias_GDP3_TSLS ,2) + variance_GDP3_TSLS

MSE_TSLS = [MSE_GDP1_TSLS, MSE_GDP2_TSLS, MSE_GDP3_TSLS]
#MSE_TSLS

# standard errors
SE_GDP1_LIML = np.sqrt(n_sim/(n_sim - 1) * variance_GDP1_LIML)
SE_GDP2_LIML = np.sqrt(n_sim/(n_sim - 1) * variance_GDP2_LIML)
SE_GDP3_LIML = np.sqrt(n_sim/(n_sim - 1) * variance_GDP3_LIML)

SE_LIML = [SE_GDP1_LIML,  SE_GDP2_LIML, SE_GDP3_LIML] 
#SE_LIML

SE_GDP1_TSLS = np.sqrt(n_sim/(n_sim - 1) * variance_GDP1_TSLS)
SE_GDP2_TSLS = np.sqrt(n_sim/(n_sim - 1) * variance_GDP2_TSLS)
SE_GDP3_TSLS = np.sqrt(n_sim/(n_sim - 1) * variance_GDP3_TSLS)

SE_TSLS = [SE_GDP1_TSLS, SE_GDP2_TSLS, SE_GDP1_TSLS]
#SE_TSLS

summary_list = [bias_LIML, bias_TSLS, variance_LIML, variance_TSLS, MSE_LIML, MSE_TSLS, SE_LIML, SE_TSLS]
#summary_list
 

##### Summary tables

fct.summary(summary_list)


##### Plots

fct.plot_function1(LIML_gpd1, TSLS_gpd1, b_1, b_2, d_2)
fct.plot_function2(LIML_gpd2, TSLS_gpd2, b_1, b_2, d_2)
fct.plot_function3(LIML_gpd3, TSLS_gpd3, b_1, b_2, d_2)



# close the output file
sys.stdout.output.close()
sys.stdout = orig_stdout

