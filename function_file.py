#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 17:49:13 2022

@author: lauritzstorch
@title: Self_Study_Data_Analytics_II

@functions file
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import eigh


# define an Output class for simultaneous console - file output
class Output():
    """Output class for simultaneous console/file output."""

    def __init__(self, path, name):

        self.terminal = sys.stdout
        self.output = open(path + name + ".txt", "w")

    def write(self, message):
        """Write both into terminal and file."""
        self.terminal.write(message)
        self.output.write(message)

    def flush(self):
        """Python 3 compatibility."""

# Data Generating processes pre define
def dgp1(d_0, d_1, b_0, b_1, b_2, d_2, xlow,xup,u_sd,n, Figureshow = "no", hetero = "no"):
    """
    Parameters
    ----------
    d_0 : TYPE: int
        DESCRIPTION. First stage intersection
    d_1 : TYPE: int
        DESCRIPTION. First stage coefficient
    b_0 : TYPE: int
        DESCRIPTION. Second stage intersection
    b_1 : TYPE: int
        DESCRIPTION. Second stage coefficient
    b_2 : TYPE: int
        DESCRIPTION. Second stage coefficient
    xlow : TYPE: int
        DESCRIPTION. Lower bound for uniformly distributed random variables
    xup : TYPE: int
        DESCRIPTION. Upper bound for uniformly distributed random variables
    u_sd : TYPE: int
        DESCRIPTION. Variance of normal distributed variables
    n : TYPE: int
        DESCRIPTION. Number of observations
    Figureshow : TYPE: plot, optional
        DESCRIPTION. Plots generated data. The default is "no".
    hetero : TYPE, optional
        DESCRIPTION. Determines if DGP is heteroscedastic or homoscedastic The default is "yes".

    Returns
    -------
    Data Generating process (heteroscedastic)

    """
    # Set numpy seed to reproduce same random data as before
    #np.random.seed(1)
    
    # covariate x_1 uniformly distributed
    x_1 = np.random.uniform(xlow,xup,n)
    
    # IV z_1 uniformly distributed
    z_1 = np.random.uniform(xlow,xup,n)
    
    if hetero == "yes":
        # X data dependent on Z: (heteroscedatstic)   
        p = pow(np.random.uniform(xlow,xup,n), 3)           # Scale factor for heteroscedasticity
        v = np.random.normal(0, 0.25+p*u_sd,n)              # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v                 # Dependent variable
    else:
        # X data dependent on Z: (homoscedatstic)
        p = sum(np.random.uniform(xlow,xup,n), 3)/n         # Scale factor
        v = np.random.normal(0, 0.25+p*u_sd,n)              # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v                 # Dependent variable
    
    # Y data dependent on X
    u = np.random.normal(0,u_sd,n)
    y = b_0 + b_1 * d + b_2 * x_1 + u
    
    # Plot v against z_1
    if Figureshow == "yes":
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        
        plt.title("d against z_1")
        plt.scatter(z_1,d,color="red")
        #plt.scatter(d,y,color="red")
    
    #print("First array: z_1, Second array: d, Third array: x_1, Fourth array: y, Fifth array: v")
    return(z_1,d,x_1,y,v)


# DGP 2: 2SLS < LIML
def dgp2(d_0, d_1, b_0, b_1, b_2, d_2, xlow,xup,u_sd,n, weak_d, Figureshow = "no", hetero = "no"):
    """
    Parameters
    ----------
    d_0 : TYPE: int
        DESCRIPTION. First stage intersection
    d_1 : TYPE: int
        DESCRIPTION. First stage coefficient
    b_0 : TYPE: int
        DESCRIPTION. Second stage intersection
    b_1 : TYPE: int
        DESCRIPTION. Second stage coefficient
    b_2 : TYPE: int
        DESCRIPTION. Second stage coefficient
    xlow : TYPE: int
        DESCRIPTION. Lower bound for uniformly distributed random variables
    xup : TYPE: int
        DESCRIPTION. Upper bound for uniformly distributed random variables
    u_sd : TYPE: int
        DESCRIPTION. Variance of normal distributed variables
    n : TYPE: int
        DESCRIPTION. Number of observations
    weak_d : TYPE: int
        DESCRIPTION. Second stage coefficient (weak instruments)
    Figureshow : TYPE: plot, optional
        DESCRIPTION. Plots generated data. The default is "no".
    hetero : TYPE, optional
        DESCRIPTION. Determines if DGP is heteroscedastic or homoscedastic The default is "yes".

    Returns
    -------
    Data Generating process (weak instruments)

    """
    # Set numpy seed to reproduce same random data as before
    #np.random.seed(1)
    
    # covariate x_1 uniformly distributed
    x_1 = np.random.uniform(xlow,xup,n)
    
    # IV z_1 uniformly distributed
    z_1 = np.random.uniform(xlow,xup,n)
    
    # create weak IVs z_2 - z_20
    z_IV = []
    for i in range(19):
        z_IV.append(np.random.uniform(xlow,xup,n))
    
    IV_z = [z_1] + z_IV                                 # Save all IVs in list
    if hetero == "yes":
        # X data dependent on Z: (heteroscedatstic)
        p = pow(np.random.uniform(xlow,xup,n), 3)       # Scale factor
        v = np.random.normal(0, 0.25+p*u_sd,n)          # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v             # Dependent variable
        for i in range(19):
            d += np.multiply(weak_d, z_IV[i])           # Loop adds weak instruments to X
    else:
        # X data dependent on Z: (homoscedatstic)
        p = sum(pow(np.random.uniform(xlow,xup,n), 3))/n    # Scale factor
        v = np.random.normal(0, 0.25+p*u_sd,n)          # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v             # Dependent variable
        for i in range(19):
            d += np.multiply(weak_d, z_IV[i])           # Loop adds weak instruments to X
    
    # Y data dependent on X
    y = b_0 + b_1 * d + b_2 * x_1 + np.random.normal(0,u_sd,n) #  Second stage


    # Plot z_1 against x 
    if Figureshow == "yes":
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        
        plt.title("Plot z_1 against x")
        plt.scatter(z_1,d,color="red")
    
    #print("First array: IV_z, Second array: d, Third array: x_1, Fourth array: y, Fifth array: v")
    return(IV_z,d,x_1,y,v)


def dgp3(d_0, d_1, b_0, b_1, b_2, d_2, xlow,xup,u_sd,n, Figureshow = "no", hetero = "no"):
    """
    Parameters
    ----------
    d_0 : TYPE: int
        DESCRIPTION. First stage intersection
    d_1 : TYPE: int
        DESCRIPTION. First stage coefficient
    b_0 : TYPE: int
        DESCRIPTION. Second stage intersection
    b_1 : TYPE: int
        DESCRIPTION. Second stage coefficient
    b_2 : TYPE: int
        DESCRIPTION. Second stage coefficient
    xlow : TYPE: int
        DESCRIPTION. Lower bound for uniformly distributed random variables
    xup : TYPE: int
        DESCRIPTION. Upper bound for uniformly distributed random variables
    u_sd : TYPE: int
        DESCRIPTION. Variance of normal distributed variables
    n : TYPE: int
        DESCRIPTION. Number of observations
    weak_d : TYPE: int
        DESCRIPTION. Second stage coefficient (weak instruments)
    Figureshow : TYPE: plot, optional
        DESCRIPTION. Plots generated data. The default is "no".
    hetero : TYPE, optional
        DESCRIPTION. Determines if DGP is heteroscedastic or homoscedastic The default is "yes".

    Returns
    -------
    Data Generating process (identifying assumption violated)

    """
    # Set numpy seed to reproduce same random data as before
    #np.random.seed(1)
    
    # covariate x_1 uniformly distributed
    x_1 = np.random.uniform(xlow,xup,n)
    
    # IV z_1 uniformly distributed
    z_1 = np.random.uniform(xlow,xup,n)
    
    if hetero == "yes":
        # X data dependent on Z: (heteroscedatstic)
        p = pow(np.random.uniform(xlow,xup,n), 3)           # Scale factor
        v = np.random.normal(0, 0.25+p*u_sd,n)              # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v                 # Dependent variable
    else:
        # X data dependent on Z: (homoscedatstic)
        p = sum(pow(np.random.uniform(xlow,xup,n), 3))/n    # Scale factor
        v = np.random.normal(0, 0.25+p*u_sd,n)              # Noise term
        d = d_0 + d_1 * z_1 + d_2 * x_1 + v                 # Dependent variable
    
    # Y data dependent on X
    u = z_1 * np.random.normal(0,u_sd,n)
    y = b_0 + b_1 * d * pow(z_1,1) + b_2 * x_1 + u          #  Second stage
    
    # Plot v against z_1
    if Figureshow == "yes":
        
        plt.rcParams["figure.figsize"] = [7.50, 3.50]
        plt.rcParams["figure.autolayout"] = True
        
        plt.title("Errors")
        plt.scatter(y,d,color="red")
    
    #print("First array: z_1, Second array: d, Third array: x_1, Fourth array: y, Fifth array: v")
    return(z_1,d, x_1,y,v)


def LIML(z, d, x, y, sample_size, Check_eigenvalues = "no"):  
    """
    Parameters
    ----------
    z : TYPE: numpy.ndarray
        DESCRIPTION. Instrument variables
    d : TYPE: numpy.ndarray
        DESCRIPTION. Dependent variables
    x : TYPE: numpy.ndarray
        DESCRIPTION. Exogenous covariates
    y : TYPE: numpy.ndarray
        DESCRIPTION. Endogenous variable
    sample_size : TYPE: int
    Check_eigenvalues : optional
        DESCRIPTION. Check whether generalized eigenvalue calculations are correct. The default is "no".

    Returns
    -------
    beta_LIML
        DESCRIPTION. Limited information Maximum Likelihood estimator

    """    
    # SRR of first stage
    if len(z) == sample_size:
        Z_vec = pd.DataFrame(data = z)
    else:
        Z_vec = []
        for i in range(len(z)):
            Z_vec.append(pd.DataFrame(z[i]))
        Z_vec = pd.concat(Z_vec, axis = 1)

    ones = pd.DataFrame(np.ones(len(y)))
    z_dat = pd.DataFrame(pd.concat([ones, Z_vec], axis = 1))     # vector of all instruments
    
    proj_z = z_dat @ np.linalg.inv(z_dat.T @ z_dat).dot(z_dat.T) # projection matrix of all instruments
    endog = pd.DataFrame( data = [y,d]).T                        # endogenous variable vector (y and x (dependent)) 

    e_f = endog - proj_z @ endog                                 
    SSR_f = e_f.T @ e_f                 # SSR of first stage
    
    # SSR of second stage
    x = pd.DataFrame(x) 
    z_dot = pd.concat([ones, x, Z_vec], axis = 1)                # vector of all instruments and covariates

    z_dot = pd.DataFrame(data = z_dot)
    proj_zx = z_dot @ np.linalg.inv(z_dot.T @ z_dot).dot(z_dot.T)# projection matrix of all instruments + covariates
    
    e_s = endog - proj_zx @ endog
    SSR_s = e_s.T @ e_s                 # SSR of second stage
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = eigh(SSR_s, SSR_f, eigvals_only=False)    # eigenvalues, eigenvectors
    
    k_estimator = min(eigvals)                      # smallest eigenvalue (solves LIML estimation problem)
    if Check_eigenvalues == "yes":                  # Check whether calculation is correct
        test_1 = round(SSR_f @ eigvecs * eigvals,6) 
        test_2 = round(SSR_s @ eigvecs, 6)
        
    # Further transformationa
    M_z = np.identity(len(proj_z)) - proj_z         
    inner = np.identity(len(M_z)) - k_estimator * M_z
    prod_up = z_dot.T @ inner @ y
    prod_low = np.linalg.inv(z_dot.T @ inner @ z_dot)
    
    # LIML estimator
    beta_LIML = pd.DataFrame(prod_low @prod_up)
    
    return beta_LIML#, k_estimator



def tsls(z,d,x,y, sample_size):
    """
    Parameters
    ----------
    z : TYPE: numpy.ndarray
        DESCRIPTION. Instrument variables
    d : TYPE: numpy.ndarray
        DESCRIPTION. Dependent variables
    x : TYPE: numpy.ndarray
        DESCRIPTION. Exogenous covariates
    y : TYPE: numpy.ndarray
        DESCRIPTION. Endogenous variable
    sample_size : TYPE: int

    Returns
    -------
    deltas: 
        DESCRIPTION. Two-Staged least squared estimator

    """    
    n = d.shape[0]                       # num of obs
    if len(z) != sample_size:            # several covariates
        z_dat = np.c_[np.ones(n),x,z[0]] # add constant
        for i in range(len(z)-1):
            z_dat = np.c_[z_dat,z[1+i]]  # add constant
        betas = np.linalg.inv(z_dat.T @ z_dat) @ z_dat.T @ d # calculate coeff           
    else:
        z_dat = np.c_[np.ones(n),x, z]   # add constant
        betas = np.linalg.inv(z_dat.T @ z_dat) @ z_dat.T @ d # calculate coeff
    
    
    x_dat = np.c_[np.ones(n),x,d] 
    gammas = np.linalg.inv(x_dat.T @ x_dat) @ x_dat.T @ y    # calculate coeff
    
    # Effect of IV on y
    # Reduced form calculations
    if len(z) != sample_size:
        delta_0 = gammas[0] + gammas[2] * betas[0]
        delta_1 = gammas[1] + gammas[2] * betas[1]
        deltas = np.array(delta_0)
        deltas = np.c_[deltas, delta_1]
        for i in range(len(z_dat.T)-2):
            delta_next = gammas[2] * betas[2+i]
            deltas = np.c_[deltas, delta_next]   
        deltas = pd.DataFrame(deltas.T)
    else: 
        delta_0 = gammas[0] + gammas[2] * betas[0]
        delta_1 = gammas[1] + gammas[2] * betas[1]
        delta_2 = gammas[2] * betas[2]
        deltas = np.array(delta_0)
        deltas = np.c_[deltas, delta_1, delta_2]
        deltas = pd.DataFrame(deltas.T)
    
    # print("First array: betas, Second array: gammas, Third array: deltas")
    return deltas



def simulation(n_sim, d_0, d_1, d_2, b_0, b_1, b_2, weak_d, xlow, xup, u_sd, n, random_n, n_red, n_sim_red):
    """
    Parameters
    ----------
    n_sim : TYPE: int
        DESCRIPTION. number of simulations
    d_0 : TYPE: float
        DESCRIPTION. intersect first stage
    d_1 : TYPE: float
        DESCRIPTION. IV coeff (first stage)
    d_2 : TYPE: float
        DESCRIPTION. covariates coeff first stage
    b_0 : TYPE: float
        DESCRIPTION. intersect second stage
    b_1 : TYPE: float
        DESCRIPTION. dependent variable coeff second stage
    b_2 : TYPE: float
        DESCRIPTION. covariate coeff second stage
    weak_d : TYPE:float
        DESCRIPTION. weak IV coeff (first stage)
    xlow : TYPE: int
        DESCRIPTION. lower bound uniform distribution
    xup : TYPE: int
        DESCRIPTION. upper bound uniform distribution
    u_sd : TYPE: float
        DESCRIPTION. 
    n : TYPE: int
        DESCRIPTION. number of observations

    Returns
    -------
    results_LIML : TYPE: list
        DESCRIPTION. Limited information maximum likelihood estimators for all simulations
    results_tsls : TYPE: list
        DESCRIPTION. Two-staged least sqaure estimators for all simulations

    """
    #all_results = np.empty( (n_sim, 2, 3) )  # initialize for results
    np.random.seed(random_n)
    np.random.seed(1)
    results_LIML = [[],[],[]]
    results_tsls = [[],[],[]]
    # Loop through many simulations
    
    # DGP1:  Reduced n to 5% of its initial value (no enough observations)
    #n_red = int(abs(0.02 * n))
    for i in range(n_sim_red):
            # Run DGP1
            # Reduced n to 5% of its initial value (no enough observations)
            z_1, d, x_1, y, v = dgp1(d_0, d_1, b_0, b_1, b_2, d_2, xlow, xup, u_sd, n_red)
            
            beta_TSLS_1 = tsls(z_1, d, x_1, y, n_red)
            
            results_tsls[0].append(beta_TSLS_1)

            beta_LIML_1 = LIML(z_1, d, x_1, y, n_red, Check_eigenvalues = "no")
            results_LIML[0].append(beta_LIML_1)
            
    for j in range(n_sim):       
            # Run DGP2
            IV_z, d, x_1, y, v = dgp2(d_0, d_1, b_0, b_1, b_2, d_2, xlow, xup, u_sd, n, weak_d)
            
            beta_TSLS_2 = tsls(IV_z, d, x_1, y, n)
            results_tsls[1].append(beta_TSLS_2)
            
            beta_LIML_2 = LIML(IV_z, d, x_1, y, n, Check_eigenvalues = "no")
            results_LIML[1].append(beta_LIML_2)
            
      
            # Run DGP3
            z_1, d, x_1, y, v = dgp3(d_0, d_1, b_0, b_1, b_2, d_2, xlow, xup, u_sd, n)
            
            beta_TSLS_3 = tsls(z_1, d, x_1, y, n)
            results_tsls[2].append(beta_TSLS_3)

            beta_LIML_3 = LIML(z_1, d, x_1, y, n, Check_eigenvalues = "no")
            results_LIML[2].append(beta_LIML_3)
            
    
    results_LIML[0] = pd.DataFrame(pd.concat(results_LIML[0], axis = 1))
    results_LIML[1] = pd.DataFrame(pd.concat(results_LIML[1], axis = 1))
    results_LIML[2] = pd.DataFrame(pd.concat(results_LIML[2], axis = 1))
    
    results_tsls[0] = pd.DataFrame(pd.concat(results_tsls[0], axis = 1))
    results_tsls[1] = pd.DataFrame(pd.concat(results_tsls[1], axis = 1))
    results_tsls[2] = pd.DataFrame(pd.concat(results_tsls[2], axis = 1))
    
    return results_LIML, results_tsls


def split_data(sim):
    """
    Parameters
    ----------
    sim : pd.DataFrame
        DESCRIPTION. Includes whole data: all estimator data, all DGPs, and so on

    Returns
    -------
    Splits data into subparts

    """
    # Indices:
        # first: LIML or TSLS
        # second: DGP
        # third: order simulations 
        # fourth: explicit value
    LIML_all = sim[0]
    TSLS_all = sim[1]
    LIML_dgp_1 = sim[0][0]
    LIML_dgp_2 = sim[0][1]
    LIML_dgp_3 = sim[0][2]
    TSLS_dgp_1 = sim[1][0]
    TSLS_dgp_2 = sim[1][1]
    TSLS_dgp_3 = sim[1][2] 
    
    print("""
          zero value: only LIML data for all dgps, 
          first value: only TSLS data for all dgps,
          second value: only LIML data for first dgps,
          third value: only LIML data for second dgps,
          fourth value: only LIML data for third dgps,
          fifth value: only TSLS data for first dgps,
          sixth value: only TSLS data for second dgps,
          seventh value: only TSLS data for third dgps
          """)
    return LIML_all, TSLS_all, LIML_dgp_1, LIML_dgp_2, LIML_dgp_3, TSLS_dgp_1, TSLS_dgp_2, TSLS_dgp_3


def bias_var(LIML, TSLS, b_0, b_1, b_2, d_0, d_1, d_2):
    """
    Parameters
    ----------
    LIML : TYPE: list
        DESCRIPTION. LIML estimator of simulations
    TSLS : TYPE: list
        DESCRIPTION. TSLS estimator of simulations

    Returns
    -------
    # Indices:
        # first: LIML or TSLS
        # second: DGP
        # third: order simulation 
        # fourth: explicit value
    
    bias_LIML : TYPE: pandas.core.series.Series
    bias_TSLS : TYPE: pandas.core.series.Series
    variance_LIML : TYPE: pandas.core.series.Series
    variance_TSLS : TYPE: pandas.core.series.Series
    
    """
    # correct parameters
    # order: intersection, covariates, and instrument(s)
    
    # DGP1:
    delta_true_1 = np.array([(b_0 + b_1 * d_0), (b_1 * d_2 + b_2) , (b_1 * d_1)])
    delta_true_1 = [round(num, 4) for num in delta_true_1]
    
    # DGP2:
    delta_true_2 = delta_true_1.copy()
    for i in range(19):
        delta_true_2.append(d_2 * b_2)
    delta_true_2 = [round(num, 4) for num in delta_true_2]
    
    # DGP3:
    delta_true_3 = delta_true_1.copy()
    delta_true_3 = [round(num, 4) for num in delta_true_3]
    
    # transform into one dataframe
    delta_true = [delta_true_1, delta_true_2, delta_true_3]
        
    bias_LIML = []
    bias_TSLS = []
    
    variance_LIML = []
    variance_TSLS = []
    for i in range(3):
        # bias calculation
        bias_LIML.append(np.subtract(np.mean(LIML[i].T), delta_true[i]))
        bias_TSLS.append(np.subtract(np.mean(TSLS[i].T), delta_true[i]))
        
        # variance calculation
        variance_LIML.append(np.mean(pow(LIML[i].T - np.mean(LIML[i].T),2)))
        variance_TSLS.append(np.mean(pow(TSLS[i].T - np.mean(TSLS[i].T),2)))
    
    return  bias_LIML, bias_TSLS, variance_LIML, variance_TSLS


def summary(lst):
    """
    Parameters
    ----------
    lst : TYPE: list
        DESCRIPTION. Contains all inferences

    Returns
    -------
    Table with inference statistics

    """
    pd.set_option('display.max_columns', None)
    # Transformation split
    DGP_1_results = []
    DGP_2_results = []
    DGP_3_results = []
    for i in range(len(lst)):
        DGP_1_results.append(lst[i][0])
        DGP_2_results.append(lst[i][1])
        DGP_3_results.append(lst[i][2])
    
    # Transform input
    DGP_1_results = pd.concat(DGP_1_results, axis = 1)
    # Rename columns
    DGP_1_results.rename(columns = {0 : "bias LIML", 1 : "bias TSLS",
                                    2 : "var LIML", 3 : "var TSLS", 
                                    4 : "mse LIML", 5 : "mse TSLS", 
                                    6 : "se LIML", 7 : "se TSLS"},
                                     index = {0 : "beta 0",
                                              1 : "beta 1", 2 : "beta 2"},
                                     inplace = True)

    # Transform input
    DGP_2_results = pd.concat(DGP_2_results, axis = 1)
    # Rename columns
    DGP_2_results.rename(columns = {0 : "bias LIML", 1 : "bias TSLS",
                                    2 : "var LIML", 3 : "var TSLS", 
                                    4 : "mse LIML", 5 : "mse TSLS", 
                                    6 : "se LIML", 7 : "se TSLS"},
                                     index = {0 : "beta 0",
                                              1 : "beta 1", 2 : "beta 2",
                                              3 : "beta 3", 4 : "beta 4",
                                              5 : "beta 5", 6 : "beta 6",
                                              7 : "beta 7", 8 : "beta 8",
                                              9 : "beta 9", 10 : "beta 10",
                                              11 : "beta 11", 12 : "beta 12",
                                              13 : "beta 13", 14 : "beta 14",
                                              15 : "beta 15", 16 : "beta 16",
                                              17 : "beta 17", 18 : "beta 18",
                                              19 : "beta 19", 20 : "beta 20",
                                              21: "beta 21"},
                                      inplace = True)
    # Transform input
    DGP_3_results = pd.concat(DGP_3_results, axis = 1)
    # Rename columns
    DGP_3_results.rename(columns = {0 : "bias LIML", 1 : "bias TSLS",
                                    2 : "var LIML", 3 : "var TSLS", 
                                    4 : "mse LIML", 5 : "mse TSLS", 
                                    6 : "se LIML", 7 : "se TSLS"}, 
                                     index = {0 : "beta 0",
                                              1 : "beta 1", 2 : "beta 2"},
                                      inplace = True)
    
    # Output: table
    table1 = print('Data Generating Process (DGP) 1 Estimation Results:', '-' * 89, 
                   round(DGP_1_results, 5), '-' * 89, '\n\n', sep='\n')
    
    table2 = print('Data Generating Process (DGP) 1 Estimation Results:', '-' * 89, 
                   round(DGP_2_results, 5), '-' * 89, '\n\n', sep='\n') 
    
    table3 = print('Data Generating Process (DGP) 1 Estimation Results:', '-' * 89, 
                   round(DGP_3_results, 5), '-' * 89, '\n\n', sep='\n') 
     
    

def plot_function1(LIML, TSLS, b_1, b_2, d_2):
    """
    Parameters
    ----------
    LIML : TYPE: pandas.core.series.Series
        DESCRIPTION.
    TSLS : TYPE: pandas.core.series.Series
        DESCRIPTION.
    b_1 : TYPE: float
    b_2 : TYPE: float
    d_2 : TYPE: float

    Returns
    -------
    Plots histogram of estimators around true value

    """
    LIML = LIML.T[1] # LATE LIML
    TSLS = TSLS.T[1] # LATE TSLS
    true_parameter = np.array((b_1 * d_2 + b_2))
 
    plt.figure()
    plt.hist(LIML, bins=20, color='red',alpha=0.5,label="LIML GDP1")
    plt.hist(TSLS, bins=20, color='blue',alpha=0.5,label="TSLS GPD2")
    plt.axvline(true_parameter,label="true LATE")
    plt.legend(loc='upper right')
    plt.title('Performance of the two estimators under DGP1')
    plt.show()
    
def plot_function2(LIML, TSLS, b_1, b_2, d_2):
    """
    Parameters
    ----------
    LIML : TYPE: pandas.core.series.Series
        DESCRIPTION.
    TSLS : TYPE: pandas.core.series.Series
        DESCRIPTION.
    b_1 : TYPE: float
    b_2 : TYPE: float
    d_2 : TYPE: float

    Returns
    -------
    Plots histogram of estimators around true value

    """
    LIML = LIML.T[1] # LATE LIML
    TSLS = TSLS.T[1] # LATE TSLS
    true_parameter = np.array((b_1 * d_2 + b_2))
 
    plt.figure()
    plt.hist(LIML, bins=20, color='red',alpha=0.5,label="LIML GDP1")
    plt.hist(TSLS, bins=20, color='blue',alpha=0.5,label="TSLS GPD2")
    plt.axvline(true_parameter,label="true LATE")
    plt.legend(loc='upper right')
    plt.title('Performance of the two estimators under DGP2')
    plt.show()
    
def plot_function3(LIML, TSLS, b_1, b_2, d_2):
    """
    Parameters
    ----------
    LIML : TYPE: pandas.core.series.Series
        DESCRIPTION.
    TSLS : TYPE: pandas.core.series.Series
        DESCRIPTION.
    b_1 : TYPE: float
    b_2 : TYPE: float
    d_2 : TYPE: float

    Returns
    -------
    Plots histogram of estimators around true value

    """
    LIML = LIML.T[1] # LATE LIML
    TSLS = TSLS.T[1] # LATE TSLS
    true_parameter = np.array((b_1 * d_2 + b_2))
 
    plt.figure()
    plt.hist(LIML, bins=20, color='red',alpha=0.5,label="LIML GDP1")
    plt.hist(TSLS, bins=20, color='blue',alpha=0.5,label="TSLS GPD2")
    plt.axvline(true_parameter,label="true LATE")
    plt.legend(loc='upper right')
    plt.title('Performance of the two estimators under DGP3')
    plt.show()
