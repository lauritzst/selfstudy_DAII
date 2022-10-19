# General information

This project was submitted as a personal assignment for the "Data Analytics II: Causal Econometrics" course. The project was graded with 23/25 points, which corresponded to 30% of the final grade.

## Course task

### Task
The topic was to set up a simulation study that compares two estimators. The criteria were to find 
 
1. a research design, (e.g., Experiment, Conditional Independence Assumption, Diff- in-Diff, ...) and parameter of interest (e.g. ATE, ATET, LATE, CATE) 
2. two estimators suitable for this problem (referring to the lecture notes which are not uploaded on my profile) 
3. three data generating processes (DGP): 
 - DGP1: The identifying assumptions hold and estimator A should perform better than B 
 - DGP2: The identifying assumptions hold and estimator B should perform better than A 
 - DGP3: One identifying assumption is violated
4. performance measures, e.g., mean-squared error, bias, variance, coverage rate, etc.

### Rules

- Flexible in the setting and estimators you choose
- Only restriction: Not a combination of mean, OLS and IPW 
- Implementation must be done in Python
- Implement the estimators from scratch (no library functions)
- You may use libraries as inputs to your functions, e.g. Lasso implementation within a new Causal ML function
- Implement DGPs from scratch (no OPPOSUM or alike) Projects that are (almost) identical will receive zero points

### Evaluation

Total of 25 points

1. 5 points for documentation: present your setting and results in a clear and
compact manner
2. 10 points for DGPs and estimators:
 - Choose at least one advanced estimator (not only mean comparison/OLS/IPW)
 - Show that you understand the estimators and thus are able to motivate interesting
(and in the ideal case simple) DGPs where their performance should diverge in DGP1-2 and break down in DGP3 (whether they actually do is part of the results)
3. 10 points for the programming: Does the code run?, Is it well commented?, Is the coding efficient (use of functions)?, ...

## Project Description

In this framework I chose the so-called local average treatment effect (LATE) in an instrument variable (IV) setting. While the IV method fails to capture either the Average Treatment Effect (ATE) or Average Treatment Effect on the Treated (ATET), because the method measures effects on subgroups, it is a common strategy to identify the LATE, which represents the treatment effect of a specific sub-sample.
The estimators under consideration were the Two-Stage Least Squares (2SLS) estimator and the Limited Information Maximum Likelihood (LIML) estimator. For a more detailed explanation please see self_study_Storch.pdf

## Code and other files

The function_file.py contains all necessary functions to calculate/generate the data-generating processes (DGPs), estimators, and plots.
The main_file.py calls and executes all necessary functions on predefined variables. In addition, inference statistics are also calculated in this file. All tables are saved in the self_study_output.txt file. The self_study_Storch.pdf explains the whole estimation procedure and gives reasoning for the results obtained in the three different DGP simulations.
