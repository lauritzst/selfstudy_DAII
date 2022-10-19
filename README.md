# General information

This project was submitted as a personal assignment for the "Data Analytics II: Causal Econometrics" course.

## Course task

### Task
The topic was to set up a simulation study that compares two estimators. The criteria were to find 
 
1. a research design, (e.g., Experiment, Conditional Independence Assumption, Diff- in-Diff, ...) and parameter of interest (e.g. ATE, ATET, LATE, CATE) 
2. two estimators suitable for this problem (referring to the lecture notes which are not uploaded on my profile) 
3. three data generating processes (DGP): 
 -   DGP1: The identifying assumptions hold and estimator A should perform better than B 
 -  DGP2: The identifying assumptions hold and estimator B should perform better than A 
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
