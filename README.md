# HW-AI
### Forecasting heatwaves with AI

### Overview
A Python code for making AI-based sub-seasonal forecasts of summer heatwaves in central Europe. The machine learning models used are linear and random forest models, both for regression and classification tasks. This code reproduces the data and experiments presented in: E. Weirich Benet, M. Pyrina, B. Jiménez Esteve, E. Fraenkel, J. Cohen, and D. I. V. Domeisen, "*publication year*": Sub-seasonal Prediction of Central European Summer Heatwaves with Linear and Random Forest Machine Learning Models, AIES, AMS, "*issue number, page range*", DOI: "*XYZ*" [1]

### Dependencies 
The code was developed using Python 3.7.10 and several external libraries, which were installed with conda (https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/): <br/> 
xarray 0.16.1 <br/> 
pandas 1.1.3 <br/>
numpy 1.20.2 <br/>
scipy 1.5.4 <br/>
sklearn (or scikit-learn) 0.24.1 <br/>
imblearn (or imbalanced-learn) 0.8.0 <br/>
matplotlib 3.4.1 <br/>
seaborn 0.11.1 <br/>
eofs 1.4.0 <br/>

In addition, the TIGRAMITE toolbox by J. Runge (https://github.com/jakobrunge/tigramite) is used for data visualization purposes. You only need to import it to run the *data_visualization.py* script.  

### Input data 
"*List the data sources and years. Refer to table in paper. Alternative: provide preprocessed data after step 1 directly as a file.*"
The sources of the input data are listed in Table 1 in [1]. 

### Code structure
The code is split up into ten inter-connected Python scripts: <br/>
1. *const.py* defines a dictionary that contains all user-selected settings and constants. You can define the folders where you want to save the output data, plots, and tables here.
2. Run *preprocessing_part1.py* to preprocess the raw data in time and space. Its output is a dataset containing the time series of the predictors and targets, which is saved to a local file. 
3. Run *data_visualization.py* to visualize some characteristics of the output of *preprocessing_part1.py* (e.g., the lagged correlations between the predictors and the target and the class imbalance of the heat wave indices).
4. Run *executer.py* to obtain the final prediction in form of plots and tables, which are displayed and saved to your chosen folder. This scripts takes the output of *preprocessing_part1.py* as its input.
5. The six remaining scripts (*preprocessing_part2.py*, *prediction.py*, *reference_forecasts.py*, *metrics.py*, *plotting.py*, and *utils.py*) contain only function definitions. These functions are called by the scripts in 1-4. Running any of these six scripts on its own does not produce any output.

### User agreement and license 
By downloading HW-AI you agree with the following points: <br/>
1. You commit to cite HW-AI in your reports or publications if used as indicated above [1]. <br/>
2. The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application the code. See the MIT License at license.txt for the full text. <br/>

