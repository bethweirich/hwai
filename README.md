# HWAI
### Heatwave forecasting with Artificial Intelligence

### Overview
A Python code for making AI-based sub-seasonal forecasts of summer heatwaves in central Europe. The machine learning models used are linear and random forest models, both for regression and classification tasks. This code reproduces the data and experiments presented in the publication: Weirich-Benet, E., M. Pyrina, B. Jiménez-Esteve, E. Fraenkel, J. Cohen, and D. I. V. Domeisen, 2023: Subseasonal Prediction of Central European Summer Heatwaves with Linear and Random Forest Machine Learning Models. Artif. Intell. Earth Syst., 2, e220038, https://doi.org/10.1175/AIES-D-22-0038.1.

The tree structure of the directories in *hwai/* is shown in *directory_tree*. 

### Dependencies 
The code was developed using Python 3.7.10 and several external libraries, which were installed with conda (https://docs.anaconda.com/anaconda/user-guide/tasks/install-packages/): <br/> 
xarray 0.16.1 <br/> 
pandas 1.1.3 <br/>
numpy 1.20.2 <br/>
scipy 1.5.3 <br/>
sklearn (or scikit-learn) 0.24.1 <br/>
imblearn (or imbalanced-learn) 0.8.0 <br/>
matplotlib 3.4.1 <br/>
seaborn 0.11.1 <br/>
eofs 1.4.0 <br/>

In addition, the TIGRAMITE toolbox by J. Runge (https://github.com/jakobrunge/tigramite) is used for data visualization purposes. You only need to import it to run the *data_visualization.py* script.  

### Input data 
The sources of the input data are listed in Table 1 in [1]. The raw data is not provided in this GitHub repository due to their large size (about 5GB). The pre-processed data (output from *preprocessing_part1.py*) and the ECMWF hindcasts are provided in the *hwai/data* folder, which allows the user to run all scripts (except for *preprocessing_part1.py*) without downloading any data. To run *preprocessing_part1.py*, the user must download the raw data and specify the path to the raw data files in *const.py*, under "path_raw_data".

### Code structure
The code is split up into ten inter-connected Python scripts, located in the *python_scripts* folder: <br/>
1. *const.py* defines a dictionary that contains all user-selected settings and constants. You can define the folders where you want to save the output data, plots, and tables here, as well as the path to the raw data.<br/>  
2. Run *preprocessing_part1.py* to preprocess the raw data (files not provided) in time and space. Its outputs are the time series of the predictors, targets, and climatology, which are saved to local files and provided in the *hwai/data* folder as: *preprocessed_predictors_7D_rolling_mean.nc*, *preprocessed_predictors_7D_simple_mean.nc*, *targets_7D_rolling_mean.nc*, *targets_7D_simple_mean.nc*, and *climatology.nc*. <br/> 
3. Run *data_visualization.py* to visualize some characteristics of the output of *preprocessing_part1.py*: the time series of the predictors, lagged correlations between the target and predictors, data histogram, class imbalance of the heat wave indices, and latitude-longitude boxes used to select the predictors).<br/> 
4. Run *executer.py* to obtain the final prediction in form of plots, tables, hyper-parameters, and time series, which are saved to *hwai/model_output*. This scripts takes the output of *preprocessing_part1.py* as its input.<br/> 
5. The six remaining scripts (*preprocessing_part2.py*, *prediction.py*, *reference_forecasts.py*, *metrics.py*, *plotting.py*, and *utils.py*) contain only function definitions. These functions are called by the scripts in 1-4. Running any of these six scripts on its own does not produce any output.
  5a) *preprocessing_part2.py* contains all data preprocessing steps that must be run after the prediction lead time is fixed (e.g., creating a dataset of lagged predictors for each lead time).
  5b) *prediction.py* contains the machine learning algorithms' training, validation and testing steps. It also calls the functions in *preprocessing_part2.py* in order. 
  5c) *reference_forecasts.py* computes three reference forecasts which are used as a baseline to assess the predictive power of the machine learning algorithms: persistence, climatology, and the ECMWF forecast. 
  5d) *metrics.py* contains all functions related to the evaluation of the machine learning and reference forecasts, comparing them to the ground truth. For regression, RMSE and Pearson correlation are calculated. Similarly, ROC AUC and the Brier score (BS) are calculated for the probabilistic classification. For the binary classification, the confusion matrix, the TPR and FPR, the frequency bias (B), and the extremal dependence index (EDI) are calculated (see Section 2e in [1]). 
  5e) *plotting.py* is used to plot all figures that appear in [1] and some additional ones (e.g., ROC, PR and reliability curves).
  5f) *utils.py* is a toolbox containing short functions. 
  
### Sample runs
The outputs from *executer.py* used for the main results (*sample_run-noCV.pdf*) and the nested cross-validation results shown in the Appendix (*sample_run-nestedCV.pdf*) are in the *sample_runs* folder. Due to their large size (about 35MB), they cannot be visualized on GitHub and must be downloaded. 

### User agreement and license 
By downloading HWAI you agree with the following points: <br/>
1. You commit to cite HWAI in your reports or publications if used as indicated above [1]. <br/>
2. The code is provided without any warranty or conditions of any kind. We assume no responsibility for errors or omissions in the results and interpretations following from application the code. See the MIT License at license.txt for the full text. <br/>

