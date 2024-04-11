__Team: Justine Cyr, Ben Darby, Ryan Webb__<br>
__DS5020, Spring 2024__<br>
__Professor: W Viles__<br>
### implementation

* This project is provided to showcase a method designed for classification, recognition, prediction and approximation.

* To get started, ensure you are in the ds5020_mlp_project directory.
    1. into the terminal, type 'Make lin_reg', 'Make log_reg' and 'Make mlp'
    2. a corresponding figure(s) for each model type will generate under the 'figs' tab. Check them out.
        The linear regression figure plots the line of best fit between the height of mothers and their daughters. The line shows a correlation between taller mothers generally having taller daughters, and shorter mothers generally having shorter daughters.
        The logistic regression generates two figures: one for the Newton model and one using gradient descent. They plot the change in vbeta, and demonstrate that in both models vbeta stabilizes. Vbeta stabilizes much more quickly using gradient descent.
        Finally, the mlp figure plots the error between the output of the mlp and the expected value. It shows the error for the training runs decreases as the mlp 'training' progresses. therefore, the mlp is successfully improving its performance over the training runs. 


* Repo Directories:    
    - data: contains the data files for the regression/mlp models
    - src: contains the source code for the regression/mlp models

 
