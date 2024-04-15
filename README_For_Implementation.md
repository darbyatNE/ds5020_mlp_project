__Team: Justine Cyr, Ben Darby, Ryan Webb__<br>
__DS5020, Spring 2024__<br>
__Professor: W Viles__<br>
### implementation

* This repository houses three type of models which are used for classification, prediction and approximation tasks.  They are linear regression, logistic regression and multi-layer perceptron.  The python scripts import data and process it to produce final parameters,  These parameters could then be used to make predictions with a new set of data inputs.  By running the scripts you will be generating pngs whose output confirms that as these models work through their iterations they are improving in their ability to perform the tasks assigned.  

* To get started, ensure you are in the ds5020_mlp_project directory.
    1. into the terminal, type 'make lin_reg', 'make log_reg' and 'make mlp'
    2. a corresponding figure(s) for each model type will generate under the 'figs' tab. Check them out.
        The linear regression figure plots the line of best fit between the height of mothers and their daughters. The line shows a correlation between taller mothers generally having taller daughters, and shorter mothers generally having shorter daughters.
        The logistic regression generates two figures: one for the Newton model and one using gradient descent. They plot the change in vbeta, and demonstrate that in both models vbeta stabilizes. Vbeta stabilizes much more quickly using gradient descent.
        Finally, the mlp figure plots the error between the output of the mlp and the expected value. It shows the error for the training runs decreases as the mlp 'training' progresses. therefore, the mlp is successfully improving its performance over the training runs. 


* Repo Directories:    
    - data: contains the data files for the regression/mlp models
    - src: contains the source code for the regression/mlp models

 
