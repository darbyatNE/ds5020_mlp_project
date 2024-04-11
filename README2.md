__Team: Justine Cyr, Ben Darby, Ryan Webb__<br>
__DS5020, Spring 2024__<br>
__Professor: W Viles__<br>
### Semester Project Paper
[An Overview of ANN/MLP: Analyzing Water Potability Data](https://docs.google.com/document/d/101gRpJYR1gP-H9hRQgJUnVGIw7PT2j6PrdwQfrNMmSM/edit)
# Project Topic: Multilayer Perceptron / Neural Network

* This project will  demonstrate the implementation of a Multi Layer Perceptron (MLP). 
    1. It consists of an input layer of nodes that receive data.
    2. Hidden layer(s) of nodes that evaluate and specify the relationship between inputs and outputs.
    3. An output layer which provides approximated results using linear algebra.

* This project is provided to showcase a method designed for classification, recognition, prediction and approximation.

* To get started, ensure you are in the ds5020_mlp_project directory.
    1. into the terminal, type 'Make lin_reg', 'Make log_reg' and 'Make mlp'
    2. a corresponding figure(s) for each regression will generate under the 'fig' tab. Check them out.
        The linear regression figure plots the line of best fit between the height of mothers and their daughters. The line shows a correlation between taller mothers generally having taller daughters, and shorter mothers generally having shorter daughters.
        The logistic regression generates two figures: one for the Newton model and one using gradient descent. They plot the change in vbeta, and demonstrate that in both models vbeta stabilizes. Vbeta stabilizes much more quickly using gradient descent.
        Finally, the mlp figure plots the error between the output of the mlp and the expected value. It shows the error for the training runs decreases as the mlp 'training' progresses. therefore, the mlp is successfully improving its performance over the training runs. 


* Repo Directories:    
    - data: contains the data files for the regression/mlp models
    - src: contains the source code for the regression/mlp models

  
# Bibliography
   [Multilayer Perceptron and Neural Networks](https://darbyatne.github.io/Multilayer_perceptron_and_neural_networks.pdf)

   [Introduction to Artificial Neural Networks](https://www.ijeit.com/vol%202/Issue%201/IJEIT1412201207_36.pdf)

   [EEG signals classification using the K-means clustering and a multilayer perceptron neural network model](https://www.sciencedirect.com/science/article/abs/pii/S0957417411006762)

   [Deep learning using multilayer perception improves the diagnostic acumen of spirometry: a single-centre Canadian study](https://pubmed.ncbi.nlm.nih.gov/36572484/)

   [A survey on neural networks for (cyber-) security and (cyber-) security of neural networks](https://www.sciencedirect.com/science/article/pii/S0925231222007184)

   [An intelligent bankruptcy prediction model using a multilayer perceptron](https://www.sciencedirect.com/science/article/pii/S2667305322000734)