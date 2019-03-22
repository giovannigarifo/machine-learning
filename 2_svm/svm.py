'''
    ---------------------------
    - Support Vector Machines -
    ---------------------------

    Author
    ------
        Giovanni Garifo, giovanni.garifo@polito.it

    Dataset
    -------
        Iris, imported from scikit-learn
'''

'''
    -------
    Imports
    -------
'''
from mymllib import * # Import functions from self written library
import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from matplotlib import cm
from sklearn import svm
import seaborn as sns



'''
    ----
    Code
    ----
'''
mute_status = '' # mute / unmute the plots and some output


print("\n----------------------------------------\n")
print("- SVM classification over Iris dataset -\n")
print("----------------------------------------\n")

iris_dataset = datasets.load_iris()
X = iris_dataset.data[:,0:2] # get only first two features for each sample
Y = iris_dataset.target # get labels

X_mi = np.mean(X) # mean of X
X_sigma = np.std(X) # standard deviation of X
X = (X-X_mi)/X_sigma # standardize the samples

print("Iris dataset loaded, with only first two dimensions:  X.shape:", X.shape, " Y.shape:", Y.shape, "\n")

# Split the datast into Train, Test and Evaluation sets, , also shuffle the data before splitting
X_train, X_val, X_test, Y_train, Y_val, Y_test = splitDataIntoTVT(X, Y, train_perc = 0.5, val_perc = 0.2, test_perc = 0.3)

print("Shape of Train, Validation and Test sets: ", X_train.shape, X_val.shape, X_test.shape, "\n")

#create hyperparameters list
C_list, Gamma_list = createHyperparametersLists(-3, 4, -3, 4)

'''
-------------- 
- Linear SVM -
--------------

Choose best hyperparameter "C" that gives the highest accuracy when predicting the labels.

The margin width is directly proportional to 1/C, so for value of C >> 1 we get a narrow margin, this means
that our model will behave like a small margin SVM, thus the decisiion boundary will depend only on fewer
points. Instead, if 0 < C < 1, our model will behave like a large margin classifier, thus expanding the margin
such that the decision boundary position will be influenced by more points.

This means that if we choose a small margin, we don't trust that out data are well separated, so it will be
difficult to classify them, and in this case, a small margin will help. But if the margin is too small, it could
be impractical to separate the classes using too few points as support vectors
'''
print("\n---------------\n")
print("- Linear SVM -\n")
print("---------------\n")

# train models and do evaluation step to obtain the value of C that gives highest accuracy
models, accuracy_list, highest_accuracy = applySVM_C(
    'linear', # kernel used
    C_list, # list of C values to use as hyperparameter for modelling
    X_train, Y_train, # train set
    X_val, Y_val, # evaluation set
    mute_status) 

# plot models and their accuracy on evaluation set
if mute_status != MUTE_LINEAR and mute_status != MUTE_ALL :
    plotModels(0, X, Y, models)
    plotAccuracyComparison(1, C_list, accuracy_list, "Tested values of C", "Linear: Accuracy of prediction on evaluation data")

# find best values of C (they can be more then one!), the ones with the highest accuracy
C_best = selectBestC(C_list, accuracy_list, highest_accuracy)
        

#use best value of C to predict on test set
print("\n------------------------------------------------")
print("Performing prediction on Test data with C=", C_best)

# test model
models, accuracy_list, accuracy = applySVM_C(
    'linear', # kernel used
    [C_best], # use only C_best as C value for the list
    X_train, Y_train, # train set
    X_test, Y_test, # test set
    NOT_MUTED) 

# plot model and accuracy for prediction on TEST data
if mute_status != MUTE_LINEAR and mute_status != MUTE_ALL :
    plotModel(2, X, Y, models[0], "Linear: result of prediction over test set with C_best = %.4f" % C_best + "\nAccuracy = %.2f" % accuracy)

'''
---------------------------- 
-  RBF SVM, only C tuning  -
----------------------------

Choose best hyperparameter "C" that gives the highest accuracy when predicting the labels.
'''

print("\n--------------------------\n")
print("- RBF SVM, only C tuning -\n")
print("--------------------------\n")

# train models and do evaluation step to obtain the value of C that gives highest accuracy
models, accuracy_list, highest_accuracy = applySVM_C(
    'rbf', # kernel used
    C_list, # list of C values to use as hyperparameter for modelling
    X_train, Y_train, # train set
    X_val, Y_val, # evaluation set
    mute_status) 

# plot models and their accuracy on evaluation set
if mute_status != MUTE_RBF_C and mute_status != MUTE_ALL:
    plotModels(4, X, Y, models)
    plotAccuracyComparison(5, C_list, accuracy_list, "Tested values of C", "RBF: Accuracy of prediction on evaluation data")

# find best values of C (they can be more then one!), the ones with the highest accuracy
C_best = selectBestC(C_list, accuracy_list, highest_accuracy)

#use best value of C to predict on test set
print("\n------------------------------------------------")
print("Performing prediction on Test data with C=", C_best)

# train models and do evaluation step to obtain the value of C that gives highest accuracy
models, accuracy_list, accuracy = applySVM_C(
    'rbf', # kernel used
    [C_best], # use only C_best as C value for the list
    X_train, Y_train, # train set
    X_test, Y_test, # test set
    NOT_MUTED) 

# plot model and accuracy for prediction on TEST data
if mute_status != MUTE_RBF_C and mute_status != MUTE_ALL :
    plotModel(6, X, Y, models[0], "RBF, C only: result of prediction over test set with C_best = %.4f" % C_best + "\nAccuracy = %.2f" % accuracy)


'''
----------------------------------
-  RBF SVM, C and Gamma tuning   -
----------------------------------

Choose best hyperparameter "C" and "Gamma" that gives the highest accuracy when predicting the labels.
'''

print("\n-------------------------------\n")
print("- RBF SVM, C and Gamma tuning -\n")
print("-------------------------------\n")

# new values for C and Gamma
C_list, Gamma_list = createHyperparametersLists(-4, 6, -4, 6)

# train models and do evaluation step to obtain the values of C and Gamma that gives highest accuracy
models, hyperparameters, highest_accuracy = applySVM_C_Gamma(
    'rbf', # kernel used
    C_list, # list of C values to use as hyperparameter for modelling
    Gamma_list, # list of Gamma values to use as hyperparameter for modelling
    X_train, Y_train, # train set
    X_val, Y_val, # evaluation set
    mute_status) 

# find best values of hyperparameters, the ones that guarantees highest accuracy
C_best, Gamma_best = selectBestHyperparameters(C_list, Gamma_list, hyperparameters, highest_accuracy)

# plot heat map of hyperparamenters
if mute_status != MUTE_RBF_C_GAMMA and mute_status != MUTE_ALL :
    plotHeatmap(8, C_list, Gamma_list, hyperparameters, "Accuracy with C, Gamma tuning")

# test model
models, hyperparameters, accuracy = applySVM_C_Gamma(
    'rbf', # kernel used
    [C_best], 
    [Gamma_best],
    X_train, Y_train, # train set
    X_test, Y_test, # test set
    NOT_MUTED) 

# plot model and accuracy for prediction on TEST data
if mute_status != MUTE_RBF_C_GAMMA and mute_status != MUTE_ALL :
    plotModel(
        9, #fignum 
        X, Y, #dataset
        models[0], 
        "RBF, C and Gamma: result of prediction over test set\nwith C_best = %.6f" % C_best + 
        ", Gamma_best = %.6f" % Gamma_best + "\n Accuracy = %.2f" % accuracy
    )



'''
---------------------------------
-  RBF SVM, K-fold validation   -
---------------------------------

Choose best hyperparameter "C" and "Gamma" that gives the highest accuracy when predicting the labels.

Perform validation using K-fold.
'''

print("\n------------------------------------------------------------\n")
print("- RBF SVM, C and Gamma tuning with K-fold cross-validation -\n")
print("------------------------------------------------------------\n")

# split data only into train and test set, also shuffle the data before splitting
X_train, X_test, Y_train, Y_test = splitDataIntoTVT(X, Y, train_perc = 0.7, val_perc = 0.0, test_perc = 0.3)

# new values for C and Gamma
C_list, Gamma_list = createHyperparametersLists(-4, 6, -4, 6)

# generate K subsets from the training set
K = 5
fold_len = int(len(X_train)/K) #length of each fold
X_folds_list = []
Y_folds_list = []

for i in range(K-1):
    X_folds_list += [X_train[i*fold_len:(i+1)*fold_len]]
    Y_folds_list += [Y_train[i*fold_len:(i+1)*fold_len]]
X_folds_list += [X_train[(K-1)*fold_len:len(X_train)]] # the last remaining fold, take all the remaing samples
Y_folds_list +=[Y_train[(K-1)*fold_len:len(X_train)]]

#For each fold, fit the model on the other (k-1) folds and evaluate on the current fold to gain statistics
hyperparameters_list = []

for k in range(K):

    # list of folds that we'll use as training data
    X_folds_train_list = []
    Y_folds_train_list = []

    #compose training data for the current setep, put together k-1 folds
    for i in range(K):
        if i!=k:
            X_folds_train_list.append(X_folds_list[i])
            Y_folds_train_list.append(Y_folds_list[i])
    
    #convert list to array
    X_folds_train = np.concatenate(X_folds_train_list, axis=0)
    Y_folds_train = np.concatenate(Y_folds_train_list, axis=0)

    #train and evaluate the model to obtain hyperparameters accuracy for this training and evaluation set
    models, hyperparameters, highest_accuracy = applySVM_C_Gamma(
        'rbf', # kernel used
        C_list, # list of C values to use as hyperparameter for modelling
        Gamma_list, # list of Gamma values to use as hyperparameter for modelling
        X_folds_train, Y_folds_train, # train set: all the folds, minus the k-th one
        X_folds_list[k], Y_folds_list[k], # evaluation set: the k-th fold
        mute_status)
    
    #save hyperparameters accuracy for k-th fold
    hyperparameters_list.append(hyperparameters)

# Among all combinations of C and Gamma, compute the average accuracy of the K validation
hyperparameters_avg = np.empty(shape=(len(C_list), len(Gamma_list))) 
highest_accuracy = 0

for c in C_list:
    for gamma in Gamma_list:
            accuracy_sum = 0.0
            accuracy_count = 0.0
            
            for hp in hyperparameters_list:
                accuracy_sum += hp[C_list.index(c), Gamma_list.index(gamma)]
                accuracy_count += 1.0

            accuracy_avg = accuracy_sum/accuracy_count
            hyperparameters_avg[C_list.index(c), Gamma_list.index(gamma)] = accuracy_avg

            if accuracy_avg>highest_accuracy:
                highest_accuracy = accuracy_avg

#check
assert(hyperparameters_avg[0,0] == 
    (hyperparameters_list[0][0][0] 
    + hyperparameters_list[1][0][0] 
    + hyperparameters_list[2][0][0] 
    + hyperparameters_list[3][0][0]
    + hyperparameters_list[4][0][0])
    /5.0
)

C_best, Gamma_best = selectBestHyperparameters(C_list, Gamma_list, hyperparameters_avg, highest_accuracy)

if mute_status != MUTE_RBF_C_GAMMA and mute_status != MUTE_ALL :
    plotHeatmap(10, C_list, Gamma_list, hyperparameters_avg, "Accuracy with C, Gamma K-fold cross-validation")

# test model
models, hyperparameters, accuracy = applySVM_C_Gamma(
    'rbf', # kernel used
    [C_best], 
    [Gamma_best],
    X_train, Y_train, # train set
    X_test, Y_test, # test set
    NOT_MUTED) 

# plot model and accuracy for prediction on TEST data
if mute_status != MUTE_RBF_C_GAMMA and mute_status != MUTE_ALL :
    plotModel(
        11, #fignum 
        X, Y, #dataset
        models[0], 
        "RBF, C and Gamma tuning with K-fold: result of prediction over test set\nwith C_best = %.6f" % C_best + 
        ", Gamma_best = %.6f" % Gamma_best + "\n Accuracy = %.2f" % accuracy
    )



# END #
plt.show()
