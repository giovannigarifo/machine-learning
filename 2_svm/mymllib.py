'''''''''''''''''''''''''''''''''''''''''''''''''''''''''
    My Machine Learning Library

    Author
    ------
        Giovanni Garifo, giovanni.garifo@polito.it
'''''''''''''''''''''''''''''''''''''''''''''''''''''''''

'''
    -------
    Imports
    -------
'''
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt
from sklearn import svm
import seaborn as sns


'''
    ---------
    CONSTANTS
    ---------
'''
NOT_MUTED = 0
MUTE_ALL = 1
MUTE_LINEAR = 2
MUTE_RBF_C = 3
MUTE_RBF_C_GAMMA = 4


'''
    ---------
    FUNCTIONS
    ---------
'''

def splitDataIntoTVT(SamplesMatrix, LabelsMatrix, train_perc, val_perc, test_perc):
    '''
    Description
    -----------
        Splits the given sample and label matrices into Train, Validation and Test sets, given the percentage of data to get for each split.
        It also shuffle the data before doing any split.

    Parameters
    ----------
    X : matrix
        samples matrix
    Y : vector
        labels vector
    train_perc : float
        percentual of the dataset to use as train set
    val_perc : float
        percentual of the dataset to use as validation set
    test_perc : float
        percentual of the dataset to use as test set

    Returns
    -------
    list of matrix / vectors
        returns the splitted sets
    '''
    try:
        if(train_perc + val_perc + test_perc != 1):
            raise Exception("\n[EXCEPTION] Dataset splitting cannot e performed, proportions are not correct. Exiting program with status 1\n") 
    except Exception as exc:
        print(exc.args[0])
        exit(1)

    # first split into train and test
    X_train, X_test, Y_train, Y_test = train_test_split(SamplesMatrix, LabelsMatrix, test_size = test_perc)

    if(val_perc == 0.0):
        #return only train and test sets
        return X_train, X_test, Y_train, Y_test
    else:
        # split again the train set into train and validation, return train, validation and test sets
        X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = val_perc)
        return X_train, X_val, X_test, Y_train, Y_val, Y_test


def createHyperparametersLists(c_exp_min, c_exp_max, gamma_exp_min, gamma_exp_max):
    '''
    Description
    -----------
        creates the list of values to use for the hyperparameters C and Gamma

    Parameters
    ----------
        the ranges to use as exponents to obtain the hyperparamenters

    Returns
    -------
    two lists
        returns the two list of C values and Gamma values
    '''

    # our range of values for C
    C_list = []
    for i in range(c_exp_min, c_exp_max):
        C_list.append(10**i)
    print("C values: ", C_list, "\n")

    # our range of values for Gamma (used in RBF svm)
    Gamma_list = []
    for i in range(gamma_exp_min, gamma_exp_max):
        Gamma_list.append(10**i)
    print("Gamma values: ", Gamma_list, "\n\n\n")

    return C_list, Gamma_list


def calcPercentageWrongPredictions(Y_known, Y_predicted):
    '''
    Description
    -----------
        Calculate the percentange of mislabeled samples

    Parameters
    ----------
    Y_known : vector
        the vector of the known labels
    Y_predicted : vector
        the vector of the predicted labels

    Returns
    -------
    float
        returns the percentage of wrong predictions 
    '''
    Y_predicted.reshape(Y_predicted.shape[0],1) # reshape to have same shape as Y_known
    i = 0
    num_mislabeled_points = 0

    while(i != Y_known.shape[0]):
        if Y_predicted[i] != Y_known[i]:
            num_mislabeled_points += 1
        i+=1

    return (100*num_mislabeled_points)/Y_known.shape[0] #calc percentage, divide by number of samples (= rows of Y_known)


def applySVM_C(kernel, C_list, X_train, Y_train, X_test, Y_test, mute_status):
    '''
    Description
    -----------
        For each value in C_list, create and apply an SVM to the training data, using
        the given kernel.

    Parameters
    ----------
    kernel : string
        kernel to use. choose between ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    C_list : list
        collection of C values to use
    X_train, Y_train : vectors
        training data
    X_test, Y_test : vectors
        test data

    Returns
    -------
    tuple
        list of models obtained, the list of accuracy values and
        the highest accuracy found.
    '''
    models = [] # list of all the different SVM that we had trained
    accuracy_list = [] # list of the accuracy obtained for each value in C_list (same length)
    highest_accuracy = -1 # the highest accuracy found

    for c in C_list:

        if mute_status != MUTE_ALL:
            print("Performing fit on train set and predict on evaluation set for C =", c)
       
        linearSVM = svm.SVC(C = c, kernel=kernel)
        linearSVM.fit(X_train, Y_train.ravel()) # fit on training data
        Y_predicted = linearSVM.predict(X_test) # predict evaluation data
        accuracy = linearSVM.score(X_test, Y_test) # get accuracy level
        models.append(linearSVM) # save model so that we can plot later

        if mute_status != MUTE_ALL:
            print(" -> Percentage of mislabeled points: %.1f%%" % (calcPercentageWrongPredictions(Y_test, Y_predicted)))
            print(" -> Gives a mean accuracy: %.3f" % accuracy, "\n")
    
        if(accuracy > highest_accuracy): 
            highest_accuracy = accuracy
    
        accuracy_list.append(accuracy)

    return models, accuracy_list, highest_accuracy
        

def applySVM_C_Gamma(kernel, C_list, Gamma_list, X_train, Y_train, X_test, Y_test, mute_status):
    '''
    Description
    -----------
        For each value in C_list and in Gamma_list, create and apply an SVM to the training data, using
        the given kernel.

    Parameters
    ----------
    kernel : string
        kernel to use. choose between ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’
    C_list : list
        collection of C values to use
    Gamma_list : list
        collection of Gamma values to use
    X_train, Y_train : vectors
        training data
    X_test, Y_test : vectors
        test data

    Returns
    -------
    tuple
        list of models obtained, the list of accuracy values and
        the highest accuracy found.
    '''
    models = [] # list of all the different SVM that we had trained
    highest_accuracy = -1 # the highest accuracy found
    hyperparameters = np.empty(shape=(len(C_list), len(Gamma_list))) # matrix of accuracy for each combination of hyperparameters

    for c in C_list:
        for gamma in Gamma_list:

            if mute_status != MUTE_ALL:
                print("Performing fit and predict for RBF SVM with C =", c, "and Gamma =", gamma)
            
            linearSVM = svm.SVC(C = c, gamma=gamma, kernel=kernel)
            linearSVM.fit(X_train, Y_train.ravel()) # fit on training data
            Y_predicted = linearSVM.predict(X_test) # predict evaluation data
            accuracy = linearSVM.score(X_test, Y_test) # get accuracy level
            models.append(linearSVM) # save model so that we can plot later

            if mute_status != MUTE_ALL:
                print(" -> Percentage of mislabeled points: %.1f%%" % (calcPercentageWrongPredictions(Y_test, Y_predicted)))
                print(" -> Gives a mean accuracy of: %.3f" % accuracy, "\n")
    
            if(accuracy > highest_accuracy): 
                highest_accuracy = accuracy
    
            hyperparameters[C_list.index(c), Gamma_list.index(gamma)] = accuracy # save accuracy for this combination of hyperparameters

    return models, hyperparameters, highest_accuracy



# stolen from scikit examples: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def make_meshgrid(x, y, h=.02):
    '''
    Description
    -----------
        Create a mesh of points to plot in

    Parameters
    ----------
        x: data to base x-axis meshgrid on
        y: data to base y-axis meshgrid on
        h: stepsize for meshgrid, optional

    Returns
    -------
        xx, yy : ndarray
    '''

    # calc range max and min values
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1

    # Return coordinate matrices from coordinate vectors.
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    return xx, yy


# stolen from scikit examples: https://scikit-learn.org/stable/auto_examples/svm/plot_iris.html
def plot_contours(axes, model, xx, yy, **params):
    '''
    Description
    -----------
        Plot the decision boundaries for a classifier.

    Parameters
    ----------
        ax: matplotlib axes object
        clf: a classifier
        xx: meshgrid ndarray
        yy: meshgrid ndarray
        params: dictionary of params to pass to contourf, optional
    '''
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = axes.contourf(xx, yy, Z, **params)
    return out


def plotModel(fignum, X, Y, model, title):
    '''
    Description
    -----------
        create a figure and plot the result of the prediction with the decision boundaries

    Parameters
    ----------
    fignum : int
        number of the figure
    X : vector
        sample vector
    Y : vector
        labels vectorS
    model : SVC
        the trained svm
    '''

    # create a figure and a set of subplots, return a figure and an array of Axes objects
    fig, axes = plt.subplots(num = fignum, nrows = 1, ncols = 1)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    # obtain a coordinate matrices to use as grid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # assemble plot
    plot_contours(axes, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    axes.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    axes.set_xlim(xx.min(), xx.max())
    axes.set_ylim(yy.min(), yy.max())
    axes.set_xlabel('Sepal length')
    axes.set_ylabel('Sepal width')
    axes.set_xticks(())
    axes.set_yticks(())
    axes.set_title(title)


def plotModels(fignum, X, Y, models):
    '''
    Description
    -----------
        create a subplots and plot the results of the predictions with the decision boundaries in each sub plot

    Parameters
    ----------
    fignum : int
        number of the figure
    X : vector
        sample vector
    Y : vector
        labels vectorS
    model : list of SVC
        the list of trained svm
    '''

    titles = ('C=10^-3', 'C=10^-2', 'C=10^-1', 'C=10^0', 'C=10^1', 'C=10^2', 'C=10^3', "Best C")
    
    # create a figure and a set of subplots, return a figure and an array of Axes objects
    fig, sub = plt.subplots(num = fignum, nrows = 3, ncols = 3)
    
    plt.subplots_adjust(wspace=0.6, hspace=0.6)

    # obtain a coordinate matrices to use as grid
    X0, X1 = X[:, 0], X[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    # use zip to obtain an iterable collection
    for model, title, axes in zip(models, titles, sub.flatten()):
        plot_contours(axes, model, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
        axes.scatter(X0, X1, c=Y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
        axes.set_xlim(xx.min(), xx.max())
        axes.set_ylim(yy.min(), yy.max())
        axes.set_xlabel('Sepal length')
        axes.set_ylabel('Sepal width')
        axes.set_xticks(())
        axes.set_yticks(())
        axes.set_title(title)



def plotAccuracyComparison(fignum, C_list, accuracy_list, x_label, title):
    '''
    Description
    -----------
        Creates a scatter plot that compares each value of C_list with the
        obtained accuracy of the corresponding model contained in accuracy_list

    Parameters
    ----------
        C_list: list of C values
        accuracy_list: list of accuracy values for each model, to obtain using `score(X,Y)`
        x_label : the label for the x axis
        title : the plot title 

    Returns
    -------
        nothing
    '''
    # plot C values and the respective accuracy
    plt.figure(fignum)
    plt.plot(C_list, accuracy_list, 'o')
    plt.xticks(np.array(C_list))
    plt.stem(C_list,accuracy_list)
    plt.ylim(0,1)
    plt.xscale('log')  
    plt.xlabel(x_label)
    plt.ylabel("Accuracy")
    plt.suptitle(title)


def plotHeatmap(fignum, C_list, Gamma_list, hyperparameters, title):
    '''
    Description
    -----------
        plot heat map of hyperparamenters, C and Gamma

    Parameters
    ----------
        fignum: int
            number of the figure to create
        C_list:  list
            list of C values
        Gamma_list: list
            list of Gamma values
        hyperparameters: matrix 
            accuracies for each C,Gamma couple
        x_label : string
            the label for the x axis
        title : string
            the plot title 

    Returns
    -------
        nothing
    '''
    plt.figure(fignum)
    plt.xscale('linear')
    axes = sns.heatmap(hyperparameters, 
                    cmap='Blues', 
                    linewidth=1, 
                    xticklabels=C_list, 
                    yticklabels=Gamma_list, 
                    annot=True, 
                    vmin=0, vmax=1, 
                    square=True) 
    axes.invert_yaxis()
    axes.set_xlabel('Gamma')
    axes.set_ylabel('C')
    axes.set_title(title)



def selectBestC(C_list, accuracy_list, highest_accuracy):
    '''
    Description
    -----------
        Select C that gave the highest accuracy

    Parameters
    ----------
    C_list : list of floats
        list of used C values
    accuracy_list : list
        list of accuracy levels for the i-th C value
    highest_accuracy : float
        the highest accuracy found

    Returns
    -------
    float
        the best C
    '''

    print("Highest accuracy = %.3f" % highest_accuracy, ", obtained for C =")

    C_bests = []

    i = 0
    for accuracy in accuracy_list:
        if(accuracy == highest_accuracy): 
            print("->", C_list[i])
            C_bests.append(C_list[i]) # get all the C values that gave maximum accuracy
        i+=1

    # Select a C value from the best, we take the smallest one
    C_best = min(C_bests)

    print("Selecting as C_best: ", C_best)
    return C_best



def selectBestHyperparameters(C_list, Gamma_list, hyperparameters, highest_accuracy):
    '''
    Description
    -----------
        Select C that gave the highest accuracy

    Parameters
    ----------
    C_list : list of floats
        list of used C values
    accuracy_list : list
        list of accuracy levels for the i-th C value
    highest_accuracy : float
        the highest accuracy found

    Returns
    -------
    float
        the best C
    '''

    C_bests_list = []
    Gamma_bests_list = []

    print("\n------------------------------------------------")
    print("Highest accuracy = %.3f" % highest_accuracy, ", obtained for hyperparameters =")

    for c in C_list:
        for gamma in Gamma_list:
            if(hyperparameters[C_list.index(c), Gamma_list.index(gamma)] == highest_accuracy): 
                print("-> C=", c, ", Gamma=", gamma)
                C_bests_list.append(c) # get all the C values that gave maximum accuracy
                Gamma_bests_list.append(gamma) # and corresponding gamma

    # Select a C value from the best, we take the smallest one, together with the corresponding value of gamma
    C_best = min(C_bests_list)
    Gamma_best = Gamma_bests_list[C_bests_list.index(C_best)]

    print("Selecting as best Hyperparameters: C=", C_best, ", Gamma=", Gamma_best)
    
    return C_best, Gamma_best