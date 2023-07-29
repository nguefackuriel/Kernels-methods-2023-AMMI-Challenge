import numpy as np
import pandas as pd
from tqdm import tqdm
from cvxopt import solvers
from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import sparse
from collections import deque
from dataloader import DataLoader
from kernels import Kernel
from kernel_svm_ import Kernel_SVM

#################

"""
Group: The hunters ( Binta Sow & Uriel Nguefack Yefou)
Data Challenge :  kernel methods in Machine learning
Professors: Jean Phillpe Vert 
TA: Juliette Marrie

Description:
-------------
This main script file implements a kernel SVM which uses vanilla k-spectrum feature embedding and 
(k,m)-mismatch embedding as a multiple kernel. This multiple kernel handles all the operations (gram matrix computation, 
function evaluation) and then after we predict on the test data and generate our submission file for each of the three data we have. 
"""
############################################# KERNELS ##############################################

def main():
    ##################################################################################################################

    print('''
    ---------------------------------------------------------------------------------
    ------Generating Submission File First dataset: This may take some time. Please be patient-----
    ---------------------------------------------------------------------------------
    ''')

    dataset = DataLoader('data/Xtr0.csv')

    labels = pd.read_csv('data/Ytr0.csv')
    y = 2.0 * np.array(labels['Bound']) - 1

    test = DataLoader('data/Xte0.csv')

    dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)

    dataset.populate_kmer_set(k = 12) 
    dataset.mismatch_preprocess(k=12, m=2)
    Kernell_1 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 13)
    dataset.mismatch_preprocess(k=13, m=2)
    Kernell_2 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 15)
    dataset.mismatch_preprocess(k=15, m=3)
    Kernell_3 = Kernel(Kernel.mismatch()).gram(dataset.data)

    # Add kernels together
    K = Kernell_1 + Kernell_2 + Kernell_3  

    training = [i for i in range(2000)]
    testing = [i for i in range(2000, 3000)]

    lmda = 1.0#0.8

    alpha = Kernel_SVM.SVM(K[training][:, training], y, lmda)   

    predictions_0 = []
    for i in tqdm(testing):
        val = 0
        for k, j in enumerate(training):
            val += alpha[k]*K[i, j]
        predictions_0.append(np.sign(val))  

    submission_0 = np.where(np.array(predictions_0) == -1.0,0,1)
    # Add the column of Ids
    y_save_0 = np.vstack([np.arange(len(submission_0)), submission_0]).T
    y_save_0[:10]




    ##################################################################################################################

    print('''
    ---------------------------------------------------------------------------------
    ------Generating Submission File Second dataset: This may take some time. Please be patient-----
    ---------------------------------------------------------------------------------
    ''')

    dataset = DataLoader('data/Xtr1.csv')

    labels = pd.read_csv('data/Ytr1.csv')
    y = 2.0 * np.array(labels['Bound']) - 1

    test = DataLoader('data/Xte1.csv')

    dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)

    dataset.populate_kmer_set(k = 12) 
    dataset.mismatch_preprocess(k=12, m=2)
    Kernell_1 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 13)
    dataset.mismatch_preprocess(k=13, m=2)
    Kernell_2 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 15)
    dataset.mismatch_preprocess(k=15, m=3)
    Kernell_3 = Kernel(Kernel.mismatch()).gram(dataset.data)

    # Add kernels together
    K = Kernell_1 + Kernell_2 + Kernell_3  

    training = [i for i in range(2000)]
    testing = [i for i in range(2000, 3000)]

    lmda = 1.0#0.8

    alpha = Kernel_SVM.SVM(K[training][:, training], y, lmda)   

    predictions_1 = []
    for i in tqdm(testing):
        val = 0
        for k, j in enumerate(training):
            val += alpha[k]*K[i, j]
        predictions_1.append(np.sign(val))  

    submission_1 = np.where(np.array(predictions_1) == -1.0,0,1)
    # Add the column of Ids
    y_save_1 = np.vstack([1000 + np.arange(len(submission_1)), submission_1]).T
    y_save_1[:10]



    ##################################################################################################################

    print('''
    ---------------------------------------------------------------------------------
    ------Generating Submission File Third dataset: This may take some time. Please be patient-----
    ---------------------------------------------------------------------------------
    ''')

    dataset = DataLoader('data/Xtr2.csv')

    labels = pd.read_csv('data/Ytr2.csv')
    y = 2.0 * np.array(labels['Bound']) - 1

    test = DataLoader('data/Xte2.csv')

    dataset.X = pd.concat([dataset.X, test.X], axis = 0, ignore_index = True)

    dataset.populate_kmer_set(k = 12) 
    dataset.mismatch_preprocess(k=12, m=2)
    Kernell_1 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 13)
    dataset.mismatch_preprocess(k=13, m=2)
    Kernell_2 = Kernel(Kernel.mismatch()).gram(dataset.data)

    dataset.populate_kmer_set(k = 15)
    dataset.mismatch_preprocess(k=15, m=3)
    Kernell_3 = Kernel(Kernel.mismatch()).gram(dataset.data)

    # Add kernels together
    K = Kernell_1 + Kernell_2 + Kernell_3  

    training = [i for i in range(2000)]
    testing = [i for i in range(2000, 3000)]

    lmda = 1.0#0.8

    alpha = Kernel_SVM.SVM(K[training][:, training], y, lmda)   

    predictions_2 = []
    for i in tqdm(testing):
        val = 0
        for k, j in enumerate(training):
            val += alpha[k]*K[i, j]
        predictions_2.append(np.sign(val))  


    submission_2 = np.where(np.array(predictions_2) == -1.0,0,1)

    # Add the column of Ids
    y_save_2 = np.vstack([2000 + np.arange(len(submission_2)), submission_2]).T
    y_save_2[:10]

    #Concatenate the predictions
    final = np.vstack((y_save_0,y_save_1,y_save_2))

    # Save as a csv file
    np.savetxt('Yte.csv', final,
            delimiter=',', header='Id,Bound', fmt='%i', comments='')
    
if __name__ == "__main__":
    main()