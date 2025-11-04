import sklearn
import csv
import numpy as np
from statistics import mean, median
from scipy.stats import variation
from sklearn.model_selection import KFold
from sklearn.linear_model import  LinearRegression
from sklearn.metrics import r2_score
from itertools import combinations
import time
import os
import math
import warnings
import multiprocessing
warnings.filterwarnings('ignore')

def sigmoid_custom(x):
    sigmoid_x= 1 / (1 + np.exp(-x))
    return sigmoid_x

def logit_custom(p):
    return np.log(p / (1 - p))

# creating dataset based on substrates of the reaction
rxn2substrates= dict()
with open('../data/chen/rxn2substrates.tsv') as file:
    reader= csv.reader(file, delimiter= '\t')
    for row in reader:
        rxn2substrates[row[0]]= row[1:]
print("Lenght rxn2substrates:\t", len(rxn2substrates))

rxn2dataset= dict()
with open('../data/chen/final_dataset_chen_kappmax_reported_pFBA.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    mets= header[2:-1]
    for row in reader:
        rxn, con= row[0], row[1]
        fluxsums= [float(x) for x in row[2:-1]]
        eta= float(row[-1])
        if eta >= 0.99 or eta <= 0.01:
            continue
        if rxn in rxn2dataset:
            rxn2dataset[rxn].append((con, fluxsums, eta))
        else:
            rxn2dataset[rxn]= [(con, fluxsums, eta)]
print("Length metabolite:\t", len(mets))
print("length rxn2dataset:\t", len(rxn2dataset))

rxn2cons = dict()
rxn2fluxsums = dict()
rxn2etas = dict()
for rxn, data in rxn2dataset.items():

    temp_cons = []
    temp_fluxsums = []
    temp_etas = []

    for con_fluxsum_eta in data:
        con, fluxsum, eta = con_fluxsum_eta[0], con_fluxsum_eta[1], con_fluxsum_eta[2]
        temp_cons.append(con)
        temp_fluxsums.append(fluxsum)
        temp_etas.append(eta)

    rxn2cons[rxn] = temp_cons
    rxn2fluxsums[rxn] = np.array(temp_fluxsums)
    rxn2etas[rxn] = np.array(temp_etas)

CVs= []
for rxn, fluxsum in rxn2fluxsums.items():
    if len(fluxsum)> 10:
        nonzero_cols = np.all(fluxsum, axis=0)
        CVs.append(mean(variation(fluxsum[:, nonzero_cols], axis=0)))
CV_threshold = math.ceil(mean(CVs)*100)/100
print("Variation Threshold:  ", CV_threshold)

def regression_model(X, y):

    model = LinearRegression()
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

    r2_folds = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        nonzero_columns = np.any(X_train, axis=0)
          
        X_train = X_train[:, nonzero_columns]
        y_train = logit_custom(y_train)

        model.fit(X_train, y_train)

        X_test = X_test[:, nonzero_columns]
        y_pred = model.predict(X_test)
        y_pred= sigmoid_custom(y_pred)

        r2_fold = r2_score(y_test, y_pred)
        r2_folds.append(r2_fold)

    avg_r2 = np.mean(r2_folds)

    return avg_r2


def filter_metabolites(mets, target_mets, fluxsum):
    nonzero_cols = np.all(fluxsum, axis=0)
    CVs= variation(fluxsum[:, nonzero_cols], axis=0)
    CV_threshold= math.ceil(mean(CVs)*100)/100
    if np.sum(CVs > CV_threshold) > 250:
        CV_threshold= np.sort(CVs)[::-1][250]
    print(f"Number of nonzero columns:  {nonzero_cols.sum()}\tCV threshold:  {CV_threshold}")    
    filtered_metabolites= set()
    for met in target_mets:
        met_index= mets.index(met)
        mets_fluxsum= fluxsum[:, met_index]
        if np.all(mets_fluxsum):
            CV=variation(mets_fluxsum)
            if CV > CV_threshold:
                filtered_metabolites.add(met)
    return filtered_metabolites

rxn2dataset_substrates = dict()
best_plus_one_data = dict()
best_plus_two_data = dict()
best_plus_three_data = dict()
r2_substrates_all = []
r2_substrates_additional_one_all = []
r2_substrates_additional_two_all = []
r2_substrates_additional_three_all = []
rxns_ten = [rxn for rxn in rxn2dataset.keys() if len(rxn2dataset[rxn]) >= 10 and len(rxn2substrates[rxn])>1]
n_jobs= os.cpu_count()-4
print("Number of filtered rxns:\t", len(rxns_ten))

with open("../data/chen/chen_ml_results_linear_regression_logit_pFBA.csv", 'w') as file:
    writer= csv.writer(file, delimiter=',')
    writer.writerow(['rxn', 'Nsubs', 'OnlySubResults', 'nOtherMets', \
                     'nOneComb', 'OneMetResults', 'nTwoComb', 'TwoMetResults', 'nThreeComb', 'ThreeMetResults'])


def additional_mets_model(met):
    metabolites = substrates + list(met)
    indices = [mets.index(met) for met in metabolites]
    X, y = rxn2fluxsums[rxn][:, indices], rxn2etas[rxn]
    r2_rxn = regression_model(X, y)
    return (met, r2_rxn)

all_results= dict()

for index, rxn in enumerate(rxns_ten):
    
    start_time = time.time()
    
    substrates= rxn2substrates[rxn]
    temp= [len(substrates)]
    indices = [mets.index(met) for met in substrates]
    X, y = rxn2fluxsums[rxn][:, indices], rxn2etas[rxn]
    
    print("index:  ", index, "\tRXN:  ", rxn, "\tNumber of Substrates:  ", len(substrates))
    r2_rxn = regression_model(X, y)
    r2_substrates_all.append(r2_rxn)
    temp.append(r2_rxn)
    print(f'Only substrates R-squared: {r2_rxn}')
    
    other_metabolites = set(mets) - set(substrates)

    temp.append(len(other_metabolites))
    print("Mets count before filtering:\t", len(other_metabolites))
    other_metabolites = filter_metabolites(mets, other_metabolites, rxn2fluxsums[rxn])
    print("Mets count after filtering:\t", len(other_metabolites))
    
    
    ### substrates plus one additional metabolite
    one_combinations= list(combinations(other_metabolites, 1))
    temp.append(len(one_combinations))
    best_r2_plus_one= ('', -1000, 0)

    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(additional_mets_model, one_combinations)
        for met, r2_rxn in results:
            if r2_rxn > best_r2_plus_one[1]:
                best_r2_plus_one= (met, r2_rxn)
    best_plus_one_data[rxn]= best_r2_plus_one
    r2_substrates_additional_one_all.append(best_r2_plus_one[1])
    temp.append(best_r2_plus_one)
    print("best added met:\t", best_r2_plus_one)
    
    
    
    ### substrates plus 2 additional metabolites
    two_combinations= list(combinations(other_metabolites, 2))
    temp.append(len(two_combinations))
    print("Number of Two Combinations:  ", len(two_combinations))
    best_r2_plus_two= ('', -1000, 0)

    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(additional_mets_model, two_combinations)
        for met, r2_rxn in results:
            if r2_rxn > best_r2_plus_two[1]:
                best_r2_plus_two= (met, r2_rxn)     
    best_plus_two_data[rxn]= best_r2_plus_two
    r2_substrates_additional_two_all.append(best_r2_plus_two[1])
    temp.append(best_r2_plus_two)
    print("best two combinations:\t", best_r2_plus_two)

    ### substrates plus 3 additional metabolites
    three_combinations = list(combinations(other_metabolites, 3))
    temp.append(len(three_combinations))
    print("Number of Three Combinations:  ", len(three_combinations))
    best_r2_plus_three = ('', -1000, 0)
    with multiprocessing.Pool(processes=n_jobs) as pool:
        results = pool.map(additional_mets_model, three_combinations)
        for met, r2_rxn in results:
            if r2_rxn > best_r2_plus_three[1]:
                best_r2_plus_three= (met, r2_rxn)     
    best_plus_three_data[rxn]= best_r2_plus_three
    r2_substrates_additional_three_all.append(best_r2_plus_three[1])
    temp.append(best_r2_plus_three)
    print("best three combinations:\t", best_r2_plus_three)

    end_time= time.time()
    print("Time:\t:", end_time-start_time)
    print("-" * 50)

    with open("../data/chen/chen_ml_results_linear_regression_logit_pFBA.csv", 'a') as file:
        writer= csv.writer(file, delimiter=',')
        row= [rxn]
        row.extend(temp)
        writer.writerow(row)
