from scipy.io import loadmat
import os

# Heckmann
# based on the supplementary data seet S1A,
# 7th column with the header: fmol_per_gDW_calib_avg
# the unit is modified to mmol/gDW by crossing *10e-12
rxn_con2abundance= dict()
import csv
from statistics import mean, median
with open('../data/heckmann/heckman_protein_ab_S1A_excel_sup.tsv', 'r') as file:
    reader= csv.reader(file, delimiter= '\t')
    header= next(reader, None)
    for row in reader:
        rxn, strain, bio_rep, abundance= row[0], row[2], row[5], float(row[6]) * 10e-12
        con= strain+ '_' + bio_rep
        rxn_con2abundance[(rxn, con)]= abundance
            
print("Heckmann Abundance:  {}, \tMean:  {}".format(len(rxn_con2abundance),
                                                   mean(rxn_con2abundance.values())))

rxn2kappmax= dict()
with open('../data/heckmann/heckmann_rxn2kappmax_pFBA.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    for row in reader:
        rxn2kappmax[row[0]]= float(row[1])
# print(len(rxn2kappmax))

import numpy as np

# creating dataset based on substrates of the reaction
rxn2substrates= dict()
with open('../data/heckmann/rxn2substrates.tsv') as file:
    reader= csv.reader(file, delimiter= '\t')
    for row in reader:
        rxn2substrates[row[0]]= row[1:]
print("Lenght rxn2substrates:\t", len(rxn2substrates))

rxn2dataset= dict()
with open('../data/heckmann/final_dataset_heckmann_kappmax_calculated_pFBA.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    mets_fluxsum= header[2:-1]
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

print("length rxn2dataset:\t", len(rxn2dataset))


rxn2cons= dict()
rxn2fluxsums= dict()
rxn2etas= dict()
rxn2samples= dict()

for rxn, data in rxn2dataset.items():
    
    temp_cons= []
    temp_fluxsums= []
    temp_etas= []
    
    for con_fluxsum_eta in data:
        con, fluxsum, eta= con_fluxsum_eta[0], con_fluxsum_eta[1], con_fluxsum_eta[2]
        temp_cons.append(con)
        temp_fluxsums.append(fluxsum)
        temp_etas.append(eta)
    
    rxn2cons[rxn]= temp_cons
    rxn2fluxsums[rxn]= np.array(temp_fluxsums)
    rxn2etas[rxn]= np.array(temp_etas)
    rxn2samples[rxn]= len(temp_etas)
    
threshold= 0.6
results= dict()
rxn2best_adj_r2= dict()
with open('../data/heckmann/heckmann_ml_results_linear_regression_logit_pFBA.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    for row in reader:
        only_subs=eval(row[2]) 
        if float(only_subs) > -1000:
            n_samples= rxn2samples[row[0]]
            num_subs= int(row[1])
            adj_r2_subs= 1- (((1- only_subs) * (n_samples-1)) / (n_samples - num_subs - 1))
            one_data= eval(row[5])
            adj_r2_one= 1- (((1- one_data[1]) * (n_samples-1)) / (n_samples - num_subs - 2))
            two_data= eval(row[7])
            adj_r2_two= 1- (((1- two_data[1]) * (n_samples-1)) / (n_samples - num_subs - 3)) 
            three_data= eval(row[9])
            adj_r2_three= 1- (((1- three_data[1]) * (n_samples-1)) / (n_samples - num_subs - 4))
            adj_r2s= [adj_r2_subs, adj_r2_one, adj_r2_two, adj_r2_three]
            best_adj_r2= max(adj_r2s)
            temp_index= adj_r2s.index(best_adj_r2)
            adj_r2s.append((temp_index, best_adj_r2))
            rxn2best_adj_r2[row[0]] = best_adj_r2
            if best_adj_r2 > threshold:
                if temp_index == 0:
                    only_subs= ((), only_subs)
                    results[row[0]]= only_subs
                elif temp_index == 1:
                    results[row[0]]= one_data
                elif temp_index == 2:
                    results[row[0]]= two_data
                elif temp_index == 3:
                    results[row[0]]= three_data
                    
with open('../data/heckmann/heckmann_ml_results_trimed_0.6.csv', 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    writer.writerow(['rxn', 'mets', 'r2', 'best_adj_r2'])
    for key, value in results.items():
        writer.writerow([key, value[0], value[1], rxn2best_adj_r2[key]])

print(f"Number of all rxns with r^2 upper than threshold {threshold}:\t{len(results)}")

### Constructing R and intercept
# from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.model_selection import KFold

def sigmoid_custom(x):
    sigmoid_x= 1 / (1 + np.exp(-x))
    return sigmoid_x

def logit_custom(p):
    return np.log(p / (1 - p))

R= dict()
intercept= dict()
rxn_r2_score= dict()
target_rxns= list(results.keys())
rxn_met_indices= dict()
rxn_indices= dict()

model= loadmat(f'../data/GEMs/iJO1366_irrev.mat')['iJO1366']
mets= [element[0][0] for element in model['mets'][0][0]]
rxns= [element[0][0] for element in model['rxns'][0][0]]


for rxn in target_rxns:
    
    rxn_index= rxns.index(rxn)
    rxn_indices[rxn]= rxn_index
    # taking substrates and the best compbination of metabolites for the specific reaction
    featured_mets= rxn2substrates[rxn]+ list(results[rxn][0])
    # index of the specific reaction regarding the fluxsum data -->
    # important because their index in the model is different
    featured_met_indices= [mets_fluxsum.index(met) for met in featured_mets]
    
    X= rxn2fluxsums[rxn][:, featured_met_indices]
    y= rxn2etas[rxn]
    y_logit= logit_custom(y)
    
    reg = LinearRegression()
    reg.fit(X, y_logit)
    
    rxn_met_indices[rxn]= [mets.index(met) for met in featured_mets]

    R[rxn]= reg.coef_
    intercept[rxn]= reg.intercept_
           
    y_pred = sigmoid_custom(reg.predict(X))
    r2_rxn= r2_score(y, y_pred)
    rxn_r2_score[rxn]= r2_rxn


rxn_con2fluxes_heckmann= dict()
with open('../data/heckmann/heckman_fluxes_pFBA_B1.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    flux_header_heckmann= next(reader, None)[1:]
    for row in reader:
        rxn= row[0]
        flux_values= [float(flux) for flux in row[1:19]]
        for strain, flux in zip(flux_header_heckmann, flux_values):
            rxn_con2fluxes_heckmann[(rxn, strain+ '_B1')]= flux

with open('../data/heckmann/heckman_fluxes_pFBA_B2.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    flux_header_heckmann= next(reader, None)[1:]
    for row in reader:
        rxn= row[0]
        flux_values= [float(flux) for flux in row[1:19]]
        for strain, flux in zip(flux_header_heckmann, flux_values):
            rxn_con2fluxes_heckmann[(rxn, strain+ '_B2')]= flux
print("Heckmann Fluxes: {}, Mean: {}".format(len(rxn_con2fluxes_heckmann),
                                             mean(rxn_con2fluxes_heckmann.values())))

con2rxns= dict()
for rxn in target_rxns:
    for con in rxn2cons[rxn]:
        if con not in con2rxns:
            con2rxns[con]= [rxn]
        else:
            con2rxns[con].append(rxn)

for key, value in con2rxns.items():
    print(key, len(value))

import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

con2corr= dict()

for target_con in con2rxns.keys():
# target_con= 'pgi1_B1'
    
    print(f"Condition:\t{target_con}")

    model= loadmat(f'../data/GEMs/modified_models_heckmann/iJO1366_irrev_{target_con.split("_")[0]}.mat')['tempModel']

    mets= [element[0][0] for element in model['mets'][0][0]]
    nMets= len(mets)
    rxns= [element[0][0] for element in model['rxns'][0][0]]
    nRxns= len(rxns)
    S = model['S'][0][0]
    b = model['b'][0][0].squeeze()
    c = model['c'][0][0].squeeze()
    lb = model['lb'][0][0].squeeze()
    ub = model['ub'][0][0].squeeze()

    target_rxns_E_available= []
    for rxn in target_rxns:
        if (rxn, target_con) in rxn_con2abundance:
            target_rxns_E_available.append(rxn)
    print(len(target_rxns_E_available))

    m= gp.Model(f"nlp_{target_con}")
    m.params.NonConvex = 2           # min: -1, max: 2, default: -1
    m.params.ScaleFlag= -1           # min: -1, max: 3, default: -1
    m.params.NumericFocus= 0          # min: 0, max: 3, default: 0
    m.params.FeasibilityTol= 1e-6     # min: 1e-9, max: 1e-2, default: 1e-6
    m.params.OptimalityTol= 1e-6      # min: 1e-9, max: 1e-2, default: 1e-6
    m.params.Presolve= 2              # default: -1, max= 2

    
    
    S_p= np.maximum(S, 0)

    v= m.addMVar(shape= nRxns, lb= lb, ub= ub, vtype= GRB.CONTINUOUS, name= 'v')
    FS= m.addMVar(shape=nMets, vtype= GRB.CONTINUOUS)

    m.addMConstr(S, v, '=', b)
    m.addConstr(FS == S_p @ v)

    alpha= 1e-2

    v_pred_all= dict()
    eta_all= dict()
    x_all= dict()


    for rxn in target_rxns_E_available:

        met_indices= rxn_met_indices[rxn]

        x= m.addVar(vtype=GRB.CONTINUOUS, lb= -4.6, ub= 4.6)

        m.addConstr(x == gp.quicksum(R[rxn][idx] * (FS[i]) for idx, i in enumerate(met_indices)) + \
                    intercept[rxn])

        x_all[rxn]= x

        eta= m.addVar(vtype=GRB.CONTINUOUS, lb= 0, ub= 1)
        # eta = 1 / (1 + exp(-x))
        m.addGenConstrLogistic(x, eta)
        eta_all[rxn]= eta

        v_pred= m.addVar(vtype=GRB.CONTINUOUS)
        m.addConstr(v_pred == rxn_con2abundance[(rxn, target_con)] * rxn2kappmax[rxn] * eta)
        v_pred_all[rxn]= v_pred


    obj_expr= gp.quicksum((v[rxn_indices[rxn]]- v_pred_all[rxn]) * (v[rxn_indices[rxn]]- v_pred_all[rxn])\
                          for rxn in target_rxns_E_available)
    obj_expr += gp.quicksum(alpha * v)

    m.setObjective(obj_expr, GRB.MINIMIZE)

    m.optimize()
    
    
    # checking one solution
    if m.status== GRB.OPTIMAL or m.status= GRB.TIME_LIMIT:
    
        rxn2flux= dict()
        for rxn_con, flux in rxn_con2fluxes_heckmann.items():
            if rxn_con[1] == target_con:
                rxn2flux[rxn_con[0]]= flux

        v_measured= []
        for rxn in rxns:
            v_measured.append(rxn2flux[rxn])
        v_measured=np.array(v_measured)

        correlation= np.corrcoef(np.log(v_measured+1e-5), np.log(v.x+1e-5))[0, 1]
        con2corr[target_con]= correlation
        print("Correlation:\t", correlation)

        count_zero_v_pred= 0
        zero_v_pred_indices= []
        count_zero_v_measure= 0
        fluxes= []
        for i, (v_m, v_p) in enumerate(zip(v_measured, v.x)):
            if v_p == 0 and v_m > 0:
                count_zero_v_pred += 1
                if v_m>0.01:
                    zero_v_pred_indices.append(i)
            if v_m == 0 and v_p > 0:
                count_zero_v_measure += 1
            fluxes.append([i, v_m, v_p])
        print("#zero v_pred & positive v_measure:  ", count_zero_v_pred)
        print("#zero v_measure & positive v_pred:  ", count_zero_v_measure)

        with open(f'../data/heckmann/qp_results/fluxes_{target_con}.csv', 'w') as file:
            writer= csv.writer(file, delimiter= ',')
            writer.writerow(['index', 'v_measured', 'v_pred'])
            for row in fluxes:
                writer.writerow(row)

        plt.figure(figsize=(5, 5))
        plt.plot(np.log(v_measured + 1e-5), np.log(v.x+ 1e-5), 'o', markersize=1, color= 'blue')
        plt.xlabel('v_measured')
        plt.ylabel('v_pred')
        plt.savefig(f'../data/heckmann/qp_results/fluxes_{target_con}.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("_" * 100)
    else:
        print(m.status)
        break

with open('../data/heckmann/qp_results/correlations.csv', 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    writer.writerow(['condition', 'corrcoef'])
    for key, value in con2corr.items():
        writer.writerow([key, value])
