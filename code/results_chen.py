import csv
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.model_selection import KFold

def sigmoid_custom(x):
    sigmoid_x= 1 / (1 + np.exp(-x))
    return sigmoid_x

def logit_custom(p):
    return np.log(p / (1 - p))



import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


qs_results_dir= '../data/chen/qp_results_0.6_TL6500/'

rxn2fluxes_predicted= dict()
rxn2fluxes_measured= dict()

con2fluxes_predicted= dict()
con2fluxes_measured= dict()

condition_corrs= []
for file_name in os.listdir(qs_results_dir):
    if file_name.endswith('.csv') and file_name != 'correlations.csv':
        with open(os.path.join(qs_results_dir, file_name), 'r') as file:
            condition= file_name.replace('fluxes_', '').split(".")[0]

            reader= csv.reader(file, delimiter= ',')
            con2fluxes_measured[condition]= []
            con2fluxes_predicted[condition]= []
            next(reader, None)
            for row in reader:
                if row[0] not in rxn2fluxes_measured:
                    rxn2fluxes_measured[row[0]]= [float(row[1])]
                else:
                    rxn2fluxes_measured[row[0]].append(float(row[1]))
                
                if row[0] not in rxn2fluxes_predicted:
                    rxn2fluxes_predicted[row[0]]= [float(row[2])]
                else:
                    rxn2fluxes_predicted[row[0]].append(float(row[2]))
                    
                con2fluxes_measured[condition].append(float(row[1]))
                con2fluxes_predicted[condition].append(float(row[2]))
            
            
            corr, p_value= pearsonr(np.log(np.array(con2fluxes_measured[condition])+1e-4),
                                    np.log(np.array(con2fluxes_predicted[condition])+1e-4))
            condition_corrs.append(corr)
            print(condition, corr, p_value)
print("Mean:\t", np.mean(np.array(condition_corrs)))
print("Std:\t", np.std(np.array(condition_corrs)))

from scipy.io import loadmat
mets= [element[0][0] for element in loadmat('../data/GEMs/GEM-yeast_irrev.mat')['model']['mets'][0][0]]
rxns= [element[0][0] for element in loadmat('../data/GEMs/GEM-yeast_irrev.mat')['model']['rxns'][0][0]]
rxnNames= [element[0][0] for element in loadmat('../data/GEMs/GEM-yeast_irrev.mat')['model']['rxnNames'][0][0]]



rxn_corrs= []
rxn2corr= dict()
means= dict()
for rxn in rxn2fluxes_predicted.keys():
    v_m= np.array(rxn2fluxes_measured[rxn])
    v_p= np.array(rxn2fluxes_predicted[rxn])
    if np.size(np.where(v_m>0)) / np.size(v_m) > 0.8:
        corr= np.corrcoef(np.array(rxn2fluxes_measured[rxn]), np.array(rxn2fluxes_predicted[rxn]))[0, 1]
        rxn_corrs.append(corr)
        rxn2corr[rxn]= corr
        means[rxn]= np.mean(v_p)
print(len(rxn_corrs))


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18,'font.family': 'arial'})

plt.figure(figsize=(4, 4))

plt.plot(np.array(rxn2fluxes_measured['453']), np.array(rxn2fluxes_predicted['453']), 'o', 
         markersize= 3, color="royalblue")

plt.plot([0, 5.2], [0, 5.2], '-', color='red', linewidth=1)

plt.xlabel(r'$v_{estimated} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)
plt.ylabel(r'$v_{predicted} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)

plt.text(0, 5, 'r_0569', fontsize=18, color='black')


plt.ylim(-0.3, 5.5)
plt.xlim(-0.3, 5.5)

ticks = np.arange(0, 5.5, 2.5)
plt.xticks(ticks)
plt.yticks(ticks)

plt.savefig('../paper/images/rxn_corrs_r0569.png', dpi=300, bbox_inches='tight')

plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

sns.histplot(rxn_corrs, bins=15, kde=False, stat="probability", binrange=(-0.5, 1),
             color="royalblue", edgecolor="black", linewidth=1.5)

plt.xlabel('Correlation')
plt.ylabel('Proportion')

plt.xticks(np.arange(-0.6, 1.1, 0.2))

plt.savefig('../paper/images/rxn_corrs_yeast.png', dpi=300, bbox_inches='tight')

plt.show()


only_subs_r2= []
one_met_r2= []
two_met_r2= []
three_met_r2= []

rxn2best_adj_r2= dict()

with open('../data/chen/chen_ml_results_linear_regression_logit_pFBA.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    for row in reader:
        only_sub=eval(row[2]) 
        if float(only_sub) > -1000:
            n_samples= rxn2samples[row[0]]
            num_subs= int(row[1])
            
            only_subs= eval(row[2])
            adj_r2_subs= 1- (((1- only_subs) * (n_samples-1)) / (n_samples - num_subs - 1))
            only_subs_r2.append(adj_r2_subs)
            
            one_data= eval(row[5])
            adj_r2_one= 1- (((1- one_data[1]) * (n_samples-1)) / (n_samples - num_subs - 2))
            one_met_r2.append(adj_r2_one)
            
            two_data= eval(row[7])
            adj_r2_two= 1- (((1- two_data[1]) * (n_samples-1)) / (n_samples - num_subs - 3)) 
            two_met_r2.append(adj_r2_two)
            
            three_data= eval(row[9])
            adj_r2_three= 1- (((1- three_data[1]) * (n_samples-1)) / (n_samples - num_subs - 4))
            three_met_r2.append(adj_r2_three)
            
            adj_r2s= [adj_r2_subs, adj_r2_one, adj_r2_two, adj_r2_three]
            best_adj_r2= max(adj_r2s)

            temp_index= adj_r2s.index(best_adj_r2)
            adj_r2s.append((temp_index, best_adj_r2))
            rxn2best_adj_r2[row[0]] = best_adj_r2


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

data = np.array([r2 for r2 in list(rxn2best_adj_r2.values())])

plt.figure(figsize=(10, 4))

bins = np.arange(np.floor(data.min()), np.ceil(data.max()) + 0.1, 0.1)

# Set bar color to royal blue and increase edge thickness
sns.histplot(data, bins=bins, kde=False, color="royalblue", edgecolor="black", linewidth=1.5)

plt.xlabel("Adjusted $R^2$")
plt.ylabel("Number of reactions")

plt.axvline(x=0.6, color='red', linestyle='--', linewidth=2)

plt.annotate("", 
             xy=(1.0, plt.gca().get_ylim()[1] * 0.5),  
             xytext=(0.6, plt.gca().get_ylim()[1] * 0.5),  
             arrowprops=dict(arrowstyle='-|>', color='red', lw=2, mutation_scale=20))

xticks = np.arange(-4, 1.1, 0.2)  
xticks = np.append(xticks, 0.6)  
xticks = np.sort(np.unique(xticks))

plt.xticks(xticks)
plt.gca().get_xticklabels()[list(xticks).index(0.6)].set_color('red')

plt.xlim(-1.5, 1)

# Show the plot
plt.savefig('../paper/images/rxn_r2s_yeast.png', dpi=300, bbox_inches='tight')
plt.show()


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t


x = np.log(np.linspace(1e-4, 1000.0001, 1000000))
y_pred = x


x_sample = np.log(np.array(con2fluxes_measured['Yu2021_N30_035R2']) + 1e-4)
y_sample = np.log(np.array(con2fluxes_predicted['Yu2021_N30_035R2']) + 1e-4)


n = len(x_sample)
x_mean = np.mean(x_sample)
SSx = np.sum((x_sample - x_mean) ** 2)
sigma = np.std(y_sample - x_sample)

t_value = t.ppf(0.95, df=n-1)

SE_y = sigma * np.sqrt(1+(1/n) + ((x - x_mean) ** 2) / SSx)

pi_upper = y_pred + t_value * SE_y
pi_lower = y_pred - t_value * SE_y

plt.figure(figsize=(4.5, 4.5))


plt.fill_between(x, pi_lower, pi_upper, color='royalblue', alpha=0.05, label='Prediction Interval')

plt.plot(np.log(np.array(con2fluxes_measured['Yu2021_N30_035R2']) + 1e-4),
         np.log(np.array(con2fluxes_predicted['Yu2021_N30_035R2'])+ 1e-4),
         'o', markersize=1, color= 'royalblue', alpha= 1)

plt.xlim(-10, 5)
plt.ylim(-10, 5)

plt.xlabel(r'$\log v_{estimated} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)
plt.ylabel(r'$\log v_{predicted} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)

ticks = np.arange(-10, 5, 2)
plt.xticks(ticks)
plt.yticks(ticks)

plt.text(-9.5, 4, 'Yu2021_N30_035R2', fontsize=18, color='black')

# Add legend
plt.legend(fontsize=14, bbox_to_anchor=(0.66, 0.23), loc='upper center')

plt.savefig('../paper/images/Yu2021_N30_035R2_pi.png', dpi=300, bbox_inches='tight')

plt.show()



from collections import Counter

def clean_subsys(subsys):
    return " ".join(subsys.split(" ")[2:])

subsystems_raw= loadmat('../data/GEMs/yeast_irrev_subsystems.mat')['subsys']
subsystems= []
subsystems_list= []
for i, element in enumerate(subsystems_raw):
    subsys_clean= []
    for subsys in element[0][0]:
        if len(subsys) > 0:
            subsys_clean.append(clean_subsys(subsys[0]))
#     print(subsys_clean)
    subsystems.append(subsys_clean)
    subsystems_list.extend(subsys_clean)
print(len(subsystems))

for i, rxn in enumerate(model_cobra.reactions):
    rxn.subsystem= subsystems[i]
    
subsystem2count= Counter(subsystems_list)
subsys_keys= list(subsystem2count.keys())
print("Number of subsystems:{}".format(len(subsys_keys)))


import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

N = len(rxns)  # nRxns
p_val_threshold= 0.02

subsys2well_predicted_proportions= dict()
subsys2enriched= dict()
strain2well_predicted_proportions= dict()

all_conditions= list(con2fluxes_predicted.keys())

for con in all_conditions:

    well_predicted= []
    poor_predicted= []

    rxn2pi= dict()

    x_sample = np.array(con2fluxes_measured[con])
    y_sample = np.array(con2fluxes_predicted[con])

    n_samples = len(x_sample)
    x_mean = np.mean(x_sample)
    SSx = np.sum((x_sample - x_mean) ** 2)
    sigma = np.std(y_sample - x_sample)

    t_value = stats.t.ppf(0.95, df=n-1)

    
    for index, vs in enumerate(zip(con2fluxes_predicted[con], con2fluxes_measured[con])):
        v_pred, v_measured= vs[0], vs[1]
        
        SE_y = sigma * np.sqrt(1+(1/n_samples) + ((v_pred - x_mean) ** 2) / SSx)
        
        pi_lower= v_measured - t_value * SE_y
        pi_upper= v_measured + t_value * SE_y
        
        
        if pi_lower <= v_pred <= pi_upper:
            well_predicted.append(index)
        else:
            poor_predicted.append(index)
            
        K = len(well_predicted)   # nWellPredicted

    strain2well_predicted_proportions[con]= K/len(rxns)
    print("{}\nWell predicted:\t{}\tPoor prdicted:\t{}".format(con, K, len(poor_predicted)))  


    well_predicted_subsystems_dict= dict()
    for index in well_predicted:
        if index >= len(model_cobra.reactions):
            continue
        rxn= model_cobra.reactions[int(index)]
        subsystems= rxn.subsystem
        
        for subsystem in subsystems:
            if subsystem not in well_predicted_subsystems_dict:
                well_predicted_subsystems_dict[subsystem] = 1
            else:
                well_predicted_subsystems_dict[subsystem] +=1
    
    p_values= []
    for subsys in subsys_keys:
        
        n = subsystem2count[subsys]   # nRxnsPathway
        
        if subsys in well_predicted_subsystems_dict: 
            k = well_predicted_subsystems_dict[subsys]     # nWellPredictedPathway
        else:
            k= 0
            
        proportions= k/n
        
        if subsys not in subsys2well_predicted_proportions:
            subsys2well_predicted_proportions[subsys]= [proportions]
        else:
            subsys2well_predicted_proportions[subsys].append(proportions)

        # cumulative distribution fucntion
        p_value = 1- stats.hypergeom.cdf(k, N, K, n)
        p_values.append(p_value)
    
#     adjusted_p_values= multipletests(np.array(p_values), method='fdr_bh')[1]
    adjusted_p_values= multipletests(np.array(p_values), method='bonferroni')[1]
    
    for subsys, p_value in zip(subsys_keys, adjusted_p_values):

        if p_value < p_val_threshold:

            if subsys not in subsys2enriched:
                subsys2enriched[subsys] = 1
            else:
                subsys2enriched[subsys] += 1
    print("_" * 50)

for key, value in subsys2well_predicted_proportions.items():
    subsys2well_predicted_proportions[key]= (np.mean(value), np.std(value))


import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 18,'font.family': 'arial'})


subsystems = list(subsys2well_predicted_proportions_sorted.keys())
subsystems= [subsys for subsys in subsystems if subsystem2count[subsys] >= 30]
means = [subsys2well_predicted_proportions_sorted[subsys][0] for subsys in subsystems]
stds = [subsys2well_predicted_proportions_sorted[subsys][1] for subsys in subsystems]


plt.figure(figsize=(13, 3.9))
x_positions = np.arange(len(subsystems))
bars = plt.bar(x_positions, means, yerr=stds, capsize=5, alpha=0.7, 
               color="royalblue", edgecolor="black", linewidth=1.5)


max_error = max(stds) if stds else 0


for bar, subsys, x_pos in zip(bars, subsystems, x_positions):
    if subsys in subsys2enriched:
        value = subsys2enriched[subsys]
        plt.text(x_pos, 1.006, str(value), va='bottom', ha='center', fontsize=10, color='black')


plt.xticks(x_positions, subsystems, rotation=60, ha= 'right')
plt.xlim(-1, len(subsystems))


plt.grid(axis='y', linestyle='--', alpha=0.6)

plt.savefig('../paper/images/subsystems_yeast.png', dpi=300, bbox_inches='tight')

plt.show()



import numpy as np
import matplotlib.pyplot as plt


keys = list(con2fluxes_measured.keys())[:30]
num_plots = len(keys)


rows, cols = 6, 5

# Create subplots
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), sharex=True, sharey=True)

# Flatten axes array for easier iteration
axes = axes.flatten()

# Loop through the keys and plot each one
for idx, (ax, key) in enumerate(zip(axes, keys)):
    ax.plot(np.log(np.array(con2fluxes_measured[key]) + 1e-4),
            np.log(np.array(con2fluxes_predicted[key]) + 1e-4),
            'o', markersize=1, color='blue')

    ax.set_title(f"{key}", fontsize=24)
    ax.set_xlim(-10, 5)
    ax.set_ylim(-10, 5)

    
    if idx % cols == 0:
        ax.set_ylabel(r'$\log v_{predicted} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=24)
    
    if idx >= num_plots - 5:
        ax.set_xlabel(r'$\log v_{estimated} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=24)

# Remove the last (empty) subplot
# fig.delaxes(axes[-1])
# fig.delaxes(axes[-2])
# fig.delaxes(axes[-3])


plt.tight_layout()
plt.savefig('../paper/images/all_conditions_yeast_p1.png', dpi=300, bbox_inches='tight')
plt.show()


