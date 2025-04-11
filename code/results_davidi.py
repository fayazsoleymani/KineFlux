import os
import csv
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.io import loadmat

corrs_kappmax_heckmann= []
qs_results_dir= '../data/davidi/qp_results_0.6_TL1e4/'

rxn2fluxes_predicted= dict()
rxn2fluxes_measured= dict()

con2fluxes_predicted= dict()
con2fluxes_measured= dict()

condition_corrs= []

con2rxns= dict()

for file_name in os.listdir(qs_results_dir):
    if file_name.endswith('.csv') and file_name != 'correlations.csv':
        with open(os.path.join(qs_results_dir, file_name), 'r') as file:
            condition= '.'.join(file_name.replace('fluxes_', '').split(".")[0:2])
            
            model= loadmat(f'../data/GEMs/modified_models_davidi_irrev/iJO1366_irrev_{condition}.mat')['modelIrrev']
            rxns= [element[0][0] for element in model['rxns'][0][0]]
            con2rxns[condition]=rxns
            reader= csv.reader(file, delimiter= ',')
            con2fluxes_measured[condition]= []
            con2fluxes_predicted[condition]= []
            next(reader, None)
            for row in reader:
                rxn_index= int(row[0])
                rxn= rxns[rxn_index]
                if rxn not in rxn2fluxes_measured:
                    rxn2fluxes_measured[rxn]= [float(row[1])]
                else:
                    rxn2fluxes_measured[rxn].append(float(row[1]))
                
                if rxn not in rxn2fluxes_predicted:
                    rxn2fluxes_predicted[rxn]= [float(row[2])]
                else:
                    rxn2fluxes_predicted[rxn].append(float(row[2]))
                    
                con2fluxes_measured[condition].append(float(row[1]))
                con2fluxes_predicted[condition].append(float(row[2]))
            
            corr, p_value= pearsonr(np.log(np.array(con2fluxes_measured[condition])+1e-4),
                                    np.log(np.array(con2fluxes_predicted[condition])+1e-4))
            condition_corrs.append(corr)
            corrs_kappmax_heckmann.append(corr)
            print(condition, corr, p_value)
print("Mean:\t", np.mean(np.array(condition_corrs)))
print("Std:\t", np.std(np.array(condition_corrs)))

import math
rxn_corrs= []
rxn2corr= dict()
for rxn in rxn2fluxes_predicted.keys():
    v_m= np.array(rxn2fluxes_measured[rxn])
    v_p= np.array(rxn2fluxes_predicted[rxn])
    if np.size(np.where(v_m>0)) / np.size(v_m) > 0.8: #np.size(np.where(v_p>0)) / np.size(v_p) > 0.0: 
        corr= np.corrcoef(np.array(rxn2fluxes_measured[rxn]), np.array(rxn2fluxes_predicted[rxn]))[0, 1]
        if not math.isnan(corr):
            rxn_corrs.append(corr)
            rxn2corr[rxn]= corr
print(len(rxn_corrs))


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4))

sns.histplot(rxn_corrs, bins=15, kde=False, stat="probability", binrange=(-0.5, 1),
             color="royalblue", edgecolor="black", linewidth=1.5)

plt.xlabel('Correlation', fontsize= 18)
plt.ylabel('Proportion', fontsize= 18)

plt.xticks(np.arange(-0.6, 1.1, 0.2), fontsize=14)
plt.yticks(fontsize=14)

plt.savefig('../paper/images/rxn_corrs_davidi.png', dpi=300, bbox_inches='tight')

plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import t



plt.figure(figsize=(4.5, 4.5))

plt.plot(np.log(np.array(con2fluxes_measured['GLC_CHEM_mu=0.21_V']) + 1e-4),
         np.log(np.array(con2fluxes_predicted['GLC_CHEM_mu=0.21_V'])+ 1e-4),
         'o', markersize=1, color= 'royalblue')

plt.xlim(-10, 5)
plt.ylim(-10, 5)

plt.xlabel(r'$\log v_{estimated} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)
plt.ylabel(r'$\log v_{predicted} \, \left(\frac{mmol}{gDW \cdot h}\right)$', fontsize=18)

ticks = np.arange(-10, 5, 2)
plt.xticks(ticks, fontsize= 14)
plt.yticks(ticks, fontsize=14)

plt.text(-9.5, 4, 'GLC_CHEM_mu=0.21_V', fontsize=18, color='black')

plt.savefig('../paper/images/GLC_CHEM_mu=0.21_V.png', dpi=300, bbox_inches='tight')

plt.show()



