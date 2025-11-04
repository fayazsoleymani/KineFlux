# gene abundance
import csv
from statistics import mean

uniprot2mw= dict()
with open("../data/chen/uniprot.tsv", 'r') as file:
    reader= csv.reader(file, delimiter= '\t')
    next(reader, None)
    for row in reader:
        uniprot2mw[row[0]]= float(row[1])

gene_con2abundance_chen= dict()
condition_files= ['Lahtvee2017', 'Yu2020', 'DiBartolomeo2020', 'Yu2021']

for condition_file in condition_files:
    with open(f'../data/chen/proteomics_{condition_file}.tsv', 'r') as file:
        reader= csv.reader(file, delimiter= '\t')
        header= next(reader, None)[1:]
        for row in reader:
            for con_index, col in enumerate(header):
                # for Yu2020 and Yu 2021 unit: fmol/mgCDW changed to mmol/gCDW by multiplying to 1e-12 * 1e3
                if condition_file in ['Yu2020', 'Yu2021']:
                    gene_con2abundance_chen[(row[0], col)]= float(row[con_index + 1]) * 1e-9
                # for Lahtvee2017 unit: copy/pgCDW changed to mmol/gCDW by multiplying to (1/6.022*1e20)* 1e12
                elif condition_file == 'Lahtvee2017':
                    gene_con2abundance_chen[(row[0], col)]= float(row[con_index + 1]) / (6.022 * 1e8)
                # for DiBartolomeo2020 g/gCDW changed to mmol/gCDW by multiplyting to (1e3/MW(g/mol))
                elif condition_file == 'DiBartolomeo2020' and row[0] in uniprot2mw:
                    gene_con2abundance_chen[(row[0], col)]= float(row[con_index + 1]) * 1e3 / uniprot2mw[row[0]]
                else:
                    break
            
print("Chen Abundance:  {}, \tMean:  {}".format(len(gene_con2abundance_chen),
                                                   mean(gene_con2abundance_chen.values())))

from cobra.io import load_matlab_model
import os

model= load_matlab_model('../data/chen/github/Yeast_kapp-main/kappEstimation/GEM-yeast-split.mat')
rxns= [rxn.id for rxn in model.reactions]
gprs= [rxn.gene_reaction_rule for rxn in model.reactions]
# mets= [met.id for met in model.metabolites]
rxn2gpr= dict()
for rxn, gpr in zip(rxns, gprs):
    if '_fwd' in rxn:
        rxn= rxn.replace('_fwd', '_f')
    elif '_rvs' in rxn:
        rxn= rxn.replace('_rvs', '_b')
    rxn2gpr[rxn]= gpr
print(len(rxn2gpr))


# fluxes and fluxsums
import scipy.io
import os
import numpy as np

rxn_con2fluxes_chen= dict()
con2fluxsum= dict()
# con2mets= dict()
condition_list_chen= []
fluxes_dir= '../data/GEMs/modified_models_chen/'
for file_name in os.listdir(fluxes_dir):
    con= file_name.replace("GEM-yeast_irrev_", "").split(".")[0]
    condition_list_chen.append(con)
    fluxes= scipy.io.loadmat(os.path.join(fluxes_dir, file_name))['modelIrrev']['fluxes'][0][0].squeeze()[:-1]
    rxns= [entry[0][0] for entry in scipy.io.loadmat(os.path.join(fluxes_dir, file_name))['modelIrrev']['rxns'][0][0]]
    S= scipy.io.loadmat(os.path.join(fluxes_dir, file_name))['modelIrrev']['S'][0][0]
    S_p= np.maximum(S, 0)
    FS= S_p @ fluxes
    con2fluxsum[con]= FS
    mets_set_chen= [element[0][0] for element in scipy.io.loadmat(os.path.join(fluxes_dir, file_name))['modelIrrev']['mets'][0][0]]
    for i in range(len(fluxes)): 
        rxn_con2fluxes_chen[(rxns[i], con)]= fluxes[i]
            
print("Chen Fluxes: {}, Mean: {}".format(len(rxn_con2fluxes_chen), mean(rxn_con2fluxes_chen.values())))


# k_app^max reported

rxn2kappmax_rep_chen= dict()
with open("../data/chen/DatasetS4_kappmax.txt", 'r') as file:
    reader= csv.reader(file, delimiter= '\t')
    next(reader, None)
    next(reader, None)
    for row in reader:
        rxn= row[0]
        if '_fwd' in rxn:
            rxn= rxn.replace('_fwd', '_f')
        elif '_rvs' in rxn:
            rxn= rxn.replace('_rvs', '_b')
        rxn2kappmax_rep_chen[rxn]= float(row[4]) * 3600 # 1/s changed to 1/h
print("Chen kappmax:{} \tMean:  {}".format(len(rxn2kappmax_rep_chen),
                                               mean(rxn2kappmax_rep_chen.values())))

# rxn2abundances

rxn_con2abundance_chen= dict()
not_found_keys= []
for rxn_con in rxn_con2fluxes_chen.keys():
    rxn, con= rxn_con[0], rxn_con[1]
    if rxn in rxn2gpr:
        gpr= rxn2gpr[rxn]
    else:
        rxn= rxn.replace('_f', '').replace('_b', '')
        gpr= rxn2gpr[rxn]
    if gpr and 'and' not in gpr:
        clean_gpr= gpr.split(' or ')
        total_abundance= 0
        for gene in clean_gpr:
            if (gene, con) in gene_con2abundance_chen:
                total_abundance += gene_con2abundance_chen[(gene, con)]
            else:
                not_found_keys.append((gene, con))
                continue
        if total_abundance > 0:
            rxn_con2abundance_chen[rxn_con]= total_abundance
print(len(rxn_con2abundance_chen))


with open('../data/chen/chen_rxn_con2abundance.csv', 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    writer.writerow(['Rxn', 'Condition', 'Abundance(mmol/gCDW)'])
    for rxn_con, abundance in rxn_con2abundance_chen.items():
        writer.writerow([rxn_con[0], rxn_con[1], abundance])

# calculat eta

def calulate_eta_chen(v, k, e):
#     print("all_abundance_data:  ", len(e))
    eta= dict()
#     not_in_rxn2kcat_count= 0
#     in_unavailabe_sample_count= 0
    unacceptable_eta_range= []
    for rxn_con, abundance in e.items():
        rxn, con= rxn_con[0], rxn_con[1]

        flux= v[rxn_con]
        if rxn in k:
            kcat= k[rxn]
        else:
            continue
        calculated_eta= flux/(kcat * abundance)
        
        if calculated_eta > 0 and calculated_eta <= 1:
            eta[(rxn_con)]= calculated_eta
        else:
            unacceptable_eta_range.append(calculated_eta)
#     print("Not in rxn2kcat: ", not_in_rxn2kcat_count)
#     print("not availabe sample count", in_unavailabe_sample_count)
    print("unacceptable eta range", len(unacceptable_eta_range))
    return eta

rxn_con2eta_chen= calulate_eta_chen(rxn_con2fluxes_chen, rxn2kappmax_rep_chen, rxn_con2abundance_chen)
print(len(rxn_con2eta_chen))

data_chen= []
for key, value in rxn_con2eta_chen.items():
    rxn= key[0]
    condition= key[1]
    fluxsums= list(con2fluxsum[condition])
    entry= [rxn]
    entry.append(condition)
    entry.extend(fluxsums)
    entry.append(value)
    data_chen.append(entry)
print(len(data_chen))

with open('../data/chen/final_dataset_chen_kappmax_reported_pFBA.csv' , 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    header= ['rxn', 'condition']
    header.extend(mets_set_chen)
    header.append('eta')
    writer.writerow(header)
    writer.writerows(data_chen)

model_irrev= load_matlab_model('../data/GEMs/GEM-yeast_irrev.mat')
rxn2substraits= dict()
for rxn in model_irrev.reactions:
    rxn2substraits[rxn.id]= set()
    for met in rxn.reactants:
        rxn2substraits[rxn.id].add(met.id)
        
with open('../data/chen/rxn2substrates.tsv', 'w') as file:
    writer= csv.writer(file, delimiter= '\t')
    for rxn, substraits in rxn2substraits.items():
        temp_row= [rxn]
        temp_row.extend(list(substraits))
        writer.writerow(temp_row)
