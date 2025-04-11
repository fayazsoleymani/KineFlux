from zeep import Client
import hashlib

wsdl = "https://www.brenda-enzymes.org/soap/brenda_zeep.wsdl"
my_email= "f.soleymani.babadi@gmail.com"
password = hashlib.sha256("@#$fsMIS$#@333".encode("utf-8")).hexdigest()
client = Client(wsdl)

def retrieve_regulators_brenda(ec, org):
    ''' finding the regulators, including activators and inhibitors
    corresponding to the EC-number and organism'''
    result= {'inhibitors': set(), 'activators': set()}
    
    parameters_inhibitor = (my_email, password, f"ecNumber*{ec}", f"organism*{org}",
                            "inhibitor*", "commentary*", "ligandStructureId*",
                            "literature*")
    result_inhibitor = client.service.getInhibitors(*parameters_inhibitor)
    for entry in result_inhibitor:
        if entry['inhibitor'] != 'more':
            result['inhibitors'].add(entry['inhibitor'])
        
    
    parameters_activator= (my_email, password, f"ecNumber*{ec}", f"organism*{org}",
                           "activatingCompound*", "commentary*", "ligandStructureId*",
                           "literature*")
    result_activator = client.service.getActivatingCompound(*parameters_activator)
    for entry in result_activator:
        if entry['activatingCompound'] != 'more':
            result['activators'].add(entry['activatingCompound'])
    
    return result

import requests
def retrieve_cid_pubchem(name):
    ''' returns the cids corresponding to a name of coumpoud'''
    url= f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/cids/JSON'
    response= requests.get(url)
    if response.status_code == 200:
        data = response.json()
        cids = data.get('IdentifierList', {}).get('CID', [])
        return cids
    else:
        return []

import time
def retrieve_smiles_pubchem(query):
    '''retrieves smiles curresponding to a query, there are two options for the query:
    name or cid'''
    if query.isnumeric():
        url= f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{query}/record/JSON'
    else:
        url= f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{query}/record/JSON'
    response= requests.get(url)
    time.sleep(1)
    if response.status_code == 200:
        result= response.json()
        for item in result['PC_Compounds'][0]['props']:
            if item['urn']['label'] == 'SMILES' and item['urn']['name'] == 'Canonical':
                return item['value']['sval']
    else:
        return None


def parse_kegg_compound(handle):
    '''parsing the response of kegg for a metabolite'''
    lines = handle.split("\n")
    output = {'kegg_id': None, 'name': [], 'pubchem': None, 'chebi': None}

    keyword = None
    for line in lines:
        if line.startswith("///"):
            break

        if not line.startswith("            "):
            keyword = line[:12].strip()

        data = line[12:].strip().strip(";")
        if keyword == "ENTRY":
            words = data.split()
            entry = words[0]
            output['kegg_id'] = entry
        if keyword == "NAME":
            output['name'].append(data)
        elif keyword == "DBLINKS":
            if ":" in data:
                db, values = data.split(":")
                values = values.split()
                if db == 'PubChem':
                    output['pubchem'] = values
                elif db == 'ChEBI':
                    output['chebi'] = values

    return output


def retrieve_kegg_id(query):
    '''retrieving the kegg id of a compound'''
    query= query.replace(",", " ")
    find_url= 'https://rest.kegg.jp/find/compound/{}'.format(query)
    find_response= requests.get(find_url)
    first_match= find_response.text.split('\n')[0].split('\t')[0]
    if first_match:
        fetch_url= 'https://rest.kegg.jp/get/{}'.format(first_match)
        fetch_response= requests.get(fetch_url)
        kegg_data= parse_kegg_compound(fetch_response.text)
        kegg_id= kegg_data.get('kegg_id', None)
        return kegg_id
    else:
        return None


from cobra.io import load_matlab_model
model= load_matlab_model('../data/GEMs/iJO1366_irrev.mat')
# model
mets= [met.id for met in model.metabolites]
metNames= [met.name for met in model.metabolites]

rxn2ec= dict()
for rxn in model.reactions:
    rxn_id= rxn.id
    ec= rxn.annotation.get('ec-code', None)
    if ec:
        rxn2ec[rxn.id]= ec

met2metName= {met.id: met.name for met in model.metabolites}
metName2kegg= {met.name: met.annotation.get('kegg.compound', '') for met in model.metabolites}


import csv

rxn2regulators_ours= dict()
rxn2regulators_ours_met_id= dict()
with open('../data/heckmann/heckmann_ml_results_trimed_0.6.csv', 'r') as file:
    reader= csv.reader(file, delimiter= ',')
    header= next(reader, None)
    for row in reader:
        added_mets= eval(row[1])
        rxn2regulators_ours_met_id[row[0]]= added_mets
        added_met_names= [met2metName[x] for x in added_mets]
        rxn2regulators_ours[row[0]]= set(added_met_names)

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
    fluxsum_mets= header[2:-1]
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
print("Length metabolite:\t", len(fluxsum_mets))
print("length rxn2dataset:\t", len(rxn2dataset))


import numpy as np

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


from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score 
from sklearn.model_selection import KFold
from scipy.io import loadmat

def sigmoid_custom(x):
    sigmoid_x= 1 / (1 + np.exp(-x))
    return sigmoid_x

def logit_custom(p):
    return np.log(p / (1 - p))

R= dict()
intercept= dict()
rxn_r2_score= dict()
target_rxns= list(rxn2regulators_ours.keys())
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
    featured_met_indices= [fluxsum_mets.index(met) for met in featured_mets]
    
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

all_model_met2cids= dict()
all_model_mets_cids= []
for metName in metNames:
    cids= retrieve_cid_pubchem(metName)
    all_model_mets_cids.extend(cids)
    all_model_met2cids[metName]= cids
all_model_mets_cids= set(all_model_mets_cids)

with open("../data/brenda/metName2cids.csv", 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    writer.writerow(["met", "cids"])
    for key, value in all_model_met2cids.items():
        writer.writerow([key, value])


def find_correlated_mets(rxn, met):
    met_index= fluxsum_mets.index(met)
    fluxsum= rxn2fluxsums[rxn]
    correlated= set()
    for index, metabolite in enumerate(fluxsum_mets):
        if met_index != index and np.any(fluxsum[:, index]) and metabolite not in rxn2substrates[rxn]:
            if np.corrcoef(fluxsum[:, met_index], fluxsum[:, index])[0,1] > 0.8:
                correlated.add(metabolite)
    return correlated


rxn2regulators_ours_extended= dict()
rxn2regulators_ours_extended_met_id= dict()
rxn_met2correlated_mets= dict()
for rxn, regulators in rxn2regulators_ours_met_id.items():
    regulators_correlated= []
    for met in regulators:
        correlated_mets= find_correlated_mets(rxn, met)
        regulators_correlated.extend(correlated_mets)
        rxn_met2correlated_mets[(rxn, met)]= correlated_mets
    rxn2regulators_ours_extended_met_id[rxn]= set(regulators_correlated)
    regulators_correlated_names= set()
    for regulator in set(regulators_correlated):
        regulators_correlated_names.add(met2metName[regulator])
    rxn2regulators_ours_extended[rxn]= set(regulators_correlated_names)

def get_coef_sign(rxn, met):

    met_index= mets.index(met)
    regulators_indices= rxn_met_indices[rxn]

    if met_index in regulators_indices:
        coef_index= regulators_indices.index(met_index)
        coef= R[rxn][coef_index]
    elif met in rxn2regulators_ours_extended_met_id[rxn]:
        for regulator in rxn2regulators_ours_met_id[rxn]:
            extended_mets= rxn_met2correlated_mets[(rxn, regulator)]

            if met in extended_mets:
                return get_coef_sign(rxn, regulator)
    else:
        print('{} does not regulate {}'.format(met, rxn))
        return
    return 'P' if coef >= 0 else 'N'


neg_coef= 0
pos_coef= 0
for rxn in target_rxns:
    substrates= rxn2substrates[rxn]
    for substrate in substrates:
        if get_coef_sign(rxn, substrate) == 'N':
            neg_coef +=1
        elif get_coef_sign(rxn, substrate) == 'P':
            pos_coef +=1
        else:
            print(substrate)
print(pos_coef, neg_coef)    


TP, FP= 0, 0

brenda_data= []

for rxn, regulators in rxn2regulators_ours_extended.items():
    if rxn in rxn2ec:
        
        # obtaining the cids and kegg ids for the regulator
        cid_regulators_ours= []
        temp_cid2met_name= dict()
        for name in regulators:
            cids= all_model_met2cids[name]
            cid_regulators_ours.extend(cids)
            for cid in cids:
                temp_cid2met_name[cid] = name
        # obtaining the regulators based on the EC-number from BRENDA
        ec= rxn2ec[rxn]
        org= 'Escherichia coli'
        all_regulators_brenda= set()
        cid_regulators_brenda= set()
        for ec_number in ec:
            results= retrieve_regulators_brenda(ec_number, org)
            
            # fetching the cids and kegg ids of the BRENDA regulators
            for element in results['inhibitors']:
                all_regulators_brenda.add(element)
                if element in metName2cid_brenda:
                    cids= metName2cid_brenda[element]
                else:
                    cids= retrieve_cid_pubchem(element)
                    metName2cid_brenda[element]= cids
                    
                for cid in cids:
                    if cid in all_model_mets_cids:
                        cid_regulators_brenda.add(cid)
                        if cid in cid_regulators_ours:
                            metName= temp_cid2met_name[cid]
                            for met in mets:
                                if met2metName[met] == metName and met in rxn2regulators_ours_extended_met_id[rxn]:
                                    print("INHIBITOR\t", met, '\t', get_coef_sign(rxn, met))
                        
                        
            for element in results['activators']:
                all_regulators_brenda.add(element)
                if element in metName2cid_brenda:
                    cids= metName2cid_brenda[element]
                else:
                    cids= retrieve_cid_pubchem(element)
                    metName2cid_brenda[element]= cids
                for cid in cids:
                    if cid in all_model_mets_cids:
                        cid_regulators_brenda.add(cid)
                        if cid in cid_regulators_ours:
                            metName= temp_cid2met_name[cid]
                            for met in mets:
                                if met2metName[met] == metName and met in rxn2regulators_ours_extended_met_id[rxn]:
                                    print("ACTIVATOR\t", met, '\t', get_coef_sign(rxn, met))
        
        
        print(rxn, ec)
        
        cid_intersection= set(cid_regulators_ours).intersection(cid_regulators_brenda)
        
        print("Ours:\t", len(regulators), len(cid_regulators_ours),
              "\nBRENDA:\t",len(all_regulators_brenda), len(cid_regulators_brenda),
              "\nIntersection Pubchem:\t", cid_intersection,
              '\n',
              '-' * 100)
        
        brenda_data.append([rxn, ec, cid_regulators_ours, cid_regulators_brenda, cid_intersection])
        
        TP += len(cid_intersection)
        FP += (len(cid_regulators_brenda) - len(cid_intersection)) 
    else:
        print(rxn, '\n', '-' * 100)
print(TP, FP)

with open("../data/brenda/metName2cids_brenda.csv", 'w') as file:
    writer=csv.writer(file, delimiter= ',')
    writer.writerow(['metName', 'cids'])
    for key, value in metName2cid_brenda.items():
        writer.writerow([key, value])
