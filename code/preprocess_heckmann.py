import csv
from cobra.io import load_matlab_model, read_sbml_model, load_json_model
from cobra.flux_analysis import pfba
from cobra.core import Reaction
from six import iteritems
from cobra.util.context import get_context
import pandas as pd
from statistics import mean
import numpy as np
import os
from scipy.io import loadmat
from cobra.util.context import get_context
from six import iteritems


rxn_con2abundance_heckmann= dict()
with open('../data/heckmann/heckman_protein_ab_S1A_excel_sup.tsv', 'r') as file:
    reader= csv.reader(file, delimiter= '\t')
    header_heckmann= next(reader, None)
    for row in reader:
        rxn, strain, bio_rep, abundance= row[0], row[2], row[5], float(row[6]) * 10e-12
        con= strain+ '_' + bio_rep
        rxn_con2abundance_heckmann[(rxn, con)]= abundance
            
print("Heckmann Abundance:  {}, \tMean:  {}".format(len(rxn_con2abundance_heckmann),
                                                   mean(rxn_con2abundance_heckmann.values())))
print(header_heckmann)


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
            
print("Heckmann Fluxes: {}, Mean: {}".format(len(rxn_con2fluxes_heckmann), mean(rxn_con2fluxes_heckmann.values())))

rxn2kappmax_cal_heckmann= dict()
for rxn_con, abundance in rxn_con2abundance_heckmann.items():
    rxn, con= rxn_con[0], rxn_con[1]
    if rxn_con in rxn_con2fluxes_heckmann:
        flux= rxn_con2fluxes_heckmann[rxn_con]
        kapp= flux/abundance
        if kapp != 0:
            if rxn not in rxn2kappmax_cal_heckmann:
                rxn2kappmax_cal_heckmann[rxn]= kapp
            else:
                if kapp > rxn2kappmax_cal_heckmann[rxn]:
                    rxn2kappmax_cal_heckmann[rxn]= kapp
print("Heckmann kappmax:{} \tMean:  {}".format(len(rxn2kappmax_cal_heckmann), mean(rxn2kappmax_cal_heckmann.values())))


with open('../data/heckmann/heckmann_rxn2kappmax_pFBA.csv', 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    writer.writerow(['rxn', 'K_app^max'])
    for key, value in rxn2kappmax_cal_heckmann.items():
        writer.writerow([key, value])


def calulate_eta_heckmann(v, k, e):
    print("all_abundance_data:  ", len(e))
    eta= dict()
    not_in_rxn2kcat_count= 0
    in_unavailabe_sample_count= 0
    unacceptable_eta_range= []
    for rxn_con, abundance in e.items():
        rxn, con= rxn_con[0], rxn_con[1]
        if rxn in k and rxn not in ['PPK2r', 'PPK2r_b', 'PPKr', 'PPKr_b']:
            kcat= k[rxn]
        else:
            not_in_rxn2kcat_count +=1
            continue
        if con.split("_")[0] not in ['WT1', 'WT2', 'tpi4', 'pgi3']:
            flux= v[rxn_con]
        else:
            in_unavailabe_sample_count+=1
            continue
        calculated_eta= flux/(kcat * abundance)
        if calculated_eta >0 and calculated_eta <= 1:
            eta[(rxn_con)]= calculated_eta
        else:
            unacceptable_eta_range.append(calculated_eta)
    print("Not in rxn2kcat: ", not_in_rxn2kcat_count)
    print("not availabe sample count", in_unavailabe_sample_count)
    print("unacceptable eta range", len(unacceptable_eta_range))
    return eta, unacceptable_eta_range


rxn_con2eta_heckmann, shitty_range_heckmann= calulate_eta_heckmann(rxn_con2fluxes_heckmann,
                                                                   rxn2kappmax_cal_heckmann,
                                                                   rxn_con2abundance_heckmann) 
print("Heckmann eta: {}, \tMean  {}".format(len(rxn_con2eta_heckmann), mean(rxn_con2eta_heckmann.values())))

def convert_to_irreversible(cobra_model):
    """Split reversible reactions into two irreversible reactions

    These two reactions will proceed in opposite directions. This
    guarentees that all reactions in the model will only allow
    positive flux values, which is useful for some modeling problems.

    cobra_model: A Model object which will be modified in place.

    """
    reactions_to_add = []
    coefficients = {}
    for reaction in cobra_model.reactions:
        # If a reaction is reverse only, the forward reaction (which
        # will be constrained to 0) will be left in the model.
        if reaction.lower_bound < 0:
            reverse_reaction = Reaction(reaction.id + "_b")
            reverse_reaction.lower_bound = max(0, -reaction.upper_bound)
            reverse_reaction.upper_bound = -reaction.lower_bound
            coefficients[
                reverse_reaction] = reaction.objective_coefficient * -1
            reaction.lower_bound = max(0, reaction.lower_bound)
            reaction.upper_bound = max(0, reaction.upper_bound)
            # Make the directions aware of each other
            reaction.notes["reflection"] = reverse_reaction.id
            reverse_reaction.notes["reflection"] = reaction.id
            reaction_dict = {k: v * -1
                             for k, v in iteritems(reaction._metabolites)}
            reverse_reaction.add_metabolites(reaction_dict)
            reverse_reaction._model = reaction._model
            reverse_reaction._genes = reaction._genes
            for gene in reaction._genes:
                gene._reaction.add(reverse_reaction)
            reverse_reaction.subsystem = reaction.subsystem
            reverse_reaction.gene_reaction_rule = reaction.gene_reaction_rule
            reactions_to_add.append(reverse_reaction)
    cobra_model.add_reactions(reactions_to_add)
    set_objective(cobra_model, coefficients, additive=True)
    
def set_objective(model, value, additive=False):
    """Set the model objective.

    Parameters
    ----------
    model : cobra model
       The model to set the objective for
    value : model.problem.Objective,
            e.g. optlang.glpk_interface.Objective, sympy.Basic or dict

        If the model objective is linear, the value can be a new Objective
        object or a dictionary with linear coefficients where each key is a
        reaction and the element the new coefficient (float).

        If the objective is not linear and `additive` is true, only values
        of class Objective.

    additive : boolmodel.reactions.Biomass_Ecoli_core.bounds = (0.1, 0.1)
        If true, add the terms to the current objective, otherwise start with
        an empty objective.
    """
    interface = model.problem
    reverse_value = model.solver.objective.expression
    reverse_value = interface.Objective(
        reverse_value, direction=model.solver.objective.direction,
        sloppy=True)

    if isinstance(value, dict):
        if not model.objective.is_Linear:
            raise ValueError('can only update non-linear objectives '
                             'additively using object of class '
                             'model.problem.Objective, not %s' %
                             type(value))

        if not additive:
            model.solver.objective = interface.Objective(
                Zero, direction=model.solver.objective.direction)
        for reaction, coef in value.items():
            model.solver.objective.set_linear_coefficients(
                {reaction.forward_variable: coef,
                 reaction.reverse_variable: -coef})

    elif isinstance(value, (Basic, optlang.interface.Objective)):
        if isinstance(value, Basic):
            value = interface.Objective(
                value, direction=model.solver.objective.direction,
                sloppy=False)
        # Check whether expression only uses variables from current model
        # clone the objective if not, faster than cloning without checking
        if not _valid_atoms(model, value.expression):
            value = interface.Objective.clone(value, model=model.solver)

        if not additive:
            model.solver.objective = value
        else:
            model.solver.objective += value.expression
    else:
        raise TypeError(
            '%r is not a valid objective for %r.' % (value, model.solver))

    context = get_context(model)
    if context:
        def reset():
            model.solver.objective = reverse_value
            model.solver.objective.direction = reverse_value.direction

        context(reset)

model= read_sbml_model('../data/GEMs/iJO1366.xml')
convert_to_irreversible(model)


condition_set_heckmann= set()
for key, value in rxn_con2eta_heckmann.items():
    condition_set_heckmann.add(key[1])
print("Conditions Heckmann:\t", len(condition_set_heckmann))

not_found= ['EX_fe2_e_b', 'EX_fe3_e_b']
met_condition2fluxsum_heckmann= dict()
for met in model.metabolites:
    for condition in condition_set_heckmann:
        for rxn in met.reactions:
            if met in [substrate for substrate in rxn.products]:
                n= rxn.get_coefficient(met.id)
                if rxn.id in not_found:
                    v= 0
                else:
                    v= rxn_con2fluxes_heckmann[(rxn.id, condition)]
                nv= n*v
                if (met.id, condition) in met_condition2fluxsum_heckmann:
                    met_condition2fluxsum_heckmann[(met.id, condition)] += nv
                else:
                    met_condition2fluxsum_heckmann[(met.id, condition)]= nv
print("fluxsums of mets in each condition Heckmann:\t", len(met_condition2fluxsum_heckmann))

mets_set_heckmann= set()
for key, value in met_condition2fluxsum_heckmann.items():
    mets_set_heckmann.add(key[0])
mets_list_heckmann= list(mets_set_heckmann)
print("Heckmann Mets:  {}".format(len(mets_set_heckmann)))

data_heckmann= []
for key, value in rxn_con2eta_heckmann_kcat.items():
    rxn= key[0]
    condition= key[1]
    fluxsums= []
    for met in mets_list_heckmann:
        fluxsums.append(met_condition2fluxsum_heckmann[(met, condition)])
    entry= [rxn]
    entry.append(condition)
    entry.extend(fluxsums)
    entry.append(value)
    data_heckmann.append(entry)
print(len(data_heckmann))

with open('../data/heckmann/final_dataset_heckmann_kcat_pFBA.csv' , 'w') as file:
    writer= csv.writer(file, delimiter= ',')
    header= ['rxn', 'condition']
    header.extend(mets_list_heckmann)
    header.append('eta')
    writer.writerow(header)
    writer.writerows(data_heckmann)

rxn2substrats= dict()
for rxn in model.reactions:
    rxn2substraits[rxn.id]= set()
    for met in rxn.reactants:
        rxn2substraits[rxn.id].add(met.id)

with open('../data/heckmann/rxn2substrates.tsv', 'w') as file:
    writer= csv.writer(file, delimiter= '\t')
    for rxn, substraits in rxn2substrats.items():
        temp_row= [rxn]
        temp_row.extend(list(substraits))
        writer.writerow(temp_row)
