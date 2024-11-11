
import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination, CausalInference
from pgmpy.factors.discrete import TabularCPD

heartDisease = pd.read_csv(r'heart.csv')
heartDisease = heartDisease.replace('?', np.nan)


heartDisease['age'] = pd.cut(heartDisease['age'], bins=[20, 30, 40, 50, 60, 70, 80], labels=['20-30', '30-40', '40-50', '50-60', '60-70', '70-80'])
heartDisease['chol'] = pd.cut(heartDisease['chol'], bins=[100, 200, 300, 400], labels=['100-200', '200-300', '300-400'])


print('Sample data from the dataset:')
print(heartDisease.head())

# Define Bayesian Network structure
model = BayesianNetwork([
    ('age', 'trestbps'),
    ('age', 'fbs'),
    ('sex', 'trestbps'),
    ('exang', 'trestbps'),
    ('trestbps', 'target'),
    ('fbs', 'target'),
    ('target', 'restecg'),
    ('target', 'thalach'),
    ('target', 'chol')
])


print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)


print('\nInference with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)


print('\n1. Probability of HeartDisease (target) given Age=30-40:')
q = HeartDisease_infer.query(variables=['target'], evidence={'age': '30-40'})
print(q)


print('\n2. Probability of HeartDisease (target) given cholesterol=200-300:')
q = HeartDisease_infer.query(variables=['target'], evidence={'chol': '200-300'})
print(q)


simp_model = BayesianNetwork([('S', 'T'), ('T', 'C'), ('S', 'C')])
cpd_s = TabularCPD(variable='S', variable_card=2, values=[[0.5], [0.5]], state_names={'S': ['m', 'f']})
cpd_t = TabularCPD(variable='T', variable_card=2, values=[[0.25, 0.75], [0.75, 0.25]], evidence=['S'], evidence_card=[2], state_names={'S': ['m', 'f'], 'T': [0, 1]})
cpd_c = TabularCPD(variable='C', variable_card=2, values=[[0.3, 0.4, 0.7, 0.8], [0.7, 0.6, 0.3, 0.2]], evidence=['S', 'T'], evidence_card=[2, 2], state_names={'S': ['m', 'f'], 'T': [0, 1], 'C': [0, 1]})


simp_model.add_cpds(cpd_s, cpd_t, cpd_c)
infer_non_adjust = VariableElimination(simp_model)
print("Non-adjusted inference when T=1:")
print(infer_non_adjust.query(variables=['C'], evidence={'T': 1}))
print("Non-adjusted inference when T=0:")
print(infer_non_adjust.query(variables=['C'], evidence={'T': 0}))

infer_adjusted = CausalInference(simp_model)
print("Adjusted inference with do(T=1):")
print(infer_adjusted.query(variables=['C'], do={'T': 1}))
print("Adjusted inference with do(T=0):")
print(infer_adjusted.query(variables=['C'], do={'T': 0}))