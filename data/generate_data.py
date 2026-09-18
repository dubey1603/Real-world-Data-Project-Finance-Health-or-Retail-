"""
generate_data.py
------------------
Generates a synthetic "Customer Churn" dataset for the
Predictive Modeling Using Machine Learning project.

Target: Churn (1 = customer left, 0 = customer stayed)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1000

age = np.random.randint(18, 70, N)
tenure_months = np.random.randint(1, 72, N)
monthly_charge = np.round(np.random.uniform(20, 120, N), 2)
support_calls = np.random.poisson(2, N)
contract_type = np.random.choice(["Month-to-Month", "One Year", "Two Year"], N, p=[0.5, 0.3, 0.2])
has_internet = np.random.choice([0, 1], N, p=[0.2, 0.8])
satisfaction_score = np.clip(np.random.normal(6, 2, N), 1, 10).round(1)

# Construct churn probability from a logistic combination of features
contract_weight = np.select(
    [contract_type == "Month-to-Month", contract_type == "One Year", contract_type == "Two Year"],
    [1.2, -0.5, -1.5],
)

logit = (
    -1.5
    + 0.03 * (tenure_months.max() - tenure_months) / 10
    + 0.02 * monthly_charge / 10
    + 0.35 * support_calls
    + contract_weight
    - 0.25 * satisfaction_score
    + 0.01 * (70 - age) / 10
)
prob_churn = 1 / (1 + np.exp(-logit))
churn = np.random.binomial(1, prob_churn)

df = pd.DataFrame({
    "CustomerID": [f"CUST{1000+i}" for i in range(N)],
    "Age": age,
    "TenureMonths": tenure_months,
    "MonthlyCharge": monthly_charge,
    "SupportCalls": support_calls,
    "ContractType": contract_type,
    "HasInternetService": has_internet,
    "SatisfactionScore": satisfaction_score,
    "Churn": churn,
})

df.to_csv("/home/claude/predictive-modeling-ml-project/data/customer_churn.csv", index=False)
print(f"Dataset generated: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Churn rate: {df['Churn'].mean():.2%}")
