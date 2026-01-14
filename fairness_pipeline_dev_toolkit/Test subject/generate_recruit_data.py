# generate_recruitment_data.py
import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000  # number of candidates

# Core candidate features
years_experience = np.random.randint(0, 21, size=N)
num_roles = np.clip(
    (years_experience / 2 + np.random.normal(loc=0, scale=1.5, size=N)).round(), 0, 15
).astype(int)

education_levels = np.array(["High School", "Bachelor", "Master", "PhD"])
education_probs = [0.15, 0.5, 0.3, 0.05]
education_level = np.random.choice(education_levels, size=N, p=education_probs)

skill_score = np.clip(
    np.random.normal(loc=65, scale=15, size=N), 0, 100
)  # e.g. coding / technical test
interview_score = np.clip(
    np.random.normal(loc=70, scale=12, size=N), 0, 100
)  # panel interview score

company_tiers = np.array(["Low", "Mid", "High"])
company_tier_probs = [0.4, 0.4, 0.2]
previous_company_tier = np.random.choice(company_tiers, size=N, p=company_tier_probs)

# Sensitive attributes
races = np.array(["White", "Black", "Asian", "Hispanic", "Other"])
race_probs = [0.35, 0.25, 0.2, 0.15, 0.05]
race = np.random.choice(races, size=N, p=race_probs)

genders = np.array(["Male", "Female", "Non-binary"])
gender_probs = [0.48, 0.48, 0.04]
gender = np.random.choice(genders, size=N, p=gender_probs)

# Map categorical to numeric contributions for the "true" logit
education_effects = {
    "High School": -0.5,
    "Bachelor": 0.0,
    "Master": 0.3,
    "PhD": 0.4,
}

company_tier_effects = {
    "Low": -0.2,
    "Mid": 0.0,
    "High": 0.25,
}

# Synthetic (problematic) bias effects – for fairness experimentation ONLY
race_effects = {
    "White": 0.2,
    "Black": -0.15,
    "Asian": 0.1,
    "Hispanic": -0.05,
    "Other": 0.0,
}

gender_effects = {
    "Male": 0.05,
    "Female": 0.0,
    "Non-binary": -0.05,
}

# Build the underlying "true" logit for hiring probability
logit = (
    -2.0
    + 0.08 * years_experience
    + 0.05 * num_roles
    + 0.03 * (skill_score - 65) / 10.0
    + 0.04 * (interview_score - 70) / 10.0
)

for i in range(N):
    logit[i] += education_effects[education_level[i]]
    logit[i] += company_tier_effects[previous_company_tier[i]]
    logit[i] += race_effects[race[i]]
    logit[i] += gender_effects[gender[i]]

# Convert logit to probability (sigmoid)
prob = 1 / (1 + np.exp(-logit))

# Sample binary target: hired vs not hired
hired = np.random.binomial(1, prob, size=N)

df = pd.DataFrame(
    {
        "years_experience": years_experience,
        "num_roles": num_roles,
        "education_level": education_level,
        "skill_score": skill_score.round(1),
        "interview_score": interview_score.round(1),
        "previous_company_tier": previous_company_tier,
        "race": race,
        "gender": gender,
        "hired": hired,
    }
)

df.to_csv("synthetic_recruitment_data.csv", index=False)
print(df.head())
print("Saved to synthetic_recruitment_data.csv")
