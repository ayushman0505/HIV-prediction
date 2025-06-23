# Install dependencies if needed:
# pip install pandas scikit-learn matplotlib seaborn joblib

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import joblib

# Step 1: Load Datasets
art_coverage = pd.read_csv('art_coverage_by_country_clean.csv')
art_pediatric_coverage = pd.read_csv('art_pediatric_coverage_by_country_clean.csv')
cases_adults = pd.read_csv('no_of_cases_adults_15_to_49_by_country_clean.csv')
deaths = pd.read_csv('no_of_deaths_by_country_clean.csv')
people_living_with_hiv = pd.read_csv('no_of_people_living_with_hiv_by_country_clean.csv')
mother_to_child = pd.read_csv('prevention_of_mother_to_child_transmission_by_country_clean.csv')
population = pd.read_csv('world_population.csv')

# Step 2: Inspect population columns and fix naming
print("✅ Columns in world_population.csv:", population.columns.tolist())

# Rename columns properly if needed
population.rename(columns={
    'Country/Territory': 'Country',  # Assuming your file has this column
    '2022 Population': 'Population',  # Use '2022 Population' as 'Population'
}, inplace=True)

# Remove any extra whitespace in country names
population['Country'] = population['Country'].str.strip()

# Step 3: Merge all datasets
df = people_living_with_hiv[['Country', 'Count_median', 'WHO Region']].copy()
df = df.rename(columns={'Count_median': 'People_Living_with_HIV'})

# Merge all other datasets
df = df.merge(art_coverage[['Country', 'Estimated ART coverage among people living with HIV (%)_median']], on='Country', how='left')
df = df.merge(art_pediatric_coverage[['Country', 'Estimated ART coverage among children (%)_median']], on='Country', how='left')
df = df.merge(cases_adults[['Country', 'Count_median']], on='Country', how='left').rename(columns={'Count_median': 'New_Cases_Adults'})
df = df.merge(deaths[['Country', 'Count_median']], on='Country', how='left').rename(columns={'Count_median': 'Deaths'})
df = df.merge(mother_to_child[['Country', 'Percentage Recieved_median']], on='Country', how='left').rename(columns={'Percentage Recieved_median': 'Mother_to_Child_Prevention'})
df = df.merge(population[['Country', 'Population']], on='Country', how='left')

# Step 4: Create Target Variable (HIV Probability)
df['HIV_Probability'] = df['People_Living_with_HIV'] / df['Population']

# Step 5: Preprocessing
# Fill missing values with column medians
df.fillna(df.median(numeric_only=True), inplace=True)

# Encode 'WHO Region'
le = LabelEncoder()
df['WHO_Region_Encoded'] = le.fit_transform(df['WHO Region'])

# Step 6: Define Features and Target
features = [
    'Estimated ART coverage among people living with HIV (%)_median',
    'Estimated ART coverage among children (%)_median',
    'New_Cases_Adults',
    'Deaths',
    'Mother_to_Child_Prevention',
    'WHO_Region_Encoded'
]

X = df[features]
y = df['HIV_Probability']

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 8: Save the trained model
joblib.dump(model, 'hiv_probability_model.pkl')
print("Model saved as 'hiv_probability_model.pkl'")
