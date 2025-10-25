# train_model.py
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Beispiel-Datensatz simulieren
# (Du kannst hier später deinen echten Datensatz laden, z. B. pd.read_csv("credit_data.csv"))
data = {
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male', 'Female', 'Male', 'Female'],
    'Married': ['Yes', 'No', 'Yes', 'Yes', 'No', 'No', 'Yes', 'No'],
    'Education': ['Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate', 'Graduate', 'Graduate', 'Not Graduate'],
    'Self_Employed': ['No', 'Yes', 'No', 'No', 'Yes', 'No', 'No', 'Yes'],
    'ApplicantIncome': [5000, 3000, 4000, 6000, 2500, 3500, 4500, 2000],
    'CoapplicantIncome': [0, 1500, 1800, 0, 1200, 800, 0, 1000],
    'LoanAmount': [130, 100, 120, 150, 80, 90, 110, 60],
    'Loan_Amount_Term': [360, 120, 180, 360, 180, 360, 360, 120],
    'Credit_History': [1, 0, 1, 1, 0, 1, 1, 0],
    'Property_Area': ['Urban', 'Rural', 'Urban', 'Semiurban', 'Rural', 'Urban', 'Semiurban', 'Rural'],
    'Loan_Status': ['Y', 'N', 'Y', 'Y', 'N', 'Y', 'Y', 'N']
}

df = pd.DataFrame(data)

# 2. Kategorische Variablen kodieren
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']:
    df[col] = le.fit_transform(df[col])

# 3. Features und Zielvariable trennen
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# 4. Trainings- und Testdaten aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Random-Forest-Modell trainieren
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 6. Modell speichern
joblib.dump(model, 'Random_Forest.sav')

print("✅ Das Modell 'Random_Forest.sav' wurde erfolgreich erstellt und gespeichert!")
