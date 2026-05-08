import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# -------------------------------------
# STEP 1: Load data
# -------------------------------------
df = pd.read_csv(r"C:\Users\Vrush\Downloads\Patient_service_data.csv")
print("Columns:", df.columns.tolist())

# -------------------------------------
# STEP 2: Clean and preprocess
# -------------------------------------
df['Suffered_from'] = df['Suffered_from'].fillna('unknown').astype(str)
df['Gender'] = df['Gender'].fillna('Unknown')

# Encode categorical fields
le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_service = LabelEncoder()
df['service_title'] = le_service.fit_transform(df['service_title'].fillna('Unknown'))

le_subservice = LabelEncoder()
df['recommomded_service'] = le_subservice.fit_transform(df['recommomded_service'].fillna('Unknown'))

# -------------------------------------
# STEP 3: Text vectorization
# -------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df['Suffered_from'])

# Combine text + gender as features
X = pd.concat([
    pd.DataFrame(X_text.toarray()),
    df[['Gender']].reset_index(drop=True)
], axis=1)
X.columns = X.columns.astype(str)

# -------------------------------------
# STEP 4: Train model for `service_title`
# -------------------------------------
y_service = df['service_title']
X_train, X_test, y_train, y_test = train_test_split(X, y_service, test_size=0.2, random_state=42)

service_model = RandomForestClassifier(n_estimators=200, random_state=42)
service_model.fit(X_train, y_train)

y_pred_service = service_model.predict(X_test)
print("\n=== Service title classification report ===")
print(classification_report(y_test, y_pred_service, zero_division=0))

# -------------------------------------
# STEP 5: Train model for `recommomded_service`
# -------------------------------------
y_sub = df['recommomded_service']

# Remove rare classes with <2 samples
y_sub_counts = pd.Series(y_sub).value_counts()
valid_classes = y_sub_counts[y_sub_counts > 1].index
mask = pd.Series(y_sub).isin(valid_classes)

X_filtered = X[mask].reset_index(drop=True)
y_sub_filtered = pd.Series(y_sub[mask]).reset_index(drop=True)

print(f"\nRemoved {len(y_sub) - len(y_sub_filtered)} rare samples. Remaining: {len(y_sub_filtered)}")

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X_filtered, y_sub_filtered, test_size=0.2, random_state=42, stratify=y_sub_filtered
)

sub_model = RandomForestClassifier(n_estimators=200, random_state=42)
sub_model.fit(X_train2, y_train2)

y_pred_sub = sub_model.predict(X_test2)
print("\n=== Sub-service classification report ===")
print(classification_report(y_test2, y_pred_sub, zero_division=0))

# -------------------------------------





import re
from spellchecker import SpellChecker

spell = SpellChecker()

def clean_and_correct_text(text):
    if not isinstance(text, str):
        return "unknown"

    # lowercase
    text = text.lower()

    # remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # tokenize
    words = text.split()

    # spell correction
    corrected_words = []
    for word in words:
        if word in spell:
            corrected_words.append(word)
        else:
            corrected_words.append(spell.correction(word) or word)

    return " ".join(corrected_words)



# STEP 6: Function to predict both
# -------------------------------------
def predict_patient_service(suffered_from, gender):
    # --- Normalize gender input ---
    gender = str(gender).strip().lower()
    if gender in ["female", "f", "woman", "lady", "girl"]:
        gender_norm = "Female"
    elif gender in ["male", "m", "man", "boy", "gentleman"]:
        gender_norm = "Male"
    else:
        gender_norm = "Unknown"

    # --- Preprocess input ---
    # new_df = pd.DataFrame({"Suffered_from": [suffered_from], "Gender": [gender_norm]})
    cleaned_text = clean_and_correct_text(suffered_from)

    new_df = pd.DataFrame({
        "Suffered_from": [cleaned_text],
        "Gender": [gender_norm]
    })

    new_df['Suffered_from'] = new_df['Suffered_from'].fillna('unknown').astype(str)
    new_df['Gender'] = le_gender.transform(new_df['Gender'])

    # --- Transform text ---
    X_new_text = vectorizer.transform(new_df['Suffered_from'])
    X_new = pd.concat(
        [pd.DataFrame(X_new_text.toarray()), new_df[['Gender']].reset_index(drop=True)],
        axis=1
    )
    X_new.columns = X.columns

    # --- Predict service ---
    pred_service = service_model.predict(X_new)
    service_name = le_service.inverse_transform(pred_service)[0]

    # --- Predict sub-service ---
    pred_sub = sub_model.predict(X_new)
    subservice_name = le_subservice.inverse_transform(pred_sub)[0]

    print("\n=== Predicted Output ===")
    print(f"Suffered_from: {suffered_from}")
    print(f"Predicted Service: {service_name}")
    print(f"Predicted Sub-Service: {subservice_name}")

    return service_name, subservice_name

# # -------------------------------------
# # STEP 7: Try an example
# # -------------------------------------
# predict_patient_service("Severe tooth pain and gum bleeding", "Female")


# -------------------------------------
# STEP 7: Manual input for prediction
# -------------------------------------
suffered_from_input = input("Enter patient symptom/condition: ")
gender_input = input("Enter patient gender (Male/Female): ")

predict_patient_service(suffered_from_input, gender_input)

