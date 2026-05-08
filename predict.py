import joblib
import re
import gc
import pandas as pd
from spellchecker import SpellChecker
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# ============================================================
# MARATHI SERVICE TRANSLATION MAP (SERVICE ONLY)
# ============================================================

SERVICE_MARATHI_MAP = {
    "Physiotherapy": "फिजिओथेरपी",
    "Neuro Physiotherapy": "न्यूरो फिजिओथेरपी",
    "Ortho Physiotherapy": "ऑर्थो फिजिओथेरपी",
    "Doctor at Home": "घरगुती डॉक्टर सेवा",
    "Doctor (physician Assistant) Visit": "डॉक्टर भेट",
    "Healthcare attendants": "आरोग्य सहाय्यक",
    "Healthcare Attendant With Food -24 Hours": "जेवणासह २४ तास आरोग्य सहाय्यक",
    "Medical Equipment": "वैद्यकीय उपकरणे",
    "Oxygen Concentrator": "ऑक्सिजन कॉन्सन्ट्रेटर",
    "Rt Insertion ( With Material )": "आरटी इन्सर्शन सेवा"
}

def translate_service_to_marathi(text):
    parts = [p.strip() for p in text.split("→")]
    translated = [SERVICE_MARATHI_MAP.get(p, p) for p in parts]
    return " → ".join(translated)

# ============================================================
# LOAD TRAINED ARTIFACTS
# ============================================================

artifacts = joblib.load("hhc_models.pkl")

service_model = artifacts["service_model"]
sub_model = artifacts["sub_model"]
vectorizer = artifacts["vectorizer"]
le_gender = artifacts["le_gender"]
le_service = artifacts["le_service"]
le_sub = artifacts["le_sub"]

medical_vocab = artifacts["medical_vocab"]


# ============================================================
# LOAD DATA FOR SIMILARITY (HISTORICAL)
# ============================================================
from sqlalchemy import create_engine
from urllib.parse import quote_plus

password = quote_plus("HHC@109")  # or the confirmed password

engine = create_engine(
    f"postgresql+psycopg2://postgres:{password}@192.168.1.109:5432/HHC_main_2024"
)

df = pd.read_sql(
    """
    SELECT
        "Suffered_from",
        service_title,
        sub_service AS recommomded_service
    FROM public.hhcweb_old_hhc_patient_suffering_from
    """,
    engine
)


df["Suffered_from"] = df["Suffered_from"].fillna("unknown").astype(str)

# ============================================================
# FILTER: ALLOWED SERVICES ONLY (NO CONVEYANCE)
# ============================================================

ALLOWED_SERVICES = [
    "Healthcare attendants",
    "Physiotherapy",
    "Doctor at Home",
    "Medical Equipment",
    "Oxygen Concentrator"
]

# ============================================================
# TEXT CLEANING
# ============================================================

spell = SpellChecker()

def clean_text(text):
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(spell.correction(w) or w for w in text.split())

# ============================================================

NON_MEDICAL_WORDS = {
    "ok", "okay", "hi", "hello", "test", "fine", "yes", "no"
}

# ============================================================


# def is_valid_medical_input(text):
#     tokens = text.split()

#     # Too short → reject
#     if len(tokens) < 3:
#         return False

#     # Must contain at least ONE medical word from training
#     for t in tokens:
#         if t in medical_vocab:
#             return True

#     return False

def is_valid_medical_input(text):
    tokens = text.split()

    # Remove obvious non-medical words
    tokens = [t for t in tokens if t not in NON_MEDICAL_WORDS]

    if not tokens:
        return False

    # If single-word input, allow ONLY if it exists in medical vocab
    if len(tokens) == 1:
        return tokens[0] in medical_vocab

    # Multi-word input: at least one medical word must exist
    return any(t in medical_vocab for t in tokens)


# ============================================================
# SIMILARITY SEARCH
# ============================================================

symptom_vectors = vectorizer.transform(df["Suffered_from"])

def recommend_services(symptom, top_k=5):
    q_vec = vectorizer.transform([symptom])
    sims = cosine_similarity(q_vec, symptom_vectors)[0]

    df_temp = df.copy()
    df_temp["similarity"] = sims


    GENERAL_SYMPTOMS = {"cold", "fever", "cough", "flu"}

    if symptom.lower().strip() in GENERAL_SYMPTOMS:
        df_temp = df_temp[
            df_temp["service_title"] == "Doctor at Home"
        ]


    # df_temp = df_temp[df_temp["service_title"].isin(ALLOWED_SERVICES)]

    SIM_THRESHOLD = 0.15

    df_temp = df_temp[
        (df_temp["service_title"].isin(ALLOWED_SERVICES)) &
        (df_temp["similarity"] >= SIM_THRESHOLD)
    ]


    if df_temp.empty:
        return pd.DataFrame(
            columns=["service_title", "recommomded_service", "count"]
        )



    recs = (
        df_temp.sort_values("similarity", ascending=False)
        .head(50)
        .groupby(["service_title", "recommomded_service"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(top_k)
    )

    return recs

# ============================================================
# EXPLANATION + CARE PLAN
# ============================================================

def generate_explanation_and_plan(symptom):
    if any(k in symptom for k in ["stroke", "paralysis", "weakness", "nerve"]):
        return (
            "The reported symptoms indicate neurological impairment affecting movement, coordination, and daily functioning.",
            [
                "Nursing care: Medication administration and vitals monitoring",
                "Physiotherapy: Mobility, balance, and strength rehabilitation",
                "Occupational therapy: Assistance with daily living activities",
                "Equipment: Walker, wheelchair, hand support devices",
                "Monitoring: Fall prevention and rehab progress tracking"
            ]
        )

    elif any(k in symptom for k in ["fracture", "bone", "knee", "back", "pain"]):
        return (
            "Symptoms suggest an orthopedic or musculoskeletal condition requiring rehabilitation.",
            [
                "Pain management nursing care",
                "Physiotherapy for joint mobility",
                "Support braces or mobility aids",
                "Assistance with daily movement",
                "Recovery monitoring"
            ]
        )

    return (
        "The condition requires general health monitoring and daily assistance.",
        [
            "Vitals monitoring",
            "Daily activity assistance",
            "Diet and hydration care",
            "Routine health follow-ups",
            "Doctor consultation if needed"
        ]
    )

# ============================================================
# FINAL HYBRID PREDICTION
# ============================================================

def hybrid_predict(symptom_input, gender_input):
    symptom = clean_text(symptom_input)


    # ================= HARD BLOCK =================
    if not is_valid_medical_input(symptom):
        print("\nInvalid input. Please enter a valid medical condition or symptoms.")
        return


    gender_norm = "Male" if gender_input.lower().startswith("m") else "Female"
    gender_enc = le_gender.transform([gender_norm])[0]

    X_text = vectorizer.transform([symptom])
    if X_text.nnz == 0:
       print("\nSymptoms not recognized. Please provide proper medical details.")
       return

    X = hstack([X_text, [[gender_enc]]])

    proba = service_model.predict_proba(X)[0]
    if proba.max() < 0.45:
        print("\nSymptoms are unclear. Please describe the condition in more detail.")
        return


    service_pred = le_service.inverse_transform(
        service_model.predict(X)
    )[0]

    sub_pred = le_sub.inverse_transform(
        sub_model.predict(X)
    )[0]

    # --------------------------------------------------------
    # CLINICAL OVERRIDE FOR STROKE
    # --------------------------------------------------------
    if "stroke" in symptom or "paralysis" in symptom:
        service_pred = "Healthcare attendants"
        sub_pred = "Healthcare Attendant - 12 Hours - Night"

    explanation, care_plan = generate_explanation_and_plan(symptom)
    recommendations = recommend_services(symptom)

    # ================= ENGLISH OUTPUT =================

    print("\n=== AI Healthcare Recommendation Summary ===\n")
    print(f"Primary Service: {service_pred} → {sub_pred}\n")

    print("Explanation:")
    print(explanation)

    print("\nSuggested Care Plan:")
    for item in care_plan:
        print(f"• {item}")

    print("\nOther Recommended Services:")

    seen = set()
    other_services = []

    for _, r in recommendations.iterrows():
        s = f"{r['service_title']} → {r['recommomded_service']}"
        if s not in seen:
            seen.add(s)
            other_services.append(s)
            print(f"- {s}")

    # ================= MARATHI OUTPUT =================

    print("\n=== एआय आरोग्यसेवा शिफारस सारांश ===\n")
    print(
        "मुख्य सेवा:",
        translate_service_to_marathi(f"{service_pred} → {sub_pred}")
    )

    print("\nस्पष्टीकरण:")
    print(
        "नोंदवलेली लक्षणे रुग्णाच्या आरोग्याशी संबंधित अडचणी दर्शवतात. "
        "या परिस्थितीत योग्य काळजी व सहाय्य आवश्यक आहे."
    )

    print("\nसुचवलेली काळजी योजना:")
    for item in [
        "नर्सिंग सेवा: औषधोपचार आणि जीवनचिन्हांचे निरीक्षण",
        "फिजिओथेरपी: हालचाल व स्नायू बळकटीकरण व्यायाम",
        "दैनंदिन सहाय्य: रोजच्या कामांसाठी मदत",
        "साधने: चालण्यासाठी किंवा सुरक्षिततेसाठी आवश्यक उपकरणे",
        "निरीक्षण: आरोग्य स्थिती व सुधारणा निरीक्षण"
    ]:
        print(f"• {item}")

    print("\nइतर शिफारस केलेल्या सेवा:")
    for s in other_services:
        print("-", translate_service_to_marathi(s))

    gc.collect()

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    symptom = input("Enter patient symptom/condition: ")
    gender = input("Enter patient gender (Male/Female): ")
    hybrid_predict(symptom, gender)
