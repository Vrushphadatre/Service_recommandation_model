# # # AI-STYLE HEALTHCARE RECOMMENDATION SYSTEM:


# # import pandas as pd
# # import re
# # from sklearn.model_selection import train_test_split
# # from sklearn.preprocessing import LabelEncoder
# # from sklearn.feature_extraction.text import TfidfVectorizer
# # from sklearn.ensemble import RandomForestClassifier
# # from sklearn.metrics.pairwise import cosine_similarity
# # from spellchecker import SpellChecker

# # # -------------------------------------------------
# # # LOAD DATA
# # # -------------------------------------------------
# # df = pd.read_csv("/mnt/data/Patient_service_data.csv")

# # df['Suffered_from'] = df['Suffered_from'].fillna('unknown').astype(str)
# # df['Gender'] = df['Gender'].fillna('Unknown')

# # # -------------------------------------------------
# # # ENCODING
# # # -------------------------------------------------
# # le_gender = LabelEncoder()
# # df['Gender_enc'] = le_gender.fit_transform(df['Gender'])

# # le_service = LabelEncoder()
# # df['service_enc'] = le_service.fit_transform(df['service_title'])

# # le_sub = LabelEncoder()
# # df['subservice_enc'] = le_sub.fit_transform(df['recommomded_service'])

# # # -------------------------------------------------
# # # TEXT VECTORIZATION
# # # -------------------------------------------------
# # vectorizer = TfidfVectorizer(max_features=3000)
# # X_text = vectorizer.fit_transform(df['Suffered_from'])

# # X = pd.concat(
# #     [pd.DataFrame(X_text.toarray()), df[['Gender_enc']].reset_index(drop=True)],
# #     axis=1
# # )
# # X.columns = X.columns.astype(str)

# # # -------------------------------------------------
# # # TRAIN MODELS
# # # -------------------------------------------------
# # service_model = RandomForestClassifier(n_estimators=200, random_state=42)
# # service_model.fit(X, df['service_enc'])

# # sub_model = RandomForestClassifier(n_estimators=200, random_state=42)
# # sub_model.fit(X, df['subservice_enc'])

# # # -------------------------------------------------
# # # SPELL + TEXT CLEANING
# # # -------------------------------------------------
# # spell = SpellChecker()

# # def clean_text(text):
# #     text = text.lower()
# #     text = re.sub(r"[^a-zA-Z\s]", " ", text)
# #     words = text.split()
# #     return " ".join([spell.correction(w) or w for w in words])

# # # -------------------------------------------------
# # # INTENT DETECTION (DATA-ALIGNED)
# # # -------------------------------------------------
# # INTENT_KEYWORDS = {
# #     "pain": ["pain", "ache", "back", "leg", "knee", "neck"],
# #     "infection": ["fever", "cold", "cough", "infection"],
# #     "general_care": ["weakness", "tired", "support", "care"],
# #     "dental": ["tooth", "gum"],
# # }

# # def detect_intents(text):
# #     intents = []
# #     for intent, keys in INTENT_KEYWORDS.items():
# #         if any(k in text for k in keys):
# #             intents.append(intent)
# #     return intents

# # # -------------------------------------------------
# # # OFFLINE RECOMMENDATION ENGINE
# # # -------------------------------------------------
# # symptom_vectors = vectorizer.transform(df['Suffered_from'])

# # def recommend_services(suffered_from, top_k=5):
# #     q_vec = vectorizer.transform([suffered_from])
# #     sims = cosine_similarity(q_vec, symptom_vectors)[0]

# #     df_temp = df.copy()
# #     df_temp['similarity'] = sims

# #     top_cases = df_temp.sort_values('similarity', ascending=False).head(50)

# #     agg = (
# #         top_cases
# #         .groupby(['service_title', 'recommomded_service'])['similarity']
# #         .sum()
# #         .reset_index()
# #         .sort_values('similarity', ascending=False)
# #     )

# #     results = []
# #     for _, r in agg.head(top_k).iterrows():
# #         confidence = round(r['similarity'] * 100, 1)
# #         results.append({
# #             "service": r['service_title'],
# #             "sub_service": r['recommomded_service'],
# #             "confidence": confidence
# #         })

# #     return results

# # # -------------------------------------------------
# # # HUMAN-LIKE EXPLANATION (OFFLINE)
# # # -------------------------------------------------
# # def generate_explanation(symptom, intents, primary_service):
# #     msg = f"Based on the reported symptoms ({symptom}), "

# #     if "pain" in intents:
# #         msg += "the condition appears related to pain or mobility issues. "
# #     if "infection" in intents:
# #         msg += "there are signs of a possible infection. "
# #     if "general_care" in intents:
# #         msg += "general medical support may be required. "

# #     msg += f"{primary_service} is recommended as it is commonly used in similar past cases."

# #     return msg

# # # -------------------------------------------------
# # # FINAL HYBRID PREDICTION
# # # -------------------------------------------------
# # def hybrid_predict(symptom_input, gender_input):
# #     clean_symptom = clean_text(symptom_input)
# #     intents = detect_intents(clean_symptom)

# #     gender_norm = "Male" if gender_input.lower().startswith("m") else \
# #                   "Female" if gender_input.lower().startswith("f") else "Unknown"

# #     gender_enc = le_gender.transform([gender_norm])[0]

# #     X_new_text = vectorizer.transform([clean_symptom])
# #     X_new = pd.concat(
# #         [pd.DataFrame(X_new_text.toarray()), pd.DataFrame([[gender_enc]], columns=['Gender_enc'])],
# #         axis=1
# #     )
# #     X_new.columns = X.columns

# #     service_pred = service_model.predict(X_new)[0]
# #     sub_pred = sub_model.predict(X_new)[0]

# #     service_name = le_service.inverse_transform([service_pred])[0]
# #     sub_name = le_sub.inverse_transform([sub_pred])[0]

# #     recommendations = recommend_services(clean_symptom)
# #     explanation = generate_explanation(symptom_input, intents, service_name)

# #     print("\n=== AI-style Healthcare Recommendation ===")
# #     print(f"\nPrimary Service: {service_name} → {sub_name}")
# #     print(f"\nExplanation:\n{explanation}")
# #     print("\nOther Recommended Services:")
# #     for r in recommendations:
# #         print(f"- {r['service']} → {r['sub_service']} (confidence: {r['confidence']}%)")

# # # -------------------------------------------------
# # # RUN
# # # -------------------------------------------------
# # symptom = input("Enter patient symptom/condition: ")
# # gender = input("Enter patient gender (Male/Female): ")
# # hybrid_predict(symptom, gender)



# # ============================================================
# # AI-STYLE HEALTHCARE RECOMMENDATION SYSTEM (OFFLINE)
# # ============================================================

# import pandas as pd
# import re
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# from spellchecker import SpellChecker

# # ------------------------------------------------------------
# # LOAD DATA
# # ------------------------------------------------------------
# df = pd.read_csv(r"/content/Patient_service_data.csv")

# df['Suffered_from'] = df['Suffered_from'].fillna('unknown').astype(str)
# df['Gender'] = df['Gender'].fillna('Unknown')

# # ------------------------------------------------------------
# # ENCODING
# # ------------------------------------------------------------
# le_gender = LabelEncoder()
# df['Gender_enc'] = le_gender.fit_transform(df['Gender'])

# le_service = LabelEncoder()
# df['service_enc'] = le_service.fit_transform(df['service_title'])

# le_sub = LabelEncoder()
# df['subservice_enc'] = le_sub.fit_transform(df['recommomded_service'])

# # ------------------------------------------------------------
# # TEXT VECTORIZATION
# # ------------------------------------------------------------
# vectorizer = TfidfVectorizer(max_features=3000)
# X_text = vectorizer.fit_transform(df['Suffered_from'])

# X = pd.concat(
#     [pd.DataFrame(X_text.toarray()), df[['Gender_enc']].reset_index(drop=True)],
#     axis=1
# )
# X.columns = X.columns.astype(str)

# # ------------------------------------------------------------
# # TRAIN MODELS
# # ------------------------------------------------------------
# service_model = RandomForestClassifier(n_estimators=200, random_state=42)
# service_model.fit(X, df['service_enc'])

# sub_model = RandomForestClassifier(n_estimators=200, random_state=42)
# sub_model.fit(X, df['subservice_enc'])

# # ------------------------------------------------------------
# # SPELL + TEXT CLEANING
# # ------------------------------------------------------------
# spell = SpellChecker()

# def clean_text(text):
#     text = text.lower()
#     text = re.sub(r"[^a-zA-Z\s]", " ", text)
#     words = text.split()
#     return " ".join([spell.correction(w) or w for w in words])

# # ------------------------------------------------------------
# # SIMPLE INTENT DETECTION (RULE BASED)
# # ------------------------------------------------------------
# INTENT_KEYWORDS = {
#     "neurological": ["paralysis", "weakness", "numb", "stroke", "nerve"],
#     "orthopedic": ["fracture", "bone", "pain", "back", "leg", "knee"],
#     "general_care": ["fever", "infection", "tired", "support"]
# }

# def detect_intents(text):
#     intents = []
#     for intent, keys in INTENT_KEYWORDS.items():
#         if any(k in text for k in keys):
#             intents.append(intent)
#     return intents

# # ------------------------------------------------------------
# # SIMILAR CASE RETRIEVAL (NO CONFIDENCE SHOWN)
# # ------------------------------------------------------------
# symptom_vectors = vectorizer.transform(df['Suffered_from'])

# def recommend_services(suffered_from, top_k=5):
#     q_vec = vectorizer.transform([suffered_from])
#     sims = cosine_similarity(q_vec, symptom_vectors)[0]

#     df_temp = df.copy()
#     df_temp['similarity'] = sims

#     top_cases = df_temp.sort_values('similarity', ascending=False).head(50)

#     recs = (
#         top_cases
#         .groupby(['service_title', 'recommomded_service'])
#         .size()
#         .reset_index(name='count')
#         .sort_values('count', ascending=False)
#     )

#     return recs.head(top_k)

# # ------------------------------------------------------------
# # EXPLANATION + CARE PLAN (CAN COMMENT / MODIFY LATER)
# # ------------------------------------------------------------
# def generate_explanation_and_plan(symptom, primary_service, primary_sub):
#     explanation = (
#         f"The reported symptoms indicate neurological and mobility-related impairment. "
#         f"{primary_service} is recommended as it is commonly used in similar past cases "
#         f"to improve movement, strength, and daily functioning."
#     )

#     care_plan = [
#         "Clinical support: Pain management, vitals monitoring, basic nursing care",
#         "Physiotherapy: Mobility training, strengthening, range-of-motion exercises",
#         "Occupational therapy: Support for safe daily activities and home adaptation",
#         "Daily assistance: Help with meals, household tasks, and transport if needed",
#         "Equipment: Walker, brace, and bathroom safety aids if prescribed",
#         "Optional monitoring: Tele check-ups, fall alerts, and rehab progress tracking"
#     ]

#     return explanation, care_plan

# # ------------------------------------------------------------
# # FINAL HYBRID PREDICTION
# # ------------------------------------------------------------
# def hybrid_predict(symptom_input, gender_input):
#     clean_symptom = clean_text(symptom_input)
#     intents = detect_intents(clean_symptom)

#     gender_norm = (
#         "Male" if gender_input.lower().startswith("m")
#         else "Female" if gender_input.lower().startswith("f")
#         else "Unknown"
#     )

#     gender_enc = le_gender.transform([gender_norm])[0]

#     X_new_text = vectorizer.transform([clean_symptom])
#     X_new = pd.concat(
#         [pd.DataFrame(X_new_text.toarray()),
#          pd.DataFrame([[gender_enc]], columns=['Gender_enc'])],
#         axis=1
#     )
#     X_new.columns = X.columns

#     service_pred = service_model.predict(X_new)[0]
#     sub_pred = sub_model.predict(X_new)[0]

#     service_name = le_service.inverse_transform([service_pred])[0]
#     sub_name = le_sub.inverse_transform([sub_pred])[0]

#     recommendations = recommend_services(clean_symptom)
#     explanation, care_plan = generate_explanation_and_plan(
#         symptom_input, service_name, sub_name
#     )

#     # --------------------------------------------------------
#     # OUTPUT
#     # --------------------------------------------------------
#     print("\n=== AI Healthcare Recommendation Summary ===\n")
#     print(f"Primary Service: {service_name} → {sub_name}\n")

#     print("Explanation:")
#     print(explanation)

#     print("\nSuggested Care Plan (Short):")
#     for item in care_plan:
#         print(f"• {item}")

#     print("\nOther Recommended Services:")
#     for _, r in recommendations.iterrows():
#         print(f"- {r['service_title']} → {r['recommomded_service']}")

# # ------------------------------------------------------------
# # RUN
# # ------------------------------------------------------------
# symptom = input("Enter patient symptom/condition: ")
# gender = input("Enter patient gender (Male/Female): ")

# hybrid_predict(symptom, gender)






# -------------------------------------Code with AI recommendation style------------------------------------------


# ============================================================
# AI-STYLE HEALTHCARE RECOMMENDATION SYSTEM (OFFLINE)
# ============================================================

import pandas as pd
import re
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

# ------------------------------------------------------------
# SERVICE NAME MARATHI TRANSLATION MAP (CONTROLLED)
# ------------------------------------------------------------
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

def translate_service_to_marathi(service_text):
    parts = [p.strip() for p in service_text.split("→")]
    translated = [SERVICE_MARATHI_MAP.get(p, p) for p in parts]
    return " → ".join(translated)

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_csv("/content/Patient_service_data.csv")

df['Suffered_from'] = df['Suffered_from'].fillna('unknown').astype(str)
df['Gender'] = df['Gender'].fillna('Unknown')

# ------------------------------------------------------------
# ENCODING
# ------------------------------------------------------------
le_gender = LabelEncoder()
df['Gender_enc'] = le_gender.fit_transform(df['Gender'])

le_service = LabelEncoder()
df['service_enc'] = le_service.fit_transform(df['service_title'])

le_sub = LabelEncoder()
df['subservice_enc'] = le_sub.fit_transform(df['recommomded_service'])

# ------------------------------------------------------------
# TEXT VECTORIZATION
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df['Suffered_from'])

X = pd.concat(
    [pd.DataFrame(X_text.toarray()),
     df[['Gender_enc']].reset_index(drop=True)],
    axis=1
)
X.columns = X.columns.astype(str)

# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
service_model = RandomForestClassifier(n_estimators=200, random_state=42)
service_model.fit(X, df['service_enc'])

sub_model = RandomForestClassifier(n_estimators=200, random_state=42)
sub_model.fit(X, df['subservice_enc'])

# ------------------------------------------------------------
# SPELL + TEXT CLEANING
# ------------------------------------------------------------
spell = SpellChecker()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return " ".join(corrected)

# ------------------------------------------------------------
# SIMILAR CASE RETRIEVAL
# ------------------------------------------------------------
symptom_vectors = vectorizer.transform(df['Suffered_from'])

def recommend_services(suffered_from, top_k=5):
    q_vec = vectorizer.transform([suffered_from])
    sims = cosine_similarity(q_vec, symptom_vectors)[0]

    df_temp = df.copy()
    df_temp['similarity'] = sims

    top_cases = df_temp.sort_values('similarity', ascending=False).head(50)

    recs = (
        top_cases
        .groupby(['service_title', 'recommomded_service'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )

    return recs.head(top_k)

# ------------------------------------------------------------
# DYNAMIC EXPLANATION + CARE PLAN (RULE-BASED)
# ------------------------------------------------------------
def generate_explanation_and_plan(symptom):
    symptom = symptom.lower()

    if any(k in symptom for k in ["stroke", "paralysis", "weakness", "nerve"]):
        explanation = (
            "The reported symptoms indicate neurological impairment affecting "
            "movement, coordination, and daily functioning."
        )
        care_plan = [
            "Nursing care: Medication administration and vitals monitoring",
            "Physiotherapy: Mobility, balance, and strength rehabilitation",
            "Occupational therapy: Assistance with daily living activities",
            "Equipment: Walker, wheelchair, hand support devices",
            "Monitoring: Fall prevention and rehab progress tracking"
        ]

    elif any(k in symptom for k in ["fracture", "bone", "knee", "back", "pain"]):
        explanation = (
            "The symptoms suggest a musculoskeletal or orthopedic condition "
            "requiring pain management and movement support."
        )
        care_plan = [
            "Nursing care: Pain management and wound care if needed",
            "Physiotherapy: Joint mobility and strengthening exercises",
            "Equipment: Knee brace, back support, walking aids",
            "Daily assistance: Help with transfers and mobility",
            "Monitoring: Pain and recovery progress tracking"
        ]

    else:
        explanation = (
            "The reported symptoms indicate a general health condition "
            "requiring basic clinical monitoring and support."
        )
        care_plan = [
            "Nursing care: Vitals monitoring and medication support",
            "General assistance: Help with daily activities",
            "Diet and hydration monitoring",
            "Regular health follow-ups",
            "Doctor consultation if symptoms persist"
        ]

    return explanation, care_plan

# ------------------------------------------------------------
# MARATHI TRANSLATION (SAFE TEMPLATES)
# ------------------------------------------------------------
def translate_to_marathi(service, subservice, care_plan, other_services):
    return {
        "title": "=== एआय आरोग्यसेवा शिफारस सारांश ===",
        "primary": f"मुख्य सेवा: {translate_service_to_marathi(service + ' → ' + subservice)}",
        "explanation": (
            "नोंदवलेली लक्षणे रुग्णाच्या आरोग्याशी संबंधित अडचणी दर्शवतात. "
            "या परिस्थितीत योग्य काळजी व सहाय्य आवश्यक आहे."
        ),
        "care_plan": [
            "नर्सिंग सेवा: औषधोपचार आणि जीवनचिन्हांचे निरीक्षण",
            "फिजिओथेरपी: हालचाल व स्नायू बळकटीकरण व्यायाम",
            "दैनंदिन सहाय्य: रोजच्या कामांसाठी मदत",
            "साधने: चालण्यासाठी किंवा सुरक्षिततेसाठी आवश्यक उपकरणे",
            "निरीक्षण: आरोग्य स्थिती व सुधारणा निरीक्षण"
        ],
        "others": [translate_service_to_marathi(s) for s in other_services]
    }

# ------------------------------------------------------------
# FINAL HYBRID PREDICTION
# ------------------------------------------------------------
def hybrid_predict(symptom_input, gender_input):
    clean_symptom = clean_text(symptom_input)

    gender_norm = (
        "Male" if gender_input.lower().startswith("m")
        else "Female" if gender_input.lower().startswith("f")
        else "Unknown"
    )

    gender_enc = le_gender.transform([gender_norm])[0]

    X_new_text = vectorizer.transform([clean_symptom])
    X_new = pd.concat(
        [pd.DataFrame(X_new_text.toarray()),
         pd.DataFrame([[gender_enc]], columns=['Gender_enc'])],
        axis=1
    )
    X_new.columns = X.columns

    service_pred = service_model.predict(X_new)[0]
    sub_pred = sub_model.predict(X_new)[0]

    service_name = le_service.inverse_transform([service_pred])[0]
    sub_name = le_sub.inverse_transform([sub_pred])[0]

    recommendations = recommend_services(clean_symptom)
    explanation, care_plan = generate_explanation_and_plan(clean_symptom)

    print("\n=== AI Healthcare Recommendation Summary ===\n")
    print(f"Primary Service: {service_name} → {sub_name}\n")
    print("Explanation:")
    print(explanation)

    print("\nSuggested Care Plan:")
    for item in care_plan:
        print(f"• {item}")

    print("\nOther Recommended Services:")
    for _, r in recommendations.iterrows():
        print(f"- {r['service_title']} → {r['recommomded_service']}")

    other_services = [
        f"{r['service_title']} → {r['recommomded_service']}"
        for _, r in recommendations.iterrows()
    ]

    marathi = translate_to_marathi(service_name, sub_name, care_plan, other_services)

    print("\n\n" + marathi["title"] + "\n")
    print(marathi["primary"] + "\n")

    print("स्पष्टीकरण:")
    print(marathi["explanation"])

    print("\nसुचवलेली काळजी योजना:")
    for item in marathi["care_plan"]:
        print(f"• {item}")

    print("\nइतर शिफारस केलेल्या सेवा:")
    for s in marathi["others"]:
        print(f"- {s}")

# ------------------------------------------------------------
# RUN
# ------------------------------------------------------------
symptom = input("Enter patient symptom/condition: ")
gender = input("Enter patient gender (Male/Female): ")

hybrid_predict(symptom, gender)
