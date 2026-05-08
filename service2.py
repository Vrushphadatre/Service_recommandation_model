# # Code with DB connection:

# import pandas as pd
# import re
# from sqlalchemy import create_engine
# from sklearn.preprocessing import LabelEncoder
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics.pairwise import cosine_similarity
# from spellchecker import SpellChecker

# # ------------------------------------------------------------
# # DATABASE CONNECTION
# # ------------------------------------------------------------
# DB_CONFIG = {
#     "dbname": "HHC_main_2024",
#     "user": "postgres",
#     "password": "postgres",
#     "host": "192.168.1.109",
#     "port": "5432"
# }

# engine = create_engine(
#     f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
#     f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
# )

# # ------------------------------------------------------------
# # SQL QUERY
# # ------------------------------------------------------------
# QUERY = """
# SELECT
#     1 AS src_order,
#     ROW_NUMBER() OVER () AS pk_id,
#     ptn."Suffered_from",
#     srv.service_title,
#     subsrv.recommomded_service AS sub_service,
#     gen.name AS pt_gender
# FROM public.hhcweb_agg_hhc_patients ptn
# LEFT JOIN public.hhcweb_agg_hhc_events eve 
#     ON eve.agg_sp_pt_id_id = ptn.agg_sp_pt_id
# RIGHT JOIN public.hhcweb_agg_hhc_event_plan_of_care evepoc 
#     ON eve.eve_id = evepoc.eve_id_id
# LEFT JOIN public.hhcweb_agg_hhc_gender gen 
#     ON ptn.gender_id_id = gen.gender_id
# RIGHT JOIN public.hhcweb_agg_hhc_services srv 
#     ON evepoc.srv_id_id = srv.srv_id
# RIGHT JOIN public.hhcweb_agg_hhc_sub_services subsrv 
#     ON evepoc.sub_srv_id_id = subsrv.sub_srv_id

# UNION ALL

# SELECT
#     2 AS src_order,
#     pk_id,
#     "Suffered_from",
#     service_title,
#     sub_service,
#     pt_gender
# FROM public.hhcweb_old_hhc_patient_suffering_from
# ORDER BY src_order, pk_id;
# """

# # ------------------------------------------------------------
# # LOAD DATA
# # ------------------------------------------------------------
# df = pd.read_sql(QUERY, engine)
# df.rename(columns={
#     "pt_gender": "Gender",
#     "sub_service": "recommomded_service"
# }, inplace=True)

# df['Suffered_from'] = df['Suffered_from'].fillna('unknown').astype(str)
# df['Gender'] = df['Gender'].fillna('Unknown')

# # ------------------------------------------------------------
# # SERVICE NAME MARATHI MAP
# # ------------------------------------------------------------
# SERVICE_MARATHI_MAP = {
#     "Physiotherapy": "फिजिओथेरपी",
#     "Neuro Physiotherapy": "न्यूरो फिजिओथेरपी",
#     "Ortho Physiotherapy": "ऑर्थो फिजिओथेरपी",
#     "Doctor at Home": "घरगुती डॉक्टर सेवा",
#     "Doctor (physician Assistant) Visit": "डॉक्टर भेट",
#     "Healthcare attendants": "आरोग्य सहाय्यक",
#     "Healthcare Attendant With Food -24 Hours": "जेवणासह २४ तास आरोग्य सहाय्यक",
#     "Medical Equipment": "वैद्यकीय उपकरणे",
#     "Oxygen Concentrator": "ऑक्सिजन कॉन्सन्ट्रेटर",
#     "Rt Insertion ( With Material )": "आरटी इन्सर्शन सेवा"
# }

# def translate_service_to_marathi(service_text):
#     parts = [p.strip() for p in service_text.split("→")]
#     translated = [SERVICE_MARATHI_MAP.get(p, p) for p in parts]
#     return " → ".join(translated)

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
#     [pd.DataFrame(X_text.toarray()),
#      df[['Gender_enc']].reset_index(drop=True)],
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
#     text = re.sub(r"[^a-z\s]", " ", text)
#     words = text.split()
#     corrected = [spell.correction(w) or w for w in words]
#     return " ".join(corrected)

# # ------------------------------------------------------------
# # SIMILAR CASE RETRIEVAL
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
# # EXPLANATION + CARE PLAN
# # ------------------------------------------------------------
# def generate_explanation_and_plan(symptom):
#     symptom = symptom.lower()
#     if any(k in symptom for k in ["stroke", "paralysis", "weakness", "nerve"]):
#         explanation = "Symptoms indicate neurological impairment affecting mobility."
#         care_plan = [
#             "Nursing care: Medication and vitals monitoring",
#             "Physiotherapy: Mobility and balance training",
#             "Occupational therapy support",
#             "Mobility equipment",
#             "Fall prevention monitoring"
#         ]
#     elif any(k in symptom for k in ["fracture", "bone", "knee", "back", "pain"]):
#         explanation = "Symptoms suggest an orthopedic or musculoskeletal condition."
#         care_plan = [
#             "Pain management nursing care",
#             "Physiotherapy for joint movement",
#             "Support braces or aids",
#             "Daily mobility assistance",
#             "Recovery monitoring"
#         ]
#     else:
#         explanation = "General health condition requiring monitoring and assistance."
#         care_plan = [
#             "Vitals monitoring",
#             "Daily activity assistance",
#             "Diet and hydration care",
#             "Routine follow-ups",
#             "Doctor consultation if needed"
#         ]
#     return explanation, care_plan

# # ------------------------------------------------------------
# # MARATHI EXPLANATION + CARE PLAN (TEMPLATE)
# # ------------------------------------------------------------
# def translate_to_marathi(service, subservice, care_plan, other_services):
#     return {
#         "title": "=== एआय आरोग्यसेवा शिफारस सारांश ===",
#         "primary": f"मुख्य सेवा: {translate_service_to_marathi(service + ' → ' + subservice)}",
#         "explanation": (
#             "नोंदवलेली लक्षणे रुग्णाच्या आरोग्याशी संबंधित अडचणी दर्शवतात. "
#             "या परिस्थितीत योग्य काळजी व सहाय्य आवश्यक आहे."
#         ),
#         "care_plan": [
#             "नर्सिंग सेवा: औषधोपचार आणि जीवनचिन्हांचे निरीक्षण",
#             "फिजिओथेरपी: हालचाल व स्नायू बळकटीकरण व्यायाम",
#             "दैनंदिन सहाय्य: रोजच्या कामांसाठी मदत",
#             "साधने: चालण्यासाठी किंवा सुरक्षिततेसाठी आवश्यक उपकरणे",
#             "निरीक्षण: आरोग्य स्थिती व सुधारणा निरीक्षण"
#         ],
#         "others": [translate_service_to_marathi(s) for s in other_services]
#     }

# # ------------------------------------------------------------
# # FINAL HYBRID PREDICTION
# # ------------------------------------------------------------
# def hybrid_predict(symptom_input, gender_input):
#     clean_symptom = clean_text(symptom_input)
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
#     explanation, care_plan = generate_explanation_and_plan(clean_symptom)

#     # ---------------- ENGLISH OUTPUT ----------------
#     print("\n=== AI Healthcare Recommendation Summary ===\n")
#     print(f"Primary Service: {service_name} → {sub_name}\n")
#     print("Explanation:")
#     print(explanation)

#     print("\nSuggested Care Plan:")
#     for item in care_plan:
#         print(f"• {item}")

#     print("\nOther Recommended Services:")
#     other_services = []
#     for _, r in recommendations.iterrows():
#         s_text = f"{r['service_title']} → {r['recommomded_service']}"
#         other_services.append(s_text)
#         print(f"- {s_text}")

#     # ---------------- MARATHI OUTPUT ----------------
#     marathi = translate_to_marathi(service_name, sub_name, care_plan, other_services)

#     print("\n\n" + marathi["title"] + "\n")
#     print(marathi["primary"] + "\n")

#     print("स्पष्टीकरण:")
#     print(marathi["explanation"])

#     print("\nसुचवलेली काळजी योजना:")
#     for item in marathi["care_plan"]:
#         print(f"• {item}")

#     print("\nइतर शिफारस केलेल्या सेवा:")
#     for s in marathi["others"]:
#         print(f"- {s}")

# # ------------------------------------------------------------
# # RUN
# # ------------------------------------------------------------
# symptom = input("Enter patient symptom/condition: ")
# gender = input("Enter patient gender (Male/Female): ")

# hybrid_predict(symptom, gender)








# Time consuming code with DB connection:

# ============================================================
# AI-STYLE HEALTHCARE RECOMMENDATION SYSTEM (DB + OPTIMIZED)
# ============================================================

import pandas as pd
import re
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity
from spellchecker import SpellChecker

# ------------------------------------------------------------
# DATABASE CONNECTION
# ------------------------------------------------------------
DB_CONFIG = {
    "dbname": "HHC_main_2024",
    "user": "postgres",
    "password": "postgres",
    "host": "192.168.1.109",
    "port": "5432"
}

engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ------------------------------------------------------------
# SQL QUERY
# ------------------------------------------------------------
QUERY = """
SELECT
    1 AS src_order,
    ROW_NUMBER() OVER () AS pk_id,
    ptn."Suffered_from",
    srv.service_title,
    subsrv.recommomded_service AS sub_service,
    gen.name AS pt_gender
FROM public.hhcweb_agg_hhc_patients ptn
LEFT JOIN public.hhcweb_agg_hhc_events eve 
    ON eve.agg_sp_pt_id_id = ptn.agg_sp_pt_id
RIGHT JOIN public.hhcweb_agg_hhc_event_plan_of_care evepoc 
    ON eve.eve_id = evepoc.eve_id_id
LEFT JOIN public.hhcweb_agg_hhc_gender gen 
    ON ptn.gender_id_id = gen.gender_id
RIGHT JOIN public.hhcweb_agg_hhc_services srv 
    ON evepoc.srv_id_id = srv.srv_id
RIGHT JOIN public.hhcweb_agg_hhc_sub_services subsrv 
    ON evepoc.sub_srv_id_id = subsrv.sub_srv_id

UNION ALL

SELECT
    2 AS src_order,
    pk_id,
    "Suffered_from",
    service_title,
    sub_service,
    pt_gender
FROM public.hhcweb_old_hhc_patient_suffering_from
ORDER BY src_order, pk_id;
"""

# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------
df = pd.read_sql(QUERY, engine)

df.rename(columns={
    "pt_gender": "Gender",
    "sub_service": "recommomded_service"
}, inplace=True)

df["Suffered_from"] = df["Suffered_from"].fillna("unknown").astype(str)
df["Gender"] = df["Gender"].fillna("Unknown")

# ------------------------------------------------------------
# TEXT CLEANING (OPTIMIZED & SAFE)
# ------------------------------------------------------------
spell = SpellChecker()

def clean_text_db(text):
    """Fast cleaning for DB text (NO spellcheck)"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return text

def clean_text_query(text):
    """Spellcheck ONLY user input"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    words = text.split()
    corrected = [spell.correction(w) or w for w in words]
    return " ".join(corrected)

# ------------------------------------------------------------
# PRE-CLEAN DB TEXT ONCE (BIG SPEED GAIN)
# ------------------------------------------------------------
df["Suffered_clean"] = df["Suffered_from"].apply(clean_text_db)

# ------------------------------------------------------------
# SERVICE NAME MARATHI MAP
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
# ENCODING
# ------------------------------------------------------------
le_gender = LabelEncoder()
df["Gender_enc"] = le_gender.fit_transform(df["Gender"])

le_service = LabelEncoder()
df["service_enc"] = le_service.fit_transform(df["service_title"])

le_sub = LabelEncoder()
df["subservice_enc"] = le_sub.fit_transform(df["recommomded_service"])

# ------------------------------------------------------------
# TEXT VECTORIZATION
# ------------------------------------------------------------
vectorizer = TfidfVectorizer(max_features=3000)
X_text = vectorizer.fit_transform(df["Suffered_clean"])

X = pd.concat(
    [
        pd.DataFrame(X_text.toarray()),
        df[["Gender_enc"]].reset_index(drop=True)
    ],
    axis=1
)
X.columns = X.columns.astype(str)

# ------------------------------------------------------------
# TRAIN MODELS
# ------------------------------------------------------------
service_model = RandomForestClassifier(n_estimators=200, random_state=42)
service_model.fit(X, df["service_enc"])

sub_model = RandomForestClassifier(n_estimators=200, random_state=42)
sub_model.fit(X, df["subservice_enc"])

# ------------------------------------------------------------
# SIMILAR CASE RETRIEVAL
# ------------------------------------------------------------
symptom_vectors = vectorizer.transform(df["Suffered_clean"])

def recommend_services(suffered_from, top_k=5):
    q_vec = vectorizer.transform([suffered_from])
    sims = cosine_similarity(q_vec, symptom_vectors)[0]

    df_temp = df.copy()
    df_temp["similarity"] = sims

    top_cases = df_temp.sort_values("similarity", ascending=False).head(50)

    recs = (
        top_cases
        .groupby(["service_title", "recommomded_service"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )

    return recs.head(top_k)

# ------------------------------------------------------------
# EXPLANATION + CARE PLAN
# ------------------------------------------------------------
def generate_explanation_and_plan(symptom):
    symptom = symptom.lower()
    if any(k in symptom for k in ["stroke", "paralysis", "weakness", "nerve"]):
        explanation = "Symptoms indicate neurological impairment affecting mobility."
        care_plan = [
            "Nursing care: Medication and vitals monitoring",
            "Physiotherapy: Mobility and balance training",
            "Occupational therapy support",
            "Mobility equipment",
            "Fall prevention monitoring"
        ]
    elif any(k in symptom for k in ["fracture", "bone", "knee", "back", "pain"]):
        explanation = "Symptoms suggest an orthopedic or musculoskeletal condition."
        care_plan = [
            "Pain management nursing care",
            "Physiotherapy for joint movement",
            "Support braces or aids",
            "Daily mobility assistance",
            "Recovery monitoring"
        ]
    else:
        explanation = "General health condition requiring monitoring and assistance."
        care_plan = [
            "Vitals monitoring",
            "Daily activity assistance",
            "Diet and hydration care",
            "Routine follow-ups",
            "Doctor consultation if needed"
        ]
    return explanation, care_plan

# ------------------------------------------------------------
# MARATHI OUTPUT
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
    clean_symptom = clean_text_query(symptom_input)

    gender_norm = (
        "Male" if gender_input.lower().startswith("m")
        else "Female" if gender_input.lower().startswith("f")
        else "Unknown"
    )
    gender_enc = le_gender.transform([gender_norm])[0]

    X_new_text = vectorizer.transform([clean_symptom])
    X_new = pd.concat(
        [
            pd.DataFrame(X_new_text.toarray()),
            pd.DataFrame([[gender_enc]], columns=["Gender_enc"])
        ],
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
    other_services = []
    for _, r in recommendations.iterrows():
        s_text = f"{r['service_title']} → {r['recommomded_service']}"
        other_services.append(s_text)
        print(f"- {s_text}")

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
