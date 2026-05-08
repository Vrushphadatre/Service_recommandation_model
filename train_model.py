# ============================================================
# STEP 1: Imports + DB connection
# ============================================================

import pandas as pd
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from scipy.sparse import hstack



DB_CONFIG = {
    "dbname": "HHC_main_2024",
    "user": "postgres",
    "password": "HHC%40109",
    "host": "192.168.1.109",
    "port": "5432"
}

engine = create_engine(
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ============================================================
# STEP 2: Load data
# ============================================================

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

df = pd.read_sql(QUERY, engine)

df.rename(columns={
    "pt_gender": "Gender",
    "sub_service": "recommomded_service"
}, inplace=True)

df["Suffered_from"] = df["Suffered_from"].fillna("unknown").astype(str)
df["Gender"] = df["Gender"].fillna("Unknown")

# ============================================================
# STEP 3: Encoding
# ============================================================

le_gender = LabelEncoder()
df["Gender_enc"] = le_gender.fit_transform(df["Gender"])

le_service = LabelEncoder()
df["service_enc"] = le_service.fit_transform(df["service_title"])

le_sub = LabelEncoder()
df["subservice_enc"] = le_sub.fit_transform(df["recommomded_service"])

# ============================================================
# STEP 4: TF-IDF (SAFE)
# ============================================================

vectorizer = TfidfVectorizer(
    max_features=3000,
    dtype="float32",
    stop_words="english"
)

X_text = vectorizer.fit_transform(df["Suffered_from"])

# Convert Gender to sparse & stack
X_gender = df["Gender_enc"].values.reshape(-1, 1)
X = hstack([X_text, X_gender])


# ============================================================
# STEP 4.1: SAVE MEDICAL VOCABULARY (IMPORTANT)
# ============================================================

medical_vocab = set(vectorizer.get_feature_names_out())


# ============================================================
# # STEP 5: Train models
# # ============================================================

from sklearn.linear_model import LogisticRegression

service_model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    solver="lbfgs"
)
service_model.fit(X, df["service_enc"])

sub_model = LogisticRegression(
    max_iter=2000,
    n_jobs=-1,
    solver="lbfgs"
)
sub_model.fit(X, df["subservice_enc"])


# ============================================================
# STEP 6: Save models (MEMORY SAFE)
# ============================================================

# joblib.dump(
#     {
#         "service_model": service_model,
#         "sub_model": sub_model,
#         "vectorizer": vectorizer,
#         "le_gender": le_gender,
#         "le_service": le_service,
#         "le_sub": le_sub
#     },
#     "hhc_models.pkl",
#     compress=3
# )


joblib.dump(
    {
        "service_model": service_model,
        "sub_model": sub_model,
        "vectorizer": vectorizer,
        "le_gender": le_gender,
        "le_service": le_service,
        "le_sub": le_sub,
        "medical_vocab": medical_vocab   
    },
    "hhc_models.pkl",
    compress=3
)


print("Training completed & models saved safely")


