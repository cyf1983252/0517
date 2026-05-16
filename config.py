import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

TWITTER_FILE = os.path.join(DATA_DIR, "twitter_10k_twitter_userdata.csv")
WEIBO_FILE = os.path.join(DATA_DIR, "weibo_10k_weibo_users.csv")

TARGET_SIZE = 10000
MISSING_RATE = 0.20
ANOMALY_RATE = 0.08

TFIDF_MAX_FEATURES = 3000
TFIDF_SVD_DIM = 400
# TF-IDF 400d + topic9+occ9+sub18 + attr1 + num14 = 451 → 454
STATIC_EMBED_DIM = 454
DYNAMIC_EMBED_DIM = 64
SOCIAL_EMBED_DIM = 64
FUSION_DIM = 582         # 454 + 64 + 64

ATTRIBUTE_TASKS = [
    "interest_domain", "interest_sub", "active_period",
    "occupation_type", "age_group", "account_scale", "content_language",
]

BATCH_SIZE = 128
EPOCHS = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

TARGET_RECALL = 0.75
TARGET_PRECISION = 0.80

RANDOM_SEED = 42
