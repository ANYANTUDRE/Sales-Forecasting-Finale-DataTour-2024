DATA_INPUT    = "../input/"
MODEL_OUTPUT  = "../models/"
FILES_OUTPUT  = "../artifacts/"

ORIG_TRAIN_FILE = "https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/train.csv"
TRAINING_FILE   = "../input/train_folds.csv"
TEST_FILE       = "https://raw.githubusercontent.com/dataafriquehub/donnee_vente/refs/heads/main/submission.csv"
SAMPLE_FILE     = "../input/sample_submission.csv"

N_FOLDS = 5
MODEL   = "voting_reg"
TRAINING_SCRIPT = "train_ohe_linear.py" if MODEL in ("logistic", "svm", "lasso_model") else "train_label_tree.py"
TARGET_COL      = "quantite_vendue_mapped"   #"quantite_vendue"
#COLS_TO_DROP   = ["Name", "City", "Study Satisfaction", "Academic Pressure", "CGPA", "Profession", "Work Pressure", "Job Satisfaction", "Dietary Habits", "Financial Stress", "Degree", "Sleep Duration", "Dietary Habits", ]
NUM_COLS        = ['prix_unitaire', 'promotion', 'jour_ferie', 'weekend', 'stock_disponible']