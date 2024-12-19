import joblib
import os
import pandas as pd
import config
from preprocessing_utils import post_process


def predict(model):
    x_test = joblib.load(os.path.join(config.MODEL_OUTPUT, f"test_label_encoded.pkl"))
    sample = pd.read_csv(config.SAMPLE_FILE)

    predictions = None
    for fold in range(config.N_FOLDS):
        clf = joblib.load(os.path.join(config.MODEL_OUTPUT, f"baseline_{model}_{fold}.pkl"))
        preds = clf.predict(x_test)
        
        if fold == 0:
            predictions = preds
        else:
            predictions += preds

    predictions = predictions / config.N_FOLDS

    sample["quantite_vendue"] = post_process(predictions)
    return sample


if __name__ == "__main__":
    #models = ["hist", "cat", "gbm", "lgbm", "xgb"]
    submission = predict(config.MODEL)
    submission.to_csv(f"{config.FILES_OUTPUT}{config.MODEL}_baseline_submission.csv", index=False)
    print("SUBMISSION FILE CREATED SUCCESSFULLY!!!")