from helper import Helper
from preprocess import activity_test_data, place_test_data

print("-----------activity model evaluation--------------")
Helper.evaluate_ner_model(
    model_path="./activity_output/model-best", eval_data=activity_test_data
)
print("--------------------------------------------------")
print("-----------place model evaluation--------------")
Helper.evaluate_ner_model(
    model_path="./place_output/model-best", eval_data=place_test_data
)
print("--------------------------------------------------")
