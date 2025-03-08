from helper import Helper


def single_label_preprocessor(data_path, label, prefix):

    data = Helper.load_tsv(data_path)
    spacy_format_data = Helper.spacy_format_converter(data, label=label)
    train_data, val_data, test_data = Helper.train_val_test_splitter(spacy_format_data)
    training_dataset_path = "./" + prefix + "_training_data.spacy"
    val_dataset_path = "./" + prefix + "_eval_data.spacy"
    Helper.save_spacy_binary_format_data(train_data, training_dataset_path)
    Helper.save_spacy_binary_format_data(val_data, val_dataset_path)
    return test_data, training_dataset_path, val_dataset_path


activity_test_data, activity_training_dataset_path, activity_val_dataset_path = (
    single_label_preprocessor(
        data_path="Dataset.tsv", label="ACTIVITY OR CAUSE", prefix="activity"
    )
)
place_test_data, place_training_dataset_path, place_val_dataset_path = (
    single_label_preprocessor(data_path="Dataset.tsv", label="PLACE", prefix="place")
)
print("activity_training_dataset_path:", activity_training_dataset_path)
print("activity_val_dataset_path:", activity_val_dataset_path)
print("place_training_dataset_path:", place_training_dataset_path)
print("place_val_dataset_path:", place_val_dataset_path)
