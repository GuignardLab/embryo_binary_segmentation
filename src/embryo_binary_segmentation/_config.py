DATA_PARAMS = {
    "data_path":"/home/polinasoloveva/Data/",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 256, 256],
    "augmentations":True
}  

FINE_TUNING = {
    "upload_model_path":None,
    "old_steps":None,
}


TRAINING_PARAMS = {
    "loss":"bce",
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 200,
    "save_model_path":"/home/polinasoloveva/Models/Test",
    "fine_tuning":False,
    "save_each":True,
}

TEST_PARAMS = {
    "data_path":"/home/polinasoloveva/Data/",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 512, 512],
    "batch_size": 2,
    "load_model_path":"/home/polinasoloveva/Models/Test/redo.model",
    "load_csv_path":"/home/polinasoloveva/Models/Test/redo.csv"
}

