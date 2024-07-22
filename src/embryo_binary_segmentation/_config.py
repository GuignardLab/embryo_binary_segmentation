DATA_PARAMS = {
    "data_path":"/home/polinasoloveva/Data/",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 256, 256],
    "augmentations":True
}  

FINE_TUNING = {
    "upload_model_path":"/home/polinasoloveva/Models/Test/best.model",
    "old_steps":2,
}


TRAINING_PARAMS = {
    "loss":"bce",
    "learning_rate": 1e-4,
    "batch_size": 8,
    "epochs": 200,
    "save_model_path":"/home/polinasoloveva/Models/Test",
    "fine_tuning":True,
    "save_each":True,
}

TEST_PARAMS = {
    "data_path":"/home/polinasoloveva/Data/Test_visualization",
    "binarize":False,
    "target_size":[64, 512, 512],
    "patch_size":[32, 512, 512],
    "batch_size": 2,
    "load_model_path":"/home/polinasoloveva/Models/Test/best.model",
    "load_csv_path":"/home/polinasoloveva/Models/Test/training_data.csv"
}

PRED_PARAMS = {
    "data_path":"/home/polinasoloveva/Data/Val/e7_woon",
    "final_load_model_path": "/home/polinasoloveva/Models/UNet_Astec_focal/best.model",
    "batch_size": 2,
    "save_pred_path":"/home/polinasoloveva/Data/Test_visualization/PRED/"
}