import os
import gin
import tensorflow.compat.v2 as tf
import pickle

def find_model_dir(dir_name):
    print(f"=== Debug: Searching for model directory in {dir_name} ===")
    for root, dirs, filenames in os.walk(dir_name):
        for filename in filenames:
            if filename.endswith(".gin") and not filename.startswith("."):
                print(f"Model directory found: {root}")
                return root
    print("Model directory not found.")
    return None

def load_model(model_name='Violin', model_dir='./pretrained/'):
    model_mapping = {
        'Violin': 'solo_violin_ckpt',
        'Flute': 'solo_flute_ckpt',
        'Tenor_Saxophone': 'solo_tenor_saxophone_ckpt',
        'Trumpet': 'solo_trumpet_ckpt',
        'Flute2':'solo_flute2_ckpt'
    }

    if model_name in model_mapping:
        model_dir = os.path.join(model_dir, model_mapping[model_name])
        model_dir = find_model_dir(model_dir)
        if model_dir is None:
            print(f"Error: Model directory for {model_name} not found in {model_dir}.")
            return None, None
        
        gin_file = os.path.join(model_dir, 'operative_config-0.gin')
        print(f"gin file path: {gin_file}")
    else:
        print("Model upload functionality is not supported in this script.")
        return None, None

    dataset_stats_file = os.path.join(model_dir, 'dataset_statistics.pkl')
    print(f"Dataset stats file path: {dataset_stats_file}")
    DATASET_STATS = None
    try:
        if tf.io.gfile.exists(dataset_stats_file):
            print(f"Loading dataset statistics from {dataset_stats_file}")
            with tf.io.gfile.GFile(dataset_stats_file, 'rb') as f:
                DATASET_STATS = pickle.load(f)
        else:
            print(f"Dataset stats file does not exist: {dataset_stats_file}")
    except Exception as err:
        print(f'Loading dataset statistics from pickle failed: {err}.')

    try:
        with gin.unlock_config():
            print(f"Parsing gin config from {gin_file}")
            gin.parse_config_file(gin_file, skip_unknown=True)
    except Exception as e:
        print(f"Error parsing gin file: {e}")

    return model_dir, DATASET_STATS
