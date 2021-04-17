from preprocess import DataPipeline
from models.sync_net_model import SyncNet
from os import path
from tensorflow import keras

def split_data(visual_inputs, audio_inputs, is_synced_labels):
    train_idx = int(len(visual_inputs) * 0.70)

    train_visual_inputs = visual_inputs[0:train_idx]
    train_audio_inputs = audio_inputs[0:train_idx]
    train_labels = is_synced_labels[0:train_idx]

    test_visual_inputs = visual_inputs[train_idx:]
    test_audio_inputs = audio_inputs[train_idx:]
    test_labels = is_synced_labels[train_idx:]

    split_data = []
    split_data.append((train_visual_inputs, train_audio_inputs, train_labels))
    split_data.append((test_visual_inputs, test_audio_inputs, test_labels))

    return split_data

if __name__ == "__main__":
    sync_net = SyncNet()
    visual_inputs, audio_inputs, is_synced_labels = DataPipeline("data/converted").get_data()

    split_data = split_data(visual_inputs, audio_inputs, is_synced_labels)
    (train_visual_inputs, train_audio_inputs, train_labels) = split_data[0]
    (test_visual_inputs, test_audio_inputs, test_labels) = split_data[1]

    if path.exists("checkpoints/model.h5"):
        model = keras.models.load_model("checkpoints/model.h5")
        model.evaluate([test_visual_inputs, test_audio_inputs], test_labels)
    else:
        sync_net.train(train_visual_inputs, train_audio_inputs, train_labels)
        sync_net.evaluate(test_visual_inputs, test_audio_inputs, test_labels)

    sync_net.save_model("checkpoints/model.h5")