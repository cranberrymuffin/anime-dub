from preprocess import DataPipeline
from models.sync_net_model import SyncNet

def split_data(video_inputs, audio_inputs, is_synced_labels):
    train_idx = int(len(video_inputs) * 0.90)

    train_video_inputs = video_inputs[0:train_idx]
    train_audio_inputs = audio_inputs[0:train_idx]
    train_labels = is_synced_labels[0:train_idx]

    test_video_inputs = video_inputs[train_idx:]
    test_audio_inputs = audio_inputs[train_idx:]
    test_labels = is_synced_labels[train_idx:]

    split_data = []
    split_data.append((train_video_inputs, train_audio_inputs, train_labels))
    split_data.append((test_video_inputs, test_audio_inputs, test_labels))

    return split_data

if __name__ == "__main__":
    video_inputs, audio_inputs, is_synced_labels = DataPipeline("data/converted").get_data()

    split_data = split_data(video_inputs, audio_inputs, is_synced_labels)
    (train_video_inputs, train_audio_inputs, train_labels) = split_data[0]
    (test_video_inputs, test_audio_inputs, test_labels) = split_data[1]

    sync_net = SyncNet()
    sync_net.train(train_video_inputs, train_audio_inputs, train_labels)
    sync_net.test(test_video_inputs, test_audio_inputs, test_labels)

    sync_net.save_model("saved_model/model.h5")