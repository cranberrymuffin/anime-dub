from preprocess import DataPipeline
from models.sync_net_model import SyncNet

if __name__ == "__main__":

    video_inputs, audio_inputs, is_synced_labels = DataPipeline("data/converted").get_data()
    sync_net = SyncNet()
    sync_net.train(video_inputs, audio_inputs, is_synced_labels)
    sync_net.save_model("saved_model/model.h5")