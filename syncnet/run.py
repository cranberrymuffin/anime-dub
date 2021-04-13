from preprocess import DataPipeline
from models.sync_net_model import sync_net

if __name__ == "__main__":

    visual_inputs, audio_inputs, is_synced_labels = DataPipeline("/Users/aparna/Downloads/converted").get_data()
