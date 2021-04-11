import argparse
from preprocess import DataPipeline
from syncnet import VisualModel, AudioModel

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', required=True, help='Directory of anime videos.')
# parser.add_argument('--mode', required=True, choices=["train"])
# args = parser.parse_args()


def train(frame_tensors, audio_tensors):

    visual_model = VisualModel()
    audio_model = AudioModel()

    for frame_tensors_1_video, audio_tensors_1_video in zip(frame_tensors, audio_tensors):
        for frame_tensor, audio_tensor in zip(frame_tensors_1_video, audio_tensors_1_video):
            visual_model.forward_lip(frame_tensor)
            audio_model.forward_aud(audio_tensor)



if __name__ == "__main__":
    frame_tensors, audio_tensors = DataPipeline("/Users/aparna/Downloads/converted/").get_data()
    # if args.mode == "train":
    train(frame_tensors, audio_tensors)
