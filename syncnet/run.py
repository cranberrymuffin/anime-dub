import argparse
from preprocess import DataPipeline
from syncnet import VisualModel, AudioModel

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', required=True, help='Directory of anime videos.')
parser.add_argument('--mode', required=True, choices=["train"])
args = parser.parse_args()


def train(frame_tensors, audio_tensors):
    visual_model = VisualModel()
    audio_model = AudioModel()

    for frame_tensor in frame_tensors:
        nn.forward_lip(frame_tensor)
    for audio_tensor in audio_tensors:
        nn.forward_aud(audio_tensor)



if __name__ == "__main__":
    frame_tensors, audio_tensors = Dataset(args.data_dir).get_data()
    if args.mode == "train":
        train(frame_tensors, audio_tensors)
