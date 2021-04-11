from preprocess import DataPipeline
from syncnet.models.visual_model import VisualModel
from syncnet.models.audio_model import AudioModel

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', required=True, help='Directory of anime videos.')
# parser.add_argument('--mode', required=True, choices=["train"])
# args = parser.parse_args()

visual_model = VisualModel()
audio_model = AudioModel()

def train(visual_tensors, audio_tensors):

    for visual_tensors_1_video, audio_tensors_1_video in zip(visual_tensors, audio_tensors):
        for visual_tensor, audio_tensor in zip(visual_tensors_1_video, audio_tensors_1_video):
            out_a, out_A = audio_model.call(audio_tensor.cuda())
            out_v, out_V = visual_model.call(visual_tensor.cuda())

            visual_model.forward_lip(visual_tensor)
            audio_model.forward_aud(audio_tensor)



if __name__ == "__main__":
    frame_tensors, audio_tensors = DataPipeline("/Users/aparna/Downloads/converted/").get_data()
    # if args.mode == "train":
    train(frame_tensors, audio_tensors)
