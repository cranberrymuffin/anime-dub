from preprocess import DataPipeline
import os
import tensorflow as tf
from models.visual_model import VisualModel
from models.audio_model import AudioModel

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


if __name__ == "__main__":
    try:
        frames_dataset = tf.data.experimental.load("saved_data/frames",
            tf.TensorSpec(shape=(1, 1, 9, 5, 224, 224, 1), dtype=tf.float32, name=None))
        audio_dataset = tf.data.experimental.load("saved_data/audio",
            tf.TensorSpec(shape=(1, 1, 9, 13, 20, 1), dtype=tf.float32, name=None))

        print("Retrieving saved dataset...")
    except:
        print("Could not find saved dataset, generating and saving new dataset...")

        frames, audio = DataPipeline("data/converted/").get_data()

        frames_dataset = tf.data.Dataset.from_tensors(frames)
        audio_dataset = tf.data.Dataset.from_tensors(audio)

        tf.data.experimental.save(frames_dataset, "saved_data/frames")
        tf.data.experimental.save(audio_dataset, "saved_data/audio")

    # To check the contents of the dataset, like this
    # for data in frames_dataset:
    #     print(data)

    # train(frame_tensors, audio_tensors)
