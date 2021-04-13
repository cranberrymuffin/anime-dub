from preprocess import DataPipeline
from models.visual_model import VisualModel
from models.audio_model import AudioModel
import tensorflow as tf
import numpy as np

# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', required=True, help='Directory of anime videos.')
# parser.add_argument('--mode', required=True, choices=["train"])
# args = parser.parse_args()

visual_model = VisualModel()
audio_model = AudioModel()

# euclidean distance between audio/visual output pairs
# they should be 0 given training data has perfect synchronicity
def euclidean_distance(audio_output, visual_output):
    return tf.norm(tf.subtract(audio_output, visual_output))

# ??? what is margin
# is_synced_labels - binary similarity metric, array of labels stating whether the audio/visual output pair at index i
#                    is synced or not.
def constrative_loss(audio_to_visual_output_distances, is_synced_labels, margin):
    N = len(audio_to_visual_output_distances) # N is batch size
    loss = 0.0

    for i in range(N):
        distance = audio_to_visual_output_distances[i]
        is_synced = is_synced_labels[i]
        loss = loss + ((is_synced * pow(distance, 2)) + ((1-is_synced) * pow(max(margin - distance, 0), 2)))

    return loss

def train(visual_tensors, audio_tensors, is_synced_labels):
    assert(len(visual_tensors) == len(audio_tensors))

    distances = np.empty(visual_tensors)

    for visual_tensors_1_video, audio_tensors_1_video in zip(visual_tensors, audio_tensors):
        for idx, visual_tensor, audio_tensor in enumerate(zip(visual_tensors_1_video, audio_tensors_1_video)):
            audio_output = audio_model.call(audio_tensor.cuda())
            visual_output = visual_model.call(visual_tensor.cuda())
            distances[idx] = euclidean_distance(audio_output, visual_output)

    constrative_loss(distances, is_synced_labels, max(distances))

if __name__ == "__main__":
    try:
        frames_dataset = tf.data.experimental.load("saved_data/frames",
            tf.TensorSpec(shape=(1, 1, 9, 5, 224, 224, 1), dtype=tf.float32, name=None))
        audio_dataset = tf.data.experimental.load("saved_data/audio",
            tf.TensorSpec(shape=(1, 1, 9, 13, 20, 1), dtype=tf.float32, name=None))

        # For loading numpy arrays
        # np.load("saved_data_np/frames")
        # np.load("saved_data_np/audio")

        print("Retrieving saved dataset...")
    except:
        print("Could not find saved dataset, generating and saving new dataset...")

        frames, audio = DataPipeline("data/converted/").get_data()

        # For saving as numpy arrays
        # np.save("saved_data_np/frames", frames)
        # np.save("saved_data_np/audio", audio)

        print("pre-processed the data")

        frames_dataset = tf.data.Dataset.from_tensors(frames)
        audio_dataset = tf.data.Dataset.from_tensors(audio)

        tf.data.experimental.save(frames_dataset, "saved_data/frames")
        tf.data.experimental.save(audio_dataset, "saved_data/audio")

    # To check the contents of the dataset, like this
    # for data in frames_dataset:
    #     print(data)

    # train(frame_tensors, audio_tensors)
