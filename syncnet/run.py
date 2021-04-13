from preprocess import DataPipeline
from models.visual_model import VisualModel
from models.audio_model import AudioModel
import tensorflow as tf
import numpy as np
from time import process_time

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
def contrastive_loss(audio_to_visual_output_distances, is_synced_labels, margin):
    N = len(audio_to_visual_output_distances)  # N is batch size
    loss = 0.0

    for i in range(N):
        distance = audio_to_visual_output_distances[i]
        is_synced = is_synced_labels[i]
        loss = loss + ((is_synced * pow(distance, 2)) + ((1 - is_synced) * pow(max(margin - distance, 0), 2)))

    return loss


def train(visual_tensors, audio_tensors, is_synced_labels):
    assert (len(visual_tensors) == len(audio_tensors))

    distances = np.empty(visual_tensors)

    for visual_tensors_1_video, audio_tensors_1_video in zip(visual_tensors, audio_tensors):
        for idx, visual_tensor, audio_tensor in enumerate(zip(visual_tensors_1_video, audio_tensors_1_video)):
            audio_output = audio_model.call(audio_tensor.cuda())
            visual_output = visual_model.call(visual_tensor.cuda())
            distances[idx] = euclidean_distance(audio_output, visual_output)

    contrastive_loss(distances, is_synced_labels, max(distances))


if __name__ == "__main__":

    try:
        print("Retrieving saved Data...")
        start = process_time()
        visual_data = np.load("data/frames.npy")
        audio_data = np.load("data/audio.npy")
        end = process_time()
        print("Data retrieval completed in " + str(end - start) + " seconds!")

    except:
        print("Could not find saved dataset, generating and saving new dataset...")

        print("Preprocessing Raw Video Data...")
        start = process_time()
        visual_data, audio_data = DataPipeline("path to anime lrs2 dataset here").get_data()

        visual_data = np.array(visual_data)
        audio_data = np.array(audio_data)

        end = process_time()
        print("Preprocessing completed in " + str(end - start) + " seconds!")

        # For saving as numpy arrays
        np.save("data/frames", visual_data)
        np.save("data/audio", audio_data)

        print("Saved data!")

    assert (visual_data.shape[0] == audio_data.shape[0])
    print("Number of Data Points: ", visual_data.shape[0])
    print("Shape of Visual Data: ", visual_data.shape)
    print("Shape of Audio Data: ", audio_data.shape)
