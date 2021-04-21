from speech2vid.models.speech2vid_model import Speech2Vid
import argparse
import time
import cv2
import preprocess
import numpy as np
import random


def current_milli_time():
    return round(time.time() * 1000)


parser = argparse.ArgumentParser(description='Speech2Vid Run Parameters')

parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='run mode')

parser.add_argument('--data-dir', type=str, required=False, default=None, help='path to data directory')
parser.add_argument('--load-from', type=str, required=False, default=None,
                    help='path to checkpoint file to load model from')
parser.add_argument('--sync_net_path', type=str, required=True, default=None,
                    help='path to checkpoint file to load syncnet model from')
parser.add_argument('--load-limit', type=int, required=False, default=30000,
                    help='limit on training data points to load')

parser.add_argument('--val-video', type=str, default=None, required=False, help='video to validate')
parser.add_argument('--val-visual', type=str, default=None, required=False, help='no audio video to validate')
parser.add_argument('--val-audio', type=str, default=None, required=False, help='audio to validate')

args = parser.parse_args()


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


def augment_data(visual_inputs, audio_inputs):
    for input_idx, mfcc in enumerate(audio_inputs):
        assert (audio_inputs[input_idx].shape == (13, 20, 1))

    augmented_visual_inputs = []
    labels = []

    for input_idx, set_of_5_frames in enumerate(visual_inputs):
        resized_set_of_5_frames = np.empty((5, 112, 112, 3))
        for idx, frame in enumerate(set_of_5_frames):
            resized_set_of_5_frames[idx] = cv2.resize(frame, (112, 112))

        shuffled_frames = np.copy(resized_set_of_5_frames)
        np.random.shuffle(shuffled_frames)
        augmented_visual_inputs.append(np.concatenate(shuffled_frames, axis=2))
        labels.append(np.concatenate(resized_set_of_5_frames, axis=2))
        assert (augmented_visual_inputs[input_idx].shape == (112, 112, 15))
        assert (labels[input_idx].shape == (112, 112, 15))
    return np.array(augmented_visual_inputs), audio_inputs, audio_inputs


if __name__ == "__main__":
    speech2vid_net = Speech2Vid(args.load_from, args.sync_net_path)

    if args.mode == "train" or args.mode == "test":
        visual_inputs, audio_inputs, is_synced_labels = preprocess.DataPipeline(args.data_dir,
                                                                                args.load_limit).get_data()

        visual_inputs = visual_inputs[0:5000]
        audio_inputs = audio_inputs[0:5000]
        is_synced_labels = is_synced_labels[0:5000]

        visual_inputs, audio_inputs, labels = augment_data(visual_inputs, audio_inputs)
        split_data = split_data(visual_inputs, audio_inputs, labels)
        (train_visual_inputs, train_audio_inputs, train_labels) = split_data[0]
        (test_visual_inputs, test_audio_inputs, test_labels) = split_data[1]

    if args.mode == "train":
        speech2vid_net.train(train_visual_inputs, train_audio_inputs, train_labels)
        speech2vid_net.predict([random.choice(test_audio_inputs), random.choice(test_visual_inputs)])
        speech2vid_net.save_model("speech2vid/checkpoints/" + str(current_milli_time()) + "_model.h5")
    elif args.mode == "test":
        if args.load_from is None:
            print("Must specify a checkpoint file in test mode. Please specify one with the \'--load-from\' flag")
            exit()
        speech2vid_net.evaluate(test_visual_inputs, test_audio_inputs, test_labels)
    else:
        if args.load_from is None:
            print("Must specify a checkpoint file in validation mode. Please specify one with the \'--load-from\' flag")
            exit()

        if args.val_video is not None:
            if args.val_visual is not None or args.val_audio is not None:
                print("WARNING: Validation will only evaluate the synchronicity of the visuals and audio found at the "
                      ".mp4 file specified in the --val_video path. Inputs passed to --args.val_visual and "
                      "--args.val_audio will be ignored.")

            print("Evaluating synchronicity of input video at " + args.val_video)
            # TODO EVAL

        elif not (args.val_visual is not None and args.val_audio is not None):
            print("ERROR: --val_visual .mp4 path must be passed with a --val_audio .wav path. To pass a single video "
                  "containing audio please use the --args.val_video flag.")
            exit()

            print(
                "Evaluating synchronicity of input visual at " + args.val_video + " with input audio at " + args.val_audio)
            # TODO EVAL
