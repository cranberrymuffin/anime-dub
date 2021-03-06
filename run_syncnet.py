from syncnet.models.sync_net_model import SyncNet
import argparse
import time
import cv2
import numpy as np
import preprocess


def current_milli_time():
    return round(time.time() * 1000)


parser = argparse.ArgumentParser(description='SyncNet Run Parameters')

parser.add_argument('--mode', type=str, required=True, choices=["train", "test", "val"], help='run mode')

parser.add_argument('--data-dir', type=str, required=False, default=None, help='path to data directory')
parser.add_argument('--load-from', type=str, required=False, default=None,
                    help='path to checkpoint file to load model from')
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
    for audio_input in audio_inputs:
        assert (audio_input.shape == (13, 20, 1))

    blw_inputs = np.empty((visual_inputs.shape[0], 5, 224, 224, 1))
    for input_idx, visual_input in enumerate(visual_inputs):
        for frame_idx, frame in enumerate(visual_input):
            blw_mouth = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[frame.shape[0] // 2:frame.shape[0], :]
            blw_mouth = cv2.resize(blw_mouth, (224, 224))
            blw_inputs[input_idx][frame_idx] = np.expand_dims(blw_mouth, axis=-1)
        assert (blw_inputs[input_idx].shape == (5, 224, 224, 1))

        return blw_inputs, audio_inputs


if __name__ == "__main__":
    sync_net = SyncNet(args.load_from)

    if args.mode == "train" or args.mode == "test":
        visual_inputs, audio_inputs, is_synced_labels = preprocess.DataPipeline(args.data_dir,
                                                                                args.load_limit).get_data()

        visual_inputs = visual_inputs[0:5000]
        audio_inputs = audio_inputs[0:5000]
        is_synced_labels = is_synced_labels[0:5000]

        visual_inputs, audio_inputs = augment_data(visual_inputs, audio_inputs)
        split_data = split_data(visual_inputs, audio_inputs, is_synced_labels)
        (train_visual_inputs, train_audio_inputs, train_labels) = split_data[0]
        (test_visual_inputs, test_audio_inputs, test_labels) = split_data[1]

    if args.mode == "train":
        sync_net.train(train_visual_inputs, train_audio_inputs, train_labels)
        sync_net.evaluate(test_visual_inputs, test_audio_inputs, test_labels)
        sync_net.save_model("syncnet/checkpoints/" + str(current_milli_time()) + "_model.h5")
    elif args.mode == "test":
        if args.load_from is None:
            print("Must specify a checkpoint file in test mode. Please specify one with the \'--load-from\' flag")
            exit()
        sync_net.evaluate(test_visual_inputs, test_audio_inputs, test_labels)
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
