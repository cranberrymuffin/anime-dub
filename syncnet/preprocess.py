import cv2
import librosa
import python_speech_features
import numpy as np
import os
import subprocess
from time import process_time

class DataPipeline:
    def __init__(self, data_path):
        self.data_path = data_path
        self.visual_inputs = []
        self.audio_inputs = []
        self.is_synced_labels = []
    """
    * The network ingests 0.2-second clips of both audio and video inputs.
    This function...
    1) Splits the video at video_path into segments of 0.2 seconds each
    2) Separates the visual and audio for each clip
    3) Formats the visual and audio using format_input_visual and format_input_audio
    4) Appends this data to instance variables self.audio_inputs self.visual_inputs
    """

    def format_video(self, video_path):
        tmp_video_path, tmp_audio_path = self.get_and_create_tmp_video_audio_files(video_path)
        try:
            frames, mfccs = self.trim_mfcc_and_visual(
                self.format_input_audio(tmp_audio_path),
                self.format_input_visual(tmp_video_path)
            )

            self.visual_inputs.extend(frames)
            self.audio_inputs.extend(mfccs)

        finally:
            self.delete_file(tmp_video_path)
            self.delete_file(tmp_audio_path)

    @staticmethod
    def group_mfccs(mfcc):
        # get the number of columns in the mfcc
        num_cols = mfcc.shape[1]

        # make sure there are more than 20 columns (video is longer that 0.2 seconds)
        # 20 columns == 0.2 seconds
        assert (num_cols > 20)

        # determine the number of columns to remove to make the audio an even multiple of 0.2 seconds
        to_trim = num_cols % 20

        # if there are columns to trim, trim them
        if to_trim > 0:
            mfcc = mfcc[:, :-to_trim]

        # group mfcc cols into sets of 20
        mfcc_groups = np.hsplit(mfcc, mfcc.shape[1] / 20)
        mfcc_groups = [np.expand_dims(mfcc, axis=-1) for mfcc in mfcc_groups]

        for mfcc_group in mfcc_groups:
            assert (mfcc_group.shape == (13, 20, 1))

        return mfcc_groups

    @staticmethod
    def group_frames(frames):
        # get the number of frames
        num_frames = frames.shape[0]

        # make sure there are more than 5 frames (video is longer that 0.2 seconds)
        # 5 frames == 0.2 seconds
        assert (num_frames > 5)

        # determine the number of frames to remove to make the video an even multiple of 0.2 seconds
        to_trim = num_frames % 5

        # if there are frames to trim, trim them
        if to_trim > 0:
            frames = frames[:-to_trim, :, :]

        # group frames into sets of 5
        frames_groups = np.split(frames, frames.shape[0] / 5)
        frame_groups = [np.expand_dims(group, axis=-1) for group in frames_groups]

        # for 3D convolution, 224 x 224 image, depth of 5, 1 color channel
        for frame_group in frame_groups:
            assert (frame_group.shape == (5, 224, 224, 1))

        return frame_groups

    def trim_mfcc_and_visual(self, mfcc_groups, frame_groups):
        num_groups = min(len(mfcc_groups), len(frame_groups))

        mfcc_groups = mfcc_groups[:num_groups]
        frame_groups = frame_groups[:num_groups]

        assert (len(mfcc_groups) == len(frame_groups))

        return mfcc_groups, frame_groups

    """
    Converts a audio clip to MFCC map of N x 13 x 20 x 1
    13 MFCC features representing powers at different frequency bins.
    Sampled at rate of 100Hz, gives 20 time steps for a 0.2-second input signal.
    """

    def format_input_audio(self, wav_file):
        wav, sample_rate = librosa.load(wav_file, sr=100)
        assert (sample_rate == 100)
        mfccs = python_speech_features.mfcc(wav, sample_rate).transpose()
        assert (mfccs.shape[0] == 13)
        return self.group_mfccs(mfccs)

    """
    Converts a video clip to sets of 5 frames (at the 25Hz frame rate)
    
    Each frame has a black and white mouth region
    Each frame has the dimension 224×224
    
    Gives 5 frames for every 0.2 second    
    """

    def format_input_visual(self, video_path):
        video = cv2.VideoCapture(video_path)

        assert (video.get(cv2.CAP_PROP_FPS) == 25)

        success, frame = video.read()
        frames = []
        while success:
            blw_mouth = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[frame.shape[0] // 2:frame.shape[0], :]
            blw_mouth = cv2.resize(blw_mouth, (224, 224))
            frames.append(blw_mouth)
            success, frame = video.read()
        frames = np.array(frames)

        for frame in frames:
            assert (frame.shape == (224, 224))

        return self.group_frames(frames)

    def get_shuffled_and_label_data(self):
        visual_data = np.array(self.visual_inputs)
        audio_data = np.array(self.audio_inputs)

        num_data_points = visual_data.shape[0]

        half_way_mark = num_data_points // 2

        # break data in 1/2
        # shuffle second half to create "false" data
        false_visual_data = visual_data[half_way_mark:num_data_points]
        np.random.shuffle(false_visual_data)
        visual_data[half_way_mark:num_data_points] = false_visual_data

        false_audio_data = audio_data[half_way_mark:num_data_points]
        np.random.shuffle(false_audio_data)
        audio_data[half_way_mark:num_data_points] = false_audio_data

        is_synced_labels = np.ones(num_data_points)
        is_synced_labels[half_way_mark:num_data_points] = 0

        new_idx_order = np.arange(num_data_points)
        np.random.shuffle(new_idx_order)

        # shuffle the labels and pairs together
        visual_data = visual_data[new_idx_order]
        audio_data = audio_data[new_idx_order]
        is_synced_labels = is_synced_labels[new_idx_order]

        return visual_data, audio_data, is_synced_labels

    def get_data(self):
        # Returns list with tensors
        try:
            print("Retrieving saved data...")
            start = process_time()
            visual_data = np.load("data/frames.npy")
            audio_data = np.load("data/audio.npy")
            is_synced_labels = np.load("data/labels.npy")
            end = process_time()
            print("Data retrieval completed in " + str(end - start) + " seconds!")

        except:

            print("Could not find saved dataset, generating and saving new dataset...")

            print("Preprocessing Raw Video Data...")
            start = process_time()
            for root, dirs, files in os.walk(self.data_path):
                for i, file in enumerate(files):
                    if file.endswith(".mp4"):
                        video_path = root + "/" + file
                        print(f"Preprocessing video {video_path}...")
                        self.format_video(video_path)
                if len(self.audio_inputs) > 30000:
                    break
            print("Done processing Raw Videos! Shuffling, labeling and saving processed data...")
            visual_data, audio_data, is_synced_labels = self.get_shuffled_and_label_data()

            # For saving as numpy arrays
            np.save("data/frames", visual_data)
            np.save("data/audio", audio_data)
            np.save("data/labels", is_synced_labels)
            print("Saved data!")
            end = process_time()
            print("Preprocessing completed in " + str(end - start) + " seconds!")

        assert (visual_data.shape[0] == audio_data.shape[0])
        num_data_points = len(is_synced_labels)
        print("Number of Data Points: ", num_data_points)
        print("Shape of Visual Data: ", visual_data.shape)
        print("Shape of Audio Data: ", audio_data.shape)

        self.visual_inputs = visual_data
        self.audio_inputs = audio_data
        self.is_synced_labels = is_synced_labels

        return self.visual_inputs, self.audio_inputs, self.is_synced_labels

    def get_and_create_tmp_video_audio_files(self, video_path):
        _, video_name = os.path.split(video_path)
        video_name = os.path.splitext(video_name)[0]

        tmp_video_path = os.path.dirname(video_path) + "/" + video_name + "_25_fps.mp4"
        tmp_audio_path = os.path.dirname(video_path) + "/" + video_name + "_audio.wav"

        self.create_tmp_video_audio_files(video_path, tmp_video_path, tmp_audio_path)

        return tmp_video_path, tmp_audio_path

    @staticmethod
    def create_tmp_video_audio_files(origional_video_path, tmp_audio_path, tmp_video_path):
        silence_suffix = " > /dev/null 2>&1 < /dev/null"

        audio_generation_cmd = "ffmpeg -i " + origional_video_path + " " + tmp_audio_path
        audio_generation_process = subprocess.Popen(audio_generation_cmd + silence_suffix, shell=True)
        audio_generation_process.wait()

        video_generation_cmd = "ffmpeg -i " + origional_video_path + " -filter:v fps=25 " + tmp_video_path
        video_generation_process = subprocess.Popen(video_generation_cmd + silence_suffix, shell=True)
        video_generation_process.wait()

    @staticmethod
    def delete_file(path_to_delete):
        rm_cmd = "rm " + path_to_delete
        rm_process = subprocess.Popen(rm_cmd, shell=True)
        rm_process.wait()