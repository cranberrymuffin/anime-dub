import cv2
import librosa
import python_speech_features
import numpy as np
import os

# 0.2 seconds
input_duration_milliseconds = 200
input_duration_seconds = 0.2


class DataPipeline:
    def __init__(self, data_path):
        self.audio_tensors = []
        self.frame_tensors = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp4"):
                    self.format_video(root + "/" + file)

    """
    * The network ingests 0.2-second clips of both audio and video inputs.
    This function...
    1) Splits the video at video_path into segments of 0.2 seconds each
    2) Separates the visual and audio for each clip
    3) Formats the visual and audio using format_input_visual and format_input_audio
    4) Appends this data to instance variables self.audio_inputs self.visual_inputs
    """
    def format_video(self, video_path):
        _25_fps_video_path = os.path.dirname(video_path) + "/_25_fps.mp4"
        os.system("ffmpeg -i " + video_path +" -filter:v fps=25 " + _25_fps_video_path)
        frames = self.format_input_visual(_25_fps_video_path)

        audio_path = os.path.dirname(video_path) + "/audio.wav"
        os.system("ffmpeg -i " + video_path + " " + audio_path)
        mfcc = self.format_input_audio(audio_path)

        frames, mfccs = self.trim_mfcc_and_visual(frames, mfcc)
        assert(len(frames) == len(mfccs))

    def trim_mfcc_and_visual(self, frames, mfcc):
        # trim mfcc cols to be a multiple of 20
        num_cols = mfcc.shape[1]
        assert(num_cols > 20)
        to_trim = num_cols % 20
        if to_trim > 0:
            mfcc = mfcc[:, :-to_trim]
        # group mfcc cols into sets of 20
        mfcc_groups = np.hsplit(mfcc, mfcc.shape[1]/20)
        num_mfccs = len(mfcc_groups)
        assert(mfcc_groups[0].shape == (13, 20))
        # trim frames to be a multiple of 5
        num_frames = frames.shape[0]
        assert(num_frames > 5)
        to_trim = num_frames % 5
        if to_trim > 0:
            frames = frames[:-to_trim, :, :]

        # group frames into sets of 5
        frames_groups = np.split(frames, frames.shape[0]/5)
        frame_groups = [np.dstack(group) for group in frames_groups]
        num_frame_groups = len(frame_groups)
        assert(frame_groups[0].shape == (111, 111, 5))

        num_groups = min(num_mfccs, num_frame_groups)
        to_return = frame_groups[:num_groups], mfcc_groups[:num_groups]
        assert(len(to_return[0]) == len(to_return[1]))
        return to_return

    """
    Converts a audio clip to MFCC map of N x 13 x 20
    13 MFCC features representing powers at different frequency bins.
    Sampled at rate of 100Hz, gives 20 time steps for a 0.2-second input signal.
    """

    def format_input_audio(self, wav_file):
        wav, sample_rate = librosa.load(wav_file, sr=100)
        assert (sample_rate == 100)
        mfcc = python_speech_features.mfcc(wav, sample_rate).transpose()
        assert (mfcc.shape[0] == 13)
        os.system("rm " + wav_file)
        return mfcc

    """
    Converts a video clip to sets of 5 frames (at the 25Hz frame rate)
    
    Each frame has a black and white mouth region
    Each frame has the dimension 111×111
    
    Gives 5 frames for every 0.2 second    
    """

    def format_input_visual(self, video_path):
        video = cv2.VideoCapture(video_path)
        print(video.get(cv2.CAP_PROP_FPS))
        assert (video.get(cv2.CAP_PROP_FPS) == 25)
        success, frame = video.read()
        frames = []
        while success:
            blw_mouth = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)[frame.shape[0] // 2:frame.shape[0], :]
            blw_mouth = cv2.resize(blw_mouth, (111, 111))
            frames.append(blw_mouth)
            success, frame = video.read()
        frames = np.array(frames)
        os.system("rm " + video_path)
        assert(frames[0].shape == (111, 111))
        return frames


    def get_data(self):
        return self.frame_tensors, self.audio_tensors


if __name__ == "__main__":
    frame_tensors, audio_tensors = DataPipeline("/Users/aparna/Downloads/converted/").get_data()
