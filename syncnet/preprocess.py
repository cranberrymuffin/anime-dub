import cv2
import os
from scipy.io import wavfile
import numpy as np
import torch
import python_speech_features
from moviepy.editor import *

class Dataset:
    def __init__(self, data_path):
        self.audio_tensors = []
        self.frame_tensors = []
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp4"):
                    self.preprocess_data(root + "/" + file)

    @staticmethod
    def augment_frame(face_frame):
        blw_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2GRAY)
        mouth_region = blw_face_frame[blw_face_frame.shape[0]//2:blw_face_frame.shape[0], :]
        return cv2.resize(mouth_region, (111,111))

    def preprocess_data(self, video_path):
        print(video_path)
        # convert video to 25 fps
        _25_fps_vid_path = os.path.dirname(video_path) + "/tmp.mp4"
        cmd_25_fps = "ffmpeg -i " + video_path + " -r 25 " + _25_fps_vid_path
        os.system(cmd_25_fps)
        video_path = _25_fps_vid_path

        # go through each frame in video
        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        frames = []
        while success:
            # convert frame to black and white and crop to mouth region (bottom half of face)
            frames.append(self.augment_frame(frame))
            success, frame = video.read()

        # group frames into sets of 5
        frames = frames[:-(len(frames) % 5)]
        num_frames = len(frames)

        if num_frames == 0:
            os.system("rm " + _25_fps_vid_path)
            return

        frames = np.split(np.array(frames), num_frames/5)

        # convert sets of 5 into tensors
        for group_of_5_frames in frames:
            visual_input_tensor = torch.from_numpy(group_of_5_frames.astype(float))
            self.frame_tensors.append(visual_input_tensor)

        # get audio
        audio_save_path = os.path.dirname(video_path) + "/" + "audio.wav"
        audio_from_video_cmd = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn " + audio_save_path
        os.system(audio_from_video_cmd)
        sample_rate, wav = wavfile.read(audio_save_path)

        # store audio as tensor
        # trim audio to same number of samples as frames in video
        mfcc = zip(*python_speech_features.mfcc(wav[:num_frames], sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])
        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        self.audio_tensors.append(torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float()))

        # remove 25 fps file
        os.system("rm " + audio_save_path)
        os.system("rm " + _25_fps_vid_path)

    def get_data(self):
        return self.frame_tensors, self.audio_tensors
