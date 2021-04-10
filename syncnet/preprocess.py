import math
import time

import cv2
import os
from scipy.io import wavfile
import numpy as np
import torch
import python_speech_features
from moviepy.editor import *
from syncnet import LipSyncEvalNet


class Dataset:
    def __init__(self, data_path):
        self.audio_tensors = []
        self.frame_tensors = []
        self.nn = LipSyncEvalNet()
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if file.endswith(".mp4"):
                    self.preprocess_data(root + "/" + file)

    """
    Converts frame to black and white and crops to mouth region (lower half of face)
    Also converts resulting output to 111 x 111 as described in syncnet paper
    """

    @staticmethod
    def augment_frame(face_frame):
        blw_face_frame = cv2.cvtColor(face_frame, cv2.COLOR_RGB2GRAY)
        mouth_region = blw_face_frame[blw_face_frame.shape[0] // 2:blw_face_frame.shape[0], :]
        return cv2.resize(mouth_region, (111, 111))

    def preprocess_data(self, video_path):
        """
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

        images = np.stack(frames, axis=3)
        images = np.expand_dims(images, axis=0)
        images = np.transpose(images, (0, 3, 4, 1, 2))

        visual_input_tensor = torch.autograd.Variable(torch.from_numpy(images.astype(float)).float())
        im_batch = [visual_input_tensor[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + opt.batch_size))]
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
    """

        video = cv2.VideoCapture(video_path)
        success, frame = video.read()
        images = []
        while success:
            # convert frame to black and white and crop to mouth region (bottom half of face)
            frame = self.augment_frame(frame)
            images.append(np.array([frame, frame, frame]))
            success, frame = video.read()

        im = np.stack(images, axis=3)
        im = np.expand_dims(im, axis=0)
        im = np.transpose(im, (0, 3, 4, 1, 2))

        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== ==========
        # Load audio
        # ========== ==========

        audio_save_path = os.path.dirname(video_path) + "/" + "audio.wav"
        audio_from_video_cmd = "ffmpeg -i " + video_path + " -ab 160k -ac 2 -ar 44100 -vn " + audio_save_path
        os.system(audio_from_video_cmd)
        sample_rate, audio = wavfile.read(audio_save_path)
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = np.stack([np.array(i) for i in mfcc])

        cc = np.expand_dims(np.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== ==========
        # Check audio and video input length
        # ========== ==========

        if (float(len(audio)) / 16000) != (float(len(images)) / 25):
            print("WARNING: Audio (%.4fs) and video (%.4fs) lengths are different." % (
                float(len(audio)) / 16000, float(len(images)) / 25))

        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== ==========
        # Generate video and audio feats
        # ========== ==========

        lastframe = min_length - 5
        im_feat = []
        cc_feat = []

        tS = time.time()
        batch_size = 16
        for i in range(0, lastframe, batch_size):
            im_batch = [imtv[:, :, vframe:vframe + 5, :, :] for vframe in range(i, min(lastframe, i + batch_size))]
            im_in = torch.cat(im_batch, 0)
            im_out = self.nn.forward_lip(im_in.cuda());
            im_feat.append(im_out.data.cpu())

            cc_batch = [cct[:, :, :, vframe * 4:vframe * 4 + 20] for vframe in
                        range(i, min(lastframe, i + batch_size))]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.nn.forward_aud(cc_in.cuda())
            cc_feat.append(cc_out.data.cpu())


    """Get audio and visual tensors"""
    def get_data(self):
        return self.frame_tensors, self.audio_tensors
