import json
import pandas as pd
import cv2
import math
import numpy as np


def save_csv(json_file, csv_file):

    video_list = json.load(open(json_file))
    train = pd.DataFrame()
    end_list = []
    videos = []
    label = []

    for videoname, labels in video_list.items():
        transitions = video_list[videoname]['transitions']
        no_of_frames = video_list[videoname]['frame_num']

        for begin, end in transitions:
            end_list.append(end)
        for frame_num in range(1, int(no_of_frames) + 1):
            filename = videoname.split('.')[0] + "_frame%d.jpg" % frame_num
            videos.append(filename)
            if frame_num in end_list:
                label.append(1)
            else:
                label.append(0)

    train['file_name'] = videos
    train['labels'] = label
    train = train[:-1]
    train.to_csv(csv_file, header=True, index=False)


def save_video_frames(json_file, input_video_dir, output_frames_dir):

    video_list = json.load(open(json_file))

    for videoFile, labels in video_list.items():
        cap = cv2.VideoCapture(input_video_dir + videoFile)
        count = 0
        old_frame = None
        while(cap.isOpened()):
            ret, frame = cap.read()
            filename = output_frames_dir + \
                videoFile.split('.')[0] + "_frame%d.jpg" % count
            count += 1
            if ret:
                if old_frame is not None:
                    stack_frame = cv2.vconcat([old_frame, frame])
                    cv2.imwrite(filename, stack_frame)
                old_frame = frame
            else:
                break
        cap.release()
        break


if __name__ == "__main__":
    json_file = "test.json"
    csv_file = "test.csv"
    # save_csv(json_file, csv_file)

    input_video_dir = "test/"
    output_frames_dir = "test_frames/"
    save_video_frames(json_file, input_video_dir, output_frames_dir)
