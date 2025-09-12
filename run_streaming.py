# Copyright (2025) Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import numpy as np
import os
import torch
import time
import cv2

from video_depth_anything.video_depth_stream import VideoDepthAnything
from utils.dc_utils import save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    cap = cv2.VideoCapture(args.input_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.max_res > 0 and max(original_height, original_width) > args.max_res:
        scale = args.max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if args.target_fps < 0 else args.target_fps

    stride = max(round(original_fps / fps), 1)

    depths = []
    frame_count = 0
    start = time.time()
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or (args.max_len > 0 and frame_count >= args.max_len):
            break
        if frame_count % stride == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            if args.max_res > 0 and max(original_height, original_width) > args.max_res:
                frame = cv2.resize(frame, (width, height))  # Resize frame

            # Inference depth
            depth = video_depth_anything.infer_video_depth_one(frame, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
            depths.append(depth)
        frame_count += 1
        if frame_count % 50 == 0:
            print(f"frame: {frame_count}/{total_frames}")
    end = time.time()

    cap.release()
    print(f"time: {end - start}s")

    video_name = os.path.basename(args.input_video)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    depths = np.stack(depths, axis=0)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
