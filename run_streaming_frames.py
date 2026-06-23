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
import glob
import matplotlib.cm as cm

from video_depth_anything.video_depth_stream import VideoDepthAnything
from utils.dc_utils import save_video

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4', help='path to folder containing input frames')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--metric', action='store_true', help='use metric model')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')

    args = parser.parse_args()

    DEVICE = f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu'

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }
    checkpoint_name = 'metric_video_depth_anything' if args.metric else 'video_depth_anything'

    video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
    video_depth_anything.load_state_dict(torch.load(f'./checkpoints/{checkpoint_name}_{args.encoder}.pth', map_location='cpu'), strict=True)
    video_depth_anything = video_depth_anything.to(DEVICE).eval()

    # Get list of image files from folder
    frame_files = sorted(glob.glob(os.path.join(args.input_video, '*.jpg')))
    total_frames = len(frame_files)
    
    if total_frames == 0:
        raise ValueError(f"No .jpg files found in {args.input_video}")
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_files[0])
    original_height, original_width = first_frame.shape[:2]
    
    height, width = original_height, original_width  # default: keep original dims
    if args.max_res > 0 and max(original_height, original_width) > args.max_res:
        scale = args.max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    # Use 30 FPS as default
    original_fps = 30
    fps = original_fps if args.target_fps < 0 else args.target_fps

    stride = max(round(original_fps / fps), 1)

    depths = []
    start = time.time()
    
    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Get colormap for depth visualization
    colormap = np.array(cm._colormaps["inferno"].colors)
    
    for frame_count, frame_path in enumerate(frame_files):
        if args.max_len > 0 and frame_count >= args.max_len:
            break
            
        if frame_count % stride == 0:
            frame = cv2.imread(frame_path)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
            
            if args.max_res > 0 and max(original_height, original_width) > args.max_res:
                frame = cv2.resize(frame, (width, height))  # Resize frame

            # Inference depth
            depth = video_depth_anything.infer_video_depth_one(frame, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
            depths.append(depth)
            
            # Save individual depth frame as JPG with colormap
            frame_name = os.path.basename(frame_path)
            frame_name_no_ext = os.path.splitext(frame_name)[0]
            depth_frame_path = os.path.join(args.output_dir, frame_name_no_ext + '_depth.jpg')
            
            # Apply same colormap as video output
            depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
            if args.grayscale:
                depth_colored = depth_normalized
            else:
                depth_colored = (colormap[depth_normalized] * 255).astype(np.uint8)
            
            # Convert RGB to BGR for cv2.imwrite
            if not args.grayscale:
                depth_colored = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            cv2.imwrite(depth_frame_path, depth_colored)
            
        if (frame_count + 1) % 50 == 0:
            print(f"frame: {frame_count + 1}/{total_frames}")
    
    end = time.time()
    print(f"time: {end - start}s")

    folder_name = os.path.basename(args.input_video.rstrip('/'))
    depth_vis_path = os.path.join(args.output_dir, folder_name + '_vis.mp4')
    depths = np.stack(depths, axis=0)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
