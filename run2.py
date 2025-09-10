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

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

class TimingStats:
    def __init__(self):
        self.start_time = time.time()
        self.model_load_time = 0
        self.read_time = 0
        self.inference_time = 0
        self.save_time = 0
        self.extra_save_time = 0
    
    def get_total_time(self):
        return time.time() - self.start_time
    
    def print_stats(self, num_frames):
        print("\nProcessing Time Breakdown:")
        print(f"Model Loading: {self.model_load_time:.2f}s")
        print(f"Video Reading: {self.read_time:.2f}s")
        print(f"Depth Inference: {self.inference_time:.2f}s")
        print(f"Video Saving: {self.save_time:.2f}s")
        if self.extra_save_time > 0:
            print(f"Additional Format Saving: {self.extra_save_time:.2f}s")
        print(f"Total Time: {self.get_total_time():.2f}s")
        
        print(f"\nPer-frame Statistics:")
        print(f"Number of Frames: {num_frames}")
        print(f"Average Processing Time per Frame: {self.inference_time/num_frames:.3f}s ({(num_frames/self.inference_time):.1f} FPS)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video Depth Anything')
    parser.add_argument('--input_video', type=str, default='./assets/example_videos/davis_rollercoaster.mp4')
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--input_size', type=int, default=518)
    parser.add_argument('--max_res', type=int, default=1280)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitl'])
    parser.add_argument('--max_len', type=int, default=-1, help='maximum length of the input video, -1 means no limit')
    parser.add_argument('--target_fps', type=int, default=-1, help='target fps of the input video, -1 means the original fps')
    parser.add_argument('--fp32', action='store_true', help='model infer with torch.float32, default is torch.float16')
    parser.add_argument('--grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument('--save_npz', action='store_true', help='save depths as npz')
    parser.add_argument('--save_exr', action='store_true', help='save depths as exr')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device number to use (default: 0)')

    args = parser.parse_args()

    if torch.cuda.is_available():
        if args.gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU device {args.gpu} not found. Available devices: 0 to {torch.cuda.device_count()-1}")
        DEVICE = f'cuda:{args.gpu}'
        print(f"Using GPU device {args.gpu}: {torch.cuda.get_device_name(args.gpu)}")
    else:
        DEVICE = 'cpu'
        print("CUDA not available, using CPU")

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    }

    timing = TimingStats()
    
    try:
        # Model loading
        model_start = time.time()
        video_depth_anything = VideoDepthAnything(**model_configs[args.encoder])
        video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{args.encoder}.pth', map_location='cpu'), strict=True)
        video_depth_anything = video_depth_anything.to(DEVICE).eval()
        timing.model_load_time = time.time() - model_start

        # Video reading
        read_start = time.time()
        frames, target_fps = read_video_frames(args.input_video, args.max_len, args.target_fps, args.max_res)
        timing.read_time = time.time() - read_start

        # Depth inference
        inference_start = time.time()
        depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=args.input_size, device=DEVICE, fp32=args.fp32)
        timing.inference_time = time.time() - inference_start
        
        # Video saving
        save_start = time.time()
        video_name = os.path.basename(args.input_video)
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        processed_video_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
        depth_vis_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
        save_video(frames, processed_video_path, fps=fps)
        save_video(depths, depth_vis_path, fps=fps, is_depths=True, grayscale=args.grayscale)
        timing.save_time = time.time() - save_start

        # Optional saving of additional formats
        if args.save_npz or args.save_exr:
            extra_save_start = time.time()
            
            if args.save_npz:
                depth_npz_path = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths.npz')
                np.savez_compressed(depth_npz_path, depths=depths)
                
            if args.save_exr:
                depth_exr_dir = os.path.join(args.output_dir, os.path.splitext(video_name)[0]+'_depths_exr')
                os.makedirs(depth_exr_dir, exist_ok=True)
                import OpenEXR
                import Imath
                for i, depth in enumerate(depths):
                    output_exr = f"{depth_exr_dir}/frame_{i:05d}.exr"
                    header = OpenEXR.Header(depth.shape[1], depth.shape[0])
                    header["channels"] = {
                        "Z": Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))
                    }
                    exr_file = OpenEXR.OutputFile(output_exr, header)
                    exr_file.writePixels({"Z": depth.tobytes()})
                    exr_file.close()
                    
            timing.extra_save_time = time.time() - extra_save_start

        # Print timing statistics
        timing.print_stats(len(frames))

    except Exception as e:
        raise RuntimeError(f"Error during video processing: {str(e)}")