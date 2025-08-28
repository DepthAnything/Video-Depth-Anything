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
import gradio as gr

import numpy as np
import os
import torch

from video_depth_anything.video_depth import VideoDepthAnything
from utils.dc_utils import read_video_frames, save_video

examples = [
    ['assets/example_videos/davis_rollercoaster.mp4', -1, -1, 1280],
]

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}

encoder='vitl'

video_depth_anything = VideoDepthAnything(**model_configs[encoder])
video_depth_anything.load_state_dict(torch.load(f'./checkpoints/video_depth_anything_{encoder}.pth', map_location='cpu'), strict=True)
video_depth_anything = video_depth_anything.to('cuda').eval()


def infer_video_depth(
    input_video: str,
    max_len: int = -1,
    target_fps: int = -1,
    max_res: int = 1280,
    output_dir: str = './outputs',
    input_size: int = 518,
):
    frames, target_fps = read_video_frames(input_video, max_len, target_fps, max_res)
    depths, fps = video_depth_anything.infer_video_depth(frames, target_fps, input_size=input_size, device='cuda')

    video_name = os.path.basename(input_video)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    processed_video_path = os.path.join(output_dir, os.path.splitext(video_name)[0]+'_src.mp4')
    depth_vis_path = os.path.join(output_dir, os.path.splitext(video_name)[0]+'_vis.mp4')
    save_video(frames, processed_video_path, fps=fps)
    save_video(depths, depth_vis_path, fps=fps, is_depths=True)

    return [processed_video_path, depth_vis_path]


def construct_demo():
    with gr.Blocks(analytics_enabled=False) as demo:
        gr.Markdown(
            f"""
            blablabla
            """
        )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                input_video = gr.Video(label="Input Video")

            # with gr.Tab(label="Output"):
            with gr.Column(scale=2):
                with gr.Row(equal_height=True):
                    processed_video = gr.Video(
                        label="Preprocessed video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )
                    depth_vis_video = gr.Video(
                        label="Generated Depth Video",
                        interactive=False,
                        autoplay=True,
                        loop=True,
                        show_share_button=True,
                        scale=5,
                    )

        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                with gr.Row(equal_height=False):
                    with gr.Accordion("Advanced Settings", open=False):
                        max_len = gr.Slider(
                            label="max process length",
                            minimum=-1,
                            maximum=1000,
                            value=-1,
                            step=1,
                        )
                        target_fps = gr.Slider(
                            label="target FPS",
                            minimum=-1,
                            maximum=30,
                            value=15,
                            step=1,
                        )
                        max_res = gr.Slider(
                            label="max side resolution",
                            minimum=480,
                            maximum=1920,
                            value=1280,
                            step=1,
                        )
                    generate_btn = gr.Button("Generate")
            with gr.Column(scale=2):
                pass

        gr.Examples(
            examples=examples,
            inputs=[
                input_video,
                max_len,
                target_fps,
                max_res
            ],
            outputs=[processed_video, depth_vis_video],
            fn=infer_video_depth,
            cache_examples="lazy",
        )

        generate_btn.click(
            fn=infer_video_depth,
            inputs=[
                input_video,
                max_len,
                target_fps,
                max_res
            ],
            outputs=[processed_video, depth_vis_video],
        )

    return demo

if __name__ == "__main__":
    demo = construct_demo()
    demo.queue()
    demo.launch()
