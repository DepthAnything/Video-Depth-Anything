<div align="center">
<h1>Video Depth Anything</h1>
  
[**Sili Chen**](https://github.com/SiliChen321) · [**Hengkai Guo**](https://guohengkai.github.io/)<sup>&dagger;</sup> · [**Shengnan Zhu**](https://github.com/Shengnan-Zhu)  · [**Feihu Zhang**](https://github.com/zhizunhu)
<br>
[**Zilong Huang**](http://speedinghzl.github.io/)   ·  [**Jiashi Feng**](https://scholar.google.com.sg/citations?user=Q8iay0gAAAAJ&hl=en)   ·  [**Bingyi Kang**](https://bingykang.github.io/)<sup>&dagger;</sup> 
<br>
ByteDance
<br>
&dagger;Corresponding author

<a href="https://arxiv.org/abs/2501.12375"><img src='https://img.shields.io/badge/arXiv-Video Depth Anything-red' alt='Paper PDF'></a>
<a href='https://videodepthanything.github.io'><img src='https://img.shields.io/badge/Project_Page-Video Depth Anything-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Video-Depth-Anything'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
</div>

</div>

This work presents **Video Depth Anything** based on [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), which can be applied to arbitrarily long videos without compromising quality, consistency, or generalization ability. Compared with other diffusion-based models, it enjoys faster inference speed, fewer parameters, and higher consistent depth accuracy.

![teaser](assets/teaser_video_v2.png)

## News
- **2025-09-12:** Support streaming mode for metric depth models.
- **2025-08-28:** Release ViT-base model for relative depth and ViT-small/base models for video metric depth.
- **2025-07-03:** 🚀🚀🚀 Release an experimental version of training-free **streaming video depth estimation**.
- **2025-07-03:** Release our implementation of [training loss](https://github.com/DepthAnything/Video-Depth-Anything/tree/main/loss).
- **2025-04-25:** 🌟🌟🌟 Release metric depth model based on Video-Depth-Anything-Large.
- **2025-04-05:** Our paper has been accepted for a **highlight** presentation at [CVPR 2025](https://cvpr.thecvf.com/) (13.5% of the accepted papers).
- **2025-03-11:** Add full dataset inference and evaluation [scripts](https://github.com/DepthAnything/Video-Depth-Anything/tree/main/benchmark).
- **2025-02-08:** Enable autocast inference. Support grayscale video, NPZ and EXR output formats.
- **2025-01-21:** Paper, project page, code, models, and demo are all released.


## Release Notes
- **2025-08-28:** 🚀🚀🚀 Metric depth models released

| δ1 | MoGe-2-L | UniDepthV2-L | DepthPro | VDA-S-Metric | VDA-B-Metric | VDA-L-Metric |
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| KITTI | 0.415 | **0.982** | 0.822 | 0.877 | 0.887 | *0.910* |
| NYUv2 | *0.967* | **0.989** | 0.953 | 0.850| 0.883 | 0.908 |
| **TAE** |  |  |  |  |  |  |
| Scannet | 2.56 | 1.41 | 2.73 | 1.48 | *1.26* | **1.09** |

- **2025-02-08:** 🚀🚀🚀 Inference speed and memory usage improvement
  <table>
    <thead>
      <tr>
        <th rowspan="2" style="text-align: center;">Model</th>
        <th colspan="2">Latency (ms)</th>
        <th colspan="2">GPU VRAM (GB)</th>
      </tr>
      <tr>
        <th>FP32</th>
        <th>FP16</th>
        <th>FP32</th>
        <th>FP16</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Video-Depth-Anything-Small</td>
        <td>9.1</td>
        <td><strong>7.5</strong></td>
        <td>7.3</td>
        <td><strong>6.8</strong></td>
      </tr>
      <tr>
        <td>Video-Depth-Anything-Large</td>
        <td>67</td>
        <td><strong>14</strong></td>
        <td>26.7</td>
        <td><strong>23.6</strong></td>
    </tbody>
  </table>

  The Latency and GPU VRAM results are obtained on a single A100 GPU with input of shape 1 x 32 x 518 × 518.

## Pre-trained Models
We provide **several models** of varying scales for robust and consistent video depth estimation.

| Relative Depth Model | Params | Checkpoint |
|:-|-:|:-:|
| Video-Depth-Anything-Small | 28.4M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true) |
| Video-Depth-Anything-Base | 113.1M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Base/blob/main/video_depth_anything_vitb.pth) | 
| Video-Depth-Anything-Large | 381.8M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true) |
| **Metric Depth Model** | **Params** | **Checkpoint**  |
| Metric-Video-Depth-Anything-Small | 28.4M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Small/blob/main/metric_video_depth_anything_vits.pth) |
| Metric-Video-Depth-Anything-Base | 113.1M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Base/blob/main/metric_video_depth_anything_vitb.pth) |
| Metric-Video-Depth-Anything-Large | 381.8M | [Download](https://huggingface.co/depth-anything/Metric-Video-Depth-Anything-Large/resolve/main/metric_video_depth_anything_vitl.pth) |


## Usage

### Preparation

```bash
git clone https://github.com/DepthAnything/Video-Depth-Anything
cd Video-Depth-Anything
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.
```bash
bash get_weights.sh
```

### Run inference on a video
We support both relative depth and metric depth:
```bash
# For relative depth
python3 run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl

# For metric depth
python3 run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl --metric
```

Options:
- `--input_video`: path of input video
- `--output_dir`: path to save the output results
- `--input_size` (optional): By default, we use input size `518` for model inference.
- `--max_res` (optional): By default, we use maximum resolution `1280` for model inference.
- `--encoder` (optional): `vits` for Video-Depth-Anything-Small, `vitb` for Video-Depth-Anything-Base, `vitl` for Video-Depth-Anything-Large.
- `--max_len` (optional): maximum length of the input video, `-1` means no limit
- `--target_fps` (optional): target fps of the input video, `-1` means the original fps
- `--metric` (optional): use metric depth models trained on Virtual KITTI and IRS datasets
- `--fp32` (optional): Use `fp32` precision for inference. By default, we use `fp16`.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.
- `--save_npz` (optional): Save the depth map in `npz` format.
- `--save_exr` (optional): Save the depth map in `exr` format.

### Run inference on a video using streaming mode (Experimental features)
We implement an experimental streaming mode **without training**. In details, we save the hidden states of temporal attentions for each frames in the caches, and only send a single frame into our video depth model during inference by reusing these past hidden states in temporal attentions. We hack our pipeline to align the original inference setting in the offline mode. Due to the inevitable gap between training and testing, we observe a **performance drop** between the streaming model and the offline model (e.g. the `d1` of ScanNet drops from `0.926` to `0.836`). Finetuning the model in the streaming mode will greatly improve the performance. We leave it for future work.

To run the streaming model:
```bash
# For relative depth
python3 run_streaming.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs_streaming --encoder vitl

# For metric depth
python3 run_streaming.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs_streaming --encoder vitl --metric
```
Options:
- `--input_video`: path of input video
- `--output_dir`: path to save the output results
- `--input_size` (optional): By default, we use input size `518` for model inference.
- `--max_res` (optional): By default, we use maximum resolution `1280` for model inference.
- `--encoder` (optional): `vits` for Video-Depth-Anything-Small, `vitb` for Video-Depth-Anything-Base, `vitl` for Video-Depth-Anything-Large.
- `--max_len` (optional): maximum length of the input video, `-1` means no limit
- `--target_fps` (optional): target fps of the input video, `-1` means the original fps
- `--metric` (optional): use metric depth models trained on Virtual KITTI and IRS datasets
- `--fp32` (optional): Use `fp32` precision for inference. By default, we use `fp16`.
- `--grayscale` (optional): Save the grayscale depth map, without applying color palette.

## Training Loss
Our training loss is in `loss/` directory. Please see the `loss/test_loss.py` for usage.

## Benchmark
Please refer to [Benchmark](./benchmark/README.md).

## Citation

If you find this project useful, please consider citing:

```bibtex
@article{video_depth_anything,
  title={Video Depth Anything: Consistent Depth Estimation for Super-Long Videos},
  author={Chen, Sili and Guo, Hengkai and Zhu, Shengnan and Zhang, Feihu and Huang, Zilong and Feng, Jiashi and Kang, Bingyi}
  journal={arXiv:2501.12375},
  year={2025}
}
```


## LICENSE
Video-Depth-Anything-Small model is under the Apache-2.0 license. Video-Depth-Anything-Base/Large model is under the CC-BY-NC-4.0 license. For business cooperation, please send an email to Hengkai Guo at guohengkaighk@gmail.com.
