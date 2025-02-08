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
- **2025-02-08:** Enable autocast inference. Support grayscale video, NPZ and EXR output formats.
- **2025-01-21:** Paper, project page, code, models, and demo are all released.


## Pre-trained Models
We provide **two models** of varying scales for robust and consistent video depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Video-Depth-Anything-V2-Small | 28.4M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Small/resolve/main/video_depth_anything_vits.pth?download=true) |
| Video-Depth-Anything-V2-Large | 381.8M | [Download](https://huggingface.co/depth-anything/Video-Depth-Anything-Large/resolve/main/video_depth_anything_vitl.pth?download=true) |


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

### Inference a video
```bash
python3 run.py --input_video ./assets/example_videos/davis_rollercoaster.mp4 --output_dir ./outputs --encoder vitl
```

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
Video-Depth-Anything-Small model is under the Apache-2.0 license. Video-Depth-Anything-Large model is under the CC-BY-NC-4.0 license.
