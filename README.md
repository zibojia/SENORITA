# Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists

## Overview

Señorita-2M is a comprehensive and high-quality dataset designed for general video editing tasks. It consists of a vast collection of videos with detailed instructions provided by video specialists. Our dataset aims to facilitate the development and evaluation of advanced video editing algorithms.

## Abstract

Recent advancements in video generation have spurred the development of video editing techniques, which can be divided into inversion-based and end-to-end methods. However, current video editing methods still suffer from several challenges. Inversion-based methods, though training free and flexible, are time-consuming during inference, struggle with fine-grained editing instructions, and produce artifacts and jitter. On the other hand, end-to-end methods, which rely on edited video pairs for training, offer faster inference speeds but often produce poor editing results due to a lack of high-quality training video pairs. In this paper, to close the gap in end-to-end methods, we introduce Señorita-2M, a high-quality video editing dataset. Señorita-2M consists of approximately 2 million video editing pairs. It is built by crafting four high-quality, specialized video editing models, each crafted and trained by our team to achieve state-of-the-art editing results. We also propose a filtering pipeline to eliminate poorly edited video pairs. Furthermore, we explore common video editing architectures to identify the most effective structure based on current pre-trained generative model. Extensive experiments show that our dataset can help to yield remarkably high-quality video editing results.

## Key Features

- **High-Quality Annotations**: Each video in the dataset is accompanied by precise and detailed instructions from professional video editors.
- **Diverse Editing Tasks**: The dataset covers a wide range of video editing tasks, including object removal, object swap, global and local stylization.
- **Large Scale**: With over 2 million video clips, Señorita-2M is one of the largest video editing datasets available.

## Dataset Construction

We built the dataset by leveraging high-quality video editing experts. Specifically, we trained four high-quality video editing experts using CogVideoX: a global stylizer, a local stylizer, an inpainting model, and a remover. These experts, along with other specialized models, are used to construct a large-scale dataset of high-quality video editing samples. Additionally, we designed a filtering pipeline that effectively removes failed video samples. We also utilized a large language model to convert video editing prompts, achieving clear and effective instructions. As a result, Señorita-2M contains approximately 2 million high-quality video editing pairs.

Furthermore, we trained multiple video editors based on different video editing architectures using this dataset to evaluate the effectiveness of various editing frameworks, ultimately achieving impressive editing capabilities.

## Editing Tasks

Our dataset consists of 17 editing tasks. Five of these tasks are edited by our trained experts, while the remaining tasks are handled by computer vision tasks. The former sub-dataset occupies around 76.8% of the video pairs in the dataset, while the latter 12 tasks take up 23.2% of the video pairs in total.

## Sample Images

### Teaser
![Teaser](images/teaser.PNG)

### Global Stylization
![Global Stylization](images/global_stylization.PNG)

### Local Stylization
![Local Stylization](images/local_stylization.PNG)

### Object Removal
![Object Removal](images/object_removal.PNG)

### Object Swap
![Object Swap](images/object_swap.PNG)

## Dataset Structure

The dataset is organized into several categories, each representing a different type of video editing task. Below is a brief overview of the categories:

1. **Global Stylization**: Videos in this category involve applying a consistent style across the entire video.
2. **Local Stylization**: This category focuses on applying styles to specific regions within the video.
3. **Object Removal**: Videos where one or more objects are removed seamlessly.
4. **Object Swap**: Involves replacing one object in the video with another.

## Citation

If you use Señorita-2M in your research, please cite our work as follows:

```
@dataset{senorita2m2025,
  author    = {Bojia Zi and Penghui Ruan and Marco Chen and Xianbiao Qi and Shaozhe Hao and Shihao Zhao and Youze Huang and Bin Liang and Rong Xiao and Kam-Fai Wong},
  title     = {Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists},
  year      = {2025},
  publisher = {Video Editing Research Group},
  url       = {https://example.com/senorita2m}
}

@article{zi2025senorita,
  title={Señorita-2M: A High-Quality Instruction-based Dataset for General Video Editing by Video Specialists},
  author={Bojia Zi and Penghui Ruan and Marco Chen and Xianbiao Qi and Shaozhe Hao and Shihao Zhao and Youze Huang and Bin Liang and Rong Xiao and Kam-Fai Wong},
  journal={arXiv preprint arXiv:2502.06734},
  year={2025},
}
```

## Authors

- Bojia Zi, The Chinese University of Hong Kong
- Penghui Ruan, The Hong Kong Polytechnic University
- Marco Chen, Tsinghua University
- Xianbiao Qi, IntelliFusion Inc.
- Shaozhe Hao, The University of Hong Kong
- Shihao Zhao, The University of Hong Kong
- Youze Huang, University of Electronic Science and Technology of China
- Bin Liang, The Chinese University of Hong Kong
- Rong Xiao, IntelliFusion Inc.
- Kam-Fai Wong, The Chinese University of Hong Kong

**Note**: * indicates equal contribution. † indicates the corresponding author.

## Links

- [Model](https://huggingface.co/PengWeixuanSZU/Senorita-2M)
- [Demo Page](https://senorita-2m-dataset.github.io/)
- [Dataset](https://huggingface.co/datasets/SENORITADATASET/Senorita)

## Contact

For more information or any queries regarding the dataset, please contact us at [info@example.com](mailto:info@example.com).

