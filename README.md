# CLIPood (ICML 2023)

CLIPood: Generalizing CLIP to Out-of-Distributions [[paper]](https://arxiv.org/abs/2302.00864)

To maintain CLIP's OOD generalizability when adapting CLIP to downstream tasks, we propose CLIPood with the following features:
- Better fine-tuning paradigm to utilize knowledge in **text modality**.
- Margin matric softmax to exploit **semantic relations**.
- Beta moving average for balancing **pre-trained** and **task-specific** knowledge.
- State-of-the-art performance on **three OOD settings**.

<p align="center">
<img src=".\figs\overview.png" height = "320" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of CLIPood.
</p>

## Get Started

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. Our datasets are built over [DomainBed](https://github.com/facebookresearch/DomainBed) and [CoOp](https://github.com/KaiyangZhou/CoOp) codebases. For usage, one should clone the above two repositories, then download data following the instructions, and arrange the folder as:
```plain
CLIPood/
|-- CoOp/
    |-- data/
        |-- caltech-101/
        |-- eurosat/
        |-- ...  # other CoOp datasets
    |-- ...
|-- DomainBed/
    |-- domainbed/
        |-- data/
            |-- domain_net/
            |-- office_home/
            |-- ...  # other DomainBed datasets
        |-- datasets.py  # the file to update next
        |-- ...
    |-- ...
|-- ...
```

3. ***(Important!)*** Replace line 192-208 in file `DomainBed/domainbed/dataset.py` with the following codes:
```python
transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])
```

4. Train and evaluate model.

## Results

We extensively experiment on three OOD settings (domain shift, open class, in the wild). On all settings CLIPood achieves remarkable improvement gain and reaches the state-of-the-art.

## Citation
If you find this repo useful, please cite our paper.

```plain
@inproceedings{shu2023CLIPood,
  title={CLIPood: Generalizing CLIP to Out-of-Distributions},
  author={Yang Shu and Xingzhuo Guo and Jialong Wu and Ximei Wang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Contact

If you have any questions or want to use the code, please contact [gxz23@mails.tsinghua.edu.cn](mailto:gxz23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Transfer-Learning-Library

https://github.com/facebookresearch/DomainBed

https://github.com/KaiyangZhou/CoOp
