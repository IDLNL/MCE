## :book: Cross-Modality Person Re-Identification with Memory-based Contrastive Embedding (AAAI 2023)
# [Paper] https://ojs.aaai.org/index.php/AAAI/article/view/25116

Pytorch Code of the proposed method for VI-ReID on SYSU-MM01 dataset [1] and  RegDB dataset [2]. 

## :wrench: Dependencies and Installation

- Python >= 3.7 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.7](https://pytorch.org/)
- Option: NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
- Option: Linux

### 1. Prepare the datasets.

- run `python pre_process_sysu.py` to pepare the dataset, the training data will be stored in ".npy" format.
- you may need to change the 'data_path' for the SYSU-MM01 dataset.

### 2. Training.
Train a model by

```bash
python train.py --dataset sysu --lr 0.00035 --batch-size 8 --gpu 0
```

- `--dataset`: which dataset "sysu" or "regdb".
- `--lr`: initial learning rate.
- `--batch-size`: person identities.
- `--gpu`:  which gpu to run.

You may need mannully define the data path first.

**Parameters**: More parameters can be found in the code.

**Training Log**: The training log will be saved in `log/" dataset_name"+ log`. Model will be saved in `save_model/`.

### 3. Testing.

Test a model on SYSU-MM01 or RegDB dataset.
```bash
python test.py --mode all --resume 'model_path' --gpu 0 --dataset sysu
```
- `--dataset`: which dataset "sysu" or "regdb".

- `--mode`: "all" or "indoor" all search or indoor search (only for sysu dataset).

- `--trial`: testing trial (only for RegDB dataset).

- `--resume`: the saved model path.

- `--gpu`:  which gpu to run.

###  4. References

[1] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.

### 5. scroll: BibTeX

```
@inproceedings{cheng2023cross,
  title={Cross-modality person re-identification with memory-based contrastive embedding},
  author={Cheng, De and Wang, Xiaolong and Wang, Nannan and Wang, Zhen and Wang, Xiaoyu and Gao, Xinbo},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={1},
  pages={425--432},
  year={2023}
}
```

If you have any question or collaboration need (research purpose or commercial purpose), please email `dcheng@xidian.edu.cn`.
