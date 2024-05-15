# Cross-Modality Person Re-Identification with Memory-based Contrastive Embedding

Pytorch Code of the proposed method for VI-ReID on SYSU-MM01 dataset [1] and  RegDB dataset [2]. 

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

  

###  5. References

[1] A. Wu, W.-s. Zheng, H.-X. Yu, S. Gong, and J. Lai. Rgb-infrared crossmodality person re-identification. In IEEE International Conference on Computer Vision (ICCV), pages 5380–5389, 2017.

[2] D. T. Nguyen, H. G. Hong, K. W. Kim, and K. R. Park. Person recognition system based on a combination of body images from visible light and thermal cameras. Sensors, 17(3):605, 2017.