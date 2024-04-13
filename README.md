NYUResnetChallenge
==========

- [kaggle challenge] https://www.kaggle.com/competitions/deep-learning-mini-project-spring-24-nyu
- [Overleaf doc] https://www.overleaf.com/project/660af2c2717a06ecc69327f9
- [google doc] https://docs.google.com/document/d/1fwO-1mw84OI6Rt7TnOZ2jH_78WhFtmXr9VgSUBSXC0I/edit?usp=sharing

## Model

Trained and modified ResNet Architecture on image classification CIFAR 10 Dataset. The model architecture is based on the paper "Deep Residual Learning for Image Recognition" by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun ([arXiv:1512.03385](https://arxiv.org/abs/1512.03385)).

## Data

The script is designed to work with the CIFAR-10 dataset. The dataset is divided into labeled and unlabeled data. 

## Results

The script logs the training progress and results to Weights & Biases (wandb). You can view the results on the [wandb dashboard](https://wandb.ai/your-username/your-project-name).


## Environment Setting
### Set Environment with conda
```shell
conda env create -f DL-MiniProj-Env.yml
conda activate DL-MiniProj
```

### Set Environment with pip
```shell
pip install -r requirements.txt
```

## Test model with Tap Args
```shell
python trainer.py --num_blocks <str list> --n_epochs <Number of Epochs> --model_arch <Model Architecture> --in_planes <Number of Planes>

# e.g. python trainer.py --lr 0.01 --momentum 0.9 --weight_decay 0.0001 --n_epochs 100 --batch_size 20
# --num_workers 4 --valid_size 0.2 --optimizer 'SGD' --self_supervise --test_data_path '/path/to/test_data'
# --test_nolabel_path '/path/to/test_nolabel' --data_dir '/path/to/data' --model_arch 'custom' --stride 1
# --in_planes 64 --num_channels '64,128,256,512' --filter_sizes '3,3,3,3' --num_blocks '2,2,2,2'
```

## Command-line Arguments
- `--lr LR`: Set the learning rate for the optimizer.
- `--momentum MOMENTUM`: Set the momentum factor for the optimizer.
- `--weight_decay WEIGHT_DECAY`: Set the weight decay (L2 penalty) for the optimizer.
- `--n_epochs N_EPOCHS`: Specify the number of training epochs.
- `--start_epoch START_EPOCH`: Set the starting epoch number.
- `--best_acc BEST_ACC`: Input the best accuracy achieved (used for resuming training).
- `--batch_size BATCH_SIZE`: Define the batch size for training.
- `--num_workers NUM_WORKERS`: Set the number of workers for data loading.
- `--valid_size VALID_SIZE`: Specify the size of the validation set as a fraction of the total dataset.
- `--optimizer OPTIMIZER`: Choose the optimizer for training.
- `--self_supervise`: Enable self-supervised learning mode (use `--self_supervise=True` to enable).
- `--test_data_path TEST_DATA_PATH`: Path to the test dataset.
- `--test_nolabel_path TEST_NOLABEL_PATH`: Path to the unlabeled test dataset.
- `--data_dir DATA_DIR`: Directory where the training/validation data is stored.
- `--model_arch MODEL_ARCH`: Choose the model architecture from predefined options or custom.
- `--stride STRIDE`: Set the stride value for convolutional layers.
- `--in_planes IN_PLANES`: Define the number of input planes (channels) for the model.
- `--num_channels NUM_CHANNELS`: Specify the number of channels for each block in the network.
- `--filter_sizes FILTER_SIZES`: Set the filter sizes for each block in the network.
- `--num_blocks NUM_BLOCKS`: Determine the number of blocks for the custom network architecture.

## Collaborators
- Tyler Li w.li@nyu.edu
- Yihao Wang yihao.w@nyu.edu
- Jin Qin jq2325@nyu.edu

## References
https://colab.research.google.com/drive/1Jwgn8r6TrNPgZh5uX8xMSCaLzhlsHHRx

https://colab.research.google.com/github/Rakshit-Shetty/Resnet-Implementation/blob/master/ResNet_Implementation_on_CIFAR10.ipynb

https://github.com/kuangliu/pytorch-cifar?tab=readme-ov-file

[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385

Dataset: [CIFAR 10] https://www.cs.toronto.edu/~kriz/cifar.html
```bibtex
@article{krizhevsky2009learning,
  title={Learning multiple layers of features from tiny images},
  author={Krizhevsky, Alex and Hinton, Geoffrey},
  journal={Citeseer},
  year={2009}
}

