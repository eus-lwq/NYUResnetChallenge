# NYUResnetChallenge

### kaggle challenge： https://www.kaggle.com/competitions/deep-learning-mini-project-spring-24-nyu
### Overleaf doc： https://www.overleaf.com/project/660af2c2717a06ecc69327f9
### google doc： https://docs.google.com/document/d/1fwO-1mw84OI6Rt7TnOZ2jH_78WhFtmXr9VgSUBSXC0I/edit?usp=sharing

Activate Environment with Conda or
```shell
conda env create -f DL-MiniProj-Env.yml
conda activate DL-MiniProj
```

Activate Environment with Pip
```shell
pip install -r requirements.txt
```

Test model with Tap Args
```shell
python trainer.py --num_blocks <str list> --n_epochs <Number of Epochs> --model_arch <Model Architecture> --in_planes <Number of Planes>
# e.g. python trainer.py --num_blocks '1,1,1,1' --n_epochs 100 --model_arch 'custom' --in_planes 32
```

--lr LR: Set the learning rate for the optimizer.
--momentum MOMENTUM: Set the momentum factor for the optimizer.
--weight_decay WEIGHT_DECAY: Set the weight decay (L2 penalty) for the optimizer.
--n_epochs N_EPOCHS: Specify the number of training epochs.
--start_epoch START_EPOCH: Set the starting epoch number.
--best_acc BEST_ACC: Input the best accuracy achieved (used for resuming training).
--batch_size BATCH_SIZE: Define the batch size for training.
--num_workers NUM_WORKERS: Set the number of workers for data loading.
--valid_size VALID_SIZE: Specify the size of the validation set as a fraction of the total dataset.
--optimizer OPTIMIZER: Choose the optimizer for training.
--self_supervise: Enable self-supervised learning mode.
--test_data_path TEST_DATA_PATH: Path to the test dataset.
--test_nolabel_path TEST_NOLABEL_PATH: Path to the unlabeled test dataset.
--data_dir DATA_DIR: Directory where the training/validation data is stored.
--model_arch MODEL_ARCH: Choose the model architecture from predefined options or custom.
--stride STRIDE: Set the stride value for convolutional layers.
--in_planes IN_PLANES: Define the number of input planes (channels) for the model.
--num_channels NUM_CHANNELS: Specify the number of channels for each block in the network.
--filter_sizes FILTER_SIZES: Set the filter sizes for each block in the network.
--num_blocks NUM_BLOCKS: Determine the number of blocks for the custom network architecture.
