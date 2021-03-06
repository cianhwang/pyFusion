import torch

from trainer import Trainer
from config import get_config
from data_loader import load_davis_dataset
from af_dataloader import load_af_dataset


def main(config):

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    if config.use_cuda:
        torch.cuda.manual_seed(config.random_seed)

    # instantiate data loaders
#     if config.is_train:
#     data_loader = load_davis_dataset(config.video_path, config.depth_path, config.seq, config.batch_size)
    data_loader = load_af_dataset(config.video_path, config.batch_size)

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        trainer.train()
    else:
        raise NotImplementedError("processing test dataset is not ready")
        trainer.test()

    

if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
