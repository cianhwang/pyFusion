import argparse
import time

arg_lists = []
parser = argparse.ArgumentParser(description='RFC')


def str2bool(v):
    return v.lower() in ('true', '1')


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


# data params
data_arg = add_argument_group('Data Params')
data_arg.add_argument('--batch_size', type=int, default=4,
                      help='# of images in each batch of data')
data_arg.add_argument('--seq', type=int, default=4,
                      help='#seq of images in each batch of data')
data_arg.add_argument('--std', type=int, default=0.17,
                      help='model distr std')

# training params
train_arg = add_argument_group('Training Params')
train_arg.add_argument('--is_train', type=str2bool, default=True,
                       help='Whether to train or test the model')
train_arg.add_argument('--epochs', type=int, default=100,
                       help='# of epochs to train for')
train_arg.add_argument('--init_lr', type=float, default=3e-4,
                       help='Initial learning rate value')

# other params
misc_arg = add_argument_group('Misc.')
misc_arg.add_argument('--use_cuda', type=str2bool, default=True,
                      help="Whether to run on the GPU")
misc_arg.add_argument('--random_seed', type=int, default=38,
                      help='Seed to ensure reproducibility')
misc_arg.add_argument('--ckpt_dir', type=str, default='./ckpt/'+time.strftime("%m%d%y_%H_%M"),
                      help='Directory in which to save model checkpoints')
misc_arg.add_argument('--logs_dir', type=str, default='./runs/'+time.strftime("%m%d%y_%H_%M"),
                      help='Directory in which Tensorboard logs wil be stored')
misc_arg.add_argument('--use_tensorboard', type=str2bool, default=True,
                      help='Whether to use tensorboard for visualization')
misc_arg.add_argument('--resume', type=str2bool, default=False,
                      help='Whether to resume training from checkpoint')
misc_arg.add_argument('--is_plot', type=str2bool, default=True,
                      help='Whether to plot outcome to tensorboard')

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

if __name__ == '__main__':
    config, unparsed = get_config()
    print(config)
