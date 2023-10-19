import os
import argparse
from solver import Solver

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

parser = argparse.ArgumentParser('Perceptual Reflection Removel')

parser.add_argument('--ref_syn_dir',
                    default=r"",
                    help="path to synthetic data")
parser.add_argument('--ref_real_dir',
                    default=r"",
                    help="path to real89 data")

parser.add_argument('--save_model_freq', default=1, type=int, help="frequency to save model")
parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
parser.add_argument('--resume_file', default=r'', help="resume file path")
parser.add_argument('--lr', default=2e-4, type=float, help="learning rate")     # 2e-4 --> 5e-5
parser.add_argument('--load_workers', default=4, type=int, help="number of workers to load data")
parser.add_argument('--batch_size', default=2, type=int, help="batch size")
parser.add_argument('--start_epoch', type=int, default=0, help="start epoch of training")
parser.add_argument('--num_epochs', type=int, default=200, help="total epoch of training")


def main():
    args = parser.parse_args()
    solver = Solver(args)
    solver.train_model()


if __name__ == '__main__':
    main()
