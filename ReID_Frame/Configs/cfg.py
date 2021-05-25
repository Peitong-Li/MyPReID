"""
@author: Lpt
@email: li2820447@163.com
@file: cfg.py
@time: 2021/4/21 15:05
@desc: the configs of project running
"""


from __future__ import absolute_import
from Utils import data_manager
import Modules
import argparse



class Config(object):
    def __init__(self):
        self.args = self.get_args()
    def get_args(self):
        parser = argparse.ArgumentParser(description='Train AlignedReID with cross entropy loss and triplet hard loss')
        # Datasets
        parser.add_argument('--root', type=str, default='E:\WorkSpace\Datasets', help="root path to data directory")
        parser.add_argument('-d', '--dataset', type=str, default='dukemtmcreid',
                            choices=data_manager.get_names())
        parser.add_argument('-j', '--workers', default=0, type=int,
                            help="number of data loading workers (default: 0)")
        parser.add_argument('--height', type=int, default=256,
                            help="height of an image (default: 256)")
        parser.add_argument('--width', type=int, default=128,
                            help="width of an image (default: 128)")

        # CUHK03-specific setting
        parser.add_argument('--split_id', type=int, default=0, help="split index")
        parser.add_argument('--cuhk03-labeled', action='store_true',
                            help="whether to use labeled images, if false, detected images are used (default: False)")
        parser.add_argument('--cuhk03-classic-split', action='store_true',
                            help="whether to use classic split by Li et al. CVPR'14 (default: False)")
        parser.add_argument('--use-metric-cuhk03', action='store_true', default=False,
                            help="whether to use cuhk03-metric (default: False)")
        # Optimization options
        parser.add_argument('--labelsmooth', action='store_true', help="label smooth")
        parser.add_argument('--optim', type=str, default='sgd', help="optimization algorithm (see optimizers.py)")
        parser.add_argument('--max_epoch', default=150, type=int,
                            help="maximum epochs to run")
        parser.add_argument('--start-epoch', default=0, type=int,
                            help="manual epoch number (useful on restarts)")
        parser.add_argument('--train-batch', default=32, type=int,
                            help="train batch size")
        parser.add_argument('--test-batch', default=32, type=int, help="test batch size")
        parser.add_argument('--lr', '--learning_rate', default=0.0002, type=float,
                            help="initial learning rate")
        parser.add_argument('--stepsize', default=30, type=int,
                            help="stepsize to decay learning rate (>0 means this is enabled)")
        parser.add_argument('--gamma', default=0.1, type=float,
                            help="learning rate decay")
        parser.add_argument('--weight_decay', default=5e-04, type=float,
                            help="weight decay (default: 5e-04)")

        # type of deep learn
        parser.add_argument('--loss_type', type=int, default=1,
                            help="Three types:1.{'softmax','metric'}  2.{'softmax'}  3.{'metric'}")

        # triplet hard loss
        parser.add_argument('--margin', type=float, default=0.3, help="margin for triplet loss")
        parser.add_argument('--num-instances', type=int, default=4,
                            help="number of instances per identity")
        parser.add_argument('--htri-only', action='store_true', default=False,
                            help="if this is True, only hardtriplet loss is used in training")
        # Architecture
        parser.add_argument('-a', '--arch', type=str, default='resnet50', choices=Modules.get_names())
        # Miscs
        parser.add_argument('--print_freq', type=int, default=10, help="print frequency")
        parser.add_argument('--seed', type=int, default=1, help="manual seed")
        parser.add_argument('--resume', type=str, default='', metavar='PATH')
        parser.add_argument('--evaluate', action='store_true', help="evaluation only")
        parser.add_argument('--eval_step', type=int, default=50,
                            help="run evaluation for every N epochs (set to -1 to test after training)")
        parser.add_argument('--start-eval', type=int, default=0, help="start to evaluate after specific epoch")
        parser.add_argument('--save_dir', type=str, default='./log')
        parser.add_argument('--use_cpu', action='store_true', help="use cpu")
        parser.add_argument('--gpu_devices', default='0', type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
        parser.add_argument('--reranking', action='store_true', help='result re_ranking')

        # parser.add_argument('--test_distance', type=str, default='global', help='test distance type')
        # parser.add_argument('--unaligned', action='store_true', help='test local feature with unalignment')

        args = parser.parse_args()
        loss_factory = [{'softmax', 'metric'}, {'softmax'}, {'metric'}]
        args.loss_type = loss_factory[int(args.loss_type)-1]
        return args

args = Config().args