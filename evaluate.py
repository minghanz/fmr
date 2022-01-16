"""
evaluate the feature-metric registration algorithm
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
import model
from data import dataset
from data.get_dataset_onet import get_dataset
import argparse
import os
import sys
import copy
import open3d
import torch
import torch.utils.data
import logging
import yaml

# LOGGER = logging.getLogger(__name__)
# LOGGER.addHandler(logging.NullHandler())

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # used GPU card no.

# visualize the point clouds
def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    open3d.io.write_point_cloud("source_pre.ply", source_temp)
    source_temp.transform(transformation)
    open3d.io.write_point_cloud("source.ply", source_temp)
    open3d.io.write_point_cloud("target.ply", target_temp)
    open3d.visualization.draw_geometries([source_temp, target_temp])


def options(argv=None):
    parser = argparse.ArgumentParser(description='Feature-metric registration')

    # required to check
    parser.add_argument('-data', '--dataset-type', default='7scene', choices=['modelnet', '7scene'],
                        metavar='DATASET', help='dataset type (default: modelnet)')
    parser.add_argument('-o', '--outfile', default='./result/result.csv', type=str,
                        metavar='FILENAME', help='output filename (.csv)')
    parser.add_argument('-p', '--perturbations', default='./data/pert_010.csv', type=str,
                        metavar='PATH', help='path to the perturbation file')  # run
    parser.add_argument('-l', '--logfile', default='./result/log_010.log', type=str,
                        metavar='LOGNAME', help='path to logfile (default: null (no logging))')
    parser.add_argument('--pretrained', default='./result/fmr_model_7scene.pth', type=str,
                        metavar='PATH', help='path to trained model file (default: null (no-use))')

    # settings for performance adjust
    parser.add_argument('--max-iter-val', default=10, type=int,
                        metavar='N', help='max-iter on IC algorithm. (default: 20)')
    parser.add_argument('--dim-k', default=1024, type=int,
                        metavar='K', help='dim. of the feature vector (default: 1024)')
    # settings for on testing
    parser.add_argument('-j', '--workers', default=2, type=int,
                        metavar='N', help='number of data loading workers (default: 4)')
    parser.add_argument('--device', default='cuda', type=str,
                        metavar='DEVICE', help='use CUDA if available (default: cpu)')
    parser.add_argument('-i', '--dataset-path', default='', type=str,
                        metavar='PATH', help='path to the input dataset')  # like '/path/to/ModelNet40'
    parser.add_argument('-c', '--categoryfile', default='', type=str,
                        metavar='PATH',
                        help='path to the categories to be tested')  # eg. './sampledata/modelnet40_half1.txt'
    parser.add_argument('--mode', default='test', help='program mode. This code is for testing')
    # ### https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    # parser.add_argument('--uniformsampling', dest='uniformsampling', action='store_true', help='uniform sampling points from the mesh')    # Minghan: if not set, this could be 10k to more than 100k
    # parser.add_argument('--no_uniformsampling', dest='uniformsampling', action='store_false', help='dsable uniform sampling points from the mesh')    # Minghan: if not set, this could be 10k to more than 100k
    # parser.set_defaults(uniformsampling=False)
    # parser.add_argument('--mag', default=5, type=float,
    #                     metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    # parser.add_argument('--mag_trans', default=0.8, type=float,
    #                     metavar='T', help='max. mag. of twist-vectors (perturbations) on training (default: 0.8)')
    # parser.add_argument('--num-points', default=1024, type=int,
    #                     metavar='N', help='points in point-cloud (default: 1024)')
    # parser.add_argument('--noise', action='store_true',
    #                     help='max. mag. of noise on training (default: 0.8)')
    # parser.add_argument('--resampling', action='store_true',
    #                     help='max. mag. of noise on training (default: 0.8)')
    # parser.add_argument('--density', action='store_true',
    #                     help='max. mag. of noise on training (default: 0.8)')
    parser.add_argument('--duo-mode', action='store_true',
                        help='use adjacent frames instead of the same frame for registration')
    parser.add_argument('--cfg', type=str, default=None,
                        help='cfg for onet-like dataset')
    parser.add_argument('--tf_cfg', type=str, default=None,
                        help='cfg for transforms')
    parser.add_argument('--param_cfg', type=str, default=None,
                        help='cfg for transform parameters')
    
    args = parser.parse_args(argv)
    return args

def save_cfgs(args, params):
    dir = os.path.dirname(args.outfile)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(args.outfile+"_params.yaml", 'w') as f:
        yaml.dump(params, f)
    with open(args.outfile+"_args.yaml", 'w') as f:
        yaml.dump(vars(args), f)
    return

def main(args):
    # dataset
    with open(args.param_cfg) as f:
        params = yaml.safe_load(f)
    save_cfgs(args, params)
    
    if args.cfg is None:
        print("loading fmr dataset")
        testset = dataset.get_datasets(args, params)
    else:
        print("loading onet dataset")
        with open(args.cfg) as f:
            cfg_dict = yaml.load(f)
        testset = get_dataset('test', cfg_dict, return_idx=True, return_category=True, )

    # testing
    fmr = model.FMRTest(args, params)
    run(args, testset, fmr)


def run(args, testset, action):
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)

    model = action.create_model()
    if args.pretrained:
        assert os.path.isfile(args.pretrained)
        checkpoint = torch.load(args.pretrained, map_location='cpu')
        if 'model' in checkpoint:
            # The checkpoints saved by train.py have model, optimizer, etc. Take the model here. 
            model.load_state_dict(checkpoint['model'])
        else:
            # The checkpoint of pretrained models only have the network model parameters.
            model.load_state_dict(checkpoint)
    model.to(args.device)

    # dataloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=args.workers)

    # testing
    logging.debug('tests, begin')
    action.evaluate(model, testloader, args.device)
    logging.debug('tests, end')


if __name__ == '__main__':
    ARGS = options()

    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s:%(name)s, %(asctime)s, %(message)s',
        filename=ARGS.logfile)

    main(ARGS)
    logging.debug('done (PID=%d)', os.getpid())
