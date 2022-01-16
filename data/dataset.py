"""
This file is used to load 3D point cloud for network training
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
import numpy
import torch.utils.data
import os
import glob
import copy
import six
import numpy as np
import torch
import torch.utils.data
import torchvision

import se_math.se3 as se3
import se_math.so3 as so3
import se_math.mesh as mesh
import se_math.transforms as transforms

import yaml

"""
The following three functions are defined for getting data from specific database 
"""


# find the total class names and its corresponding index from a folder
# (see the data storage structure of modelnet40)
def find_classes(root):
    """ find ${root}/${class}/* """
    classes = [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the indexes from given class names
def classes_to_cinfo(classes):
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


# get the whole 3D point cloud paths for a given class
def glob_dataset(root, class_to_idx, ptns):
    """ glob ${root}/${class}/${ptns[i]} """
    root = os.path.expanduser(root)
    samples = []

    # loop all the folderName (class name) to find the class in class_to_idx
    for target in sorted(os.listdir(root)):
        d = os.path.join(root, target)
        if not os.path.isdir(d):
            continue
        # check if it is the class we want
        target_idx = class_to_idx.get(target)
        if target_idx is None:
            continue
        # to find the all point cloud paths in the class folder
        for i, ptn in enumerate(ptns):
            gptn = os.path.join(d, ptn)
            names = glob.glob(gptn)
            for path in sorted(names):
                item = (path, target_idx)
                samples.append(item)
    return samples


def T44_from_txt(txt_path):
    with open(txt_path) as f:
        lines = f.readlines()
        linesmat = lines[1:5]
        mat = [[float(x) for x in line.split()] for line in linesmat]
        mat = np.array(mat)
        # print(mat.shape, mat) #4*4
    return mat

# a general class for obtaining the 3D point cloud data from a database
class PointCloudDataset(torch.utils.data.Dataset):
    """ glob ${rootdir}/${classes}/${pattern}
    """

    def __init__(self, rootdir, pattern, fileloader, transform=None, classinfo=None):
        super().__init__()

        if isinstance(pattern, six.string_types):
            pattern = [pattern]

        # find all the class names
        if classinfo is not None:
            classes, class_to_idx = classinfo
        else:
            classes, class_to_idx = find_classes(rootdir)

        # get all the 3D point cloud paths for the class of class_to_idx
        samples = glob_dataset(rootdir, class_to_idx, pattern)
        if not samples:
            raise RuntimeError("Empty: rootdir={}, pattern(s)={}".format(rootdir, pattern))

        self.fileloader = fileloader
        self.transform = transform

        self.classes = classes
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        define the getitem function for Dataloader of torch
        load a 3D point cloud by using a path index
        :param index:
        :return:
        """
        path, target = self.samples[index]
        try:
            sample = self.fileloader(path)
        except Exception as e:
            print(e)
            return self.__getitem__(index+1)

        ### the mat is only used in duo_mode (see TransformedDataset)
        mat = None
        txt = path.replace('.ply', '.info.txt')
        if os.path.exists(txt):
            mat = T44_from_txt(txt)
            mat = torch.from_numpy(mat).to(dtype=torch.float)

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target, mat

    def split(self, rate):
        """ dateset -> dataset1, dataset2. s.t.
            len(dataset1) = rate * len(dataset),
            len(dataset2) = (1-rate) * len(dataset)
        """
        orig_size = len(self)
        select = np.zeros(orig_size, dtype=int)
        csize = np.zeros(len(self.classes), dtype=int)

        for i in range(orig_size):
            _, target = self.samples[i]
            csize[target] += 1
        dsize = (csize * rate).astype(int)
        for i in range(orig_size):
            _, target = self.samples[i]
            if dsize[target] > 0:
                select[i] = 1
                dsize[target] -= 1

        dataset1 = copy.deepcopy(self)
        dataset2 = copy.deepcopy(self)

        samples1 = list(map(lambda i: dataset1.samples[i], np.where(select == 1)[0]))
        samples2 = list(map(lambda i: dataset2.samples[i], np.where(select == 0)[0]))

        dataset1.samples = samples1
        dataset2.samples = samples2
        return dataset1, dataset2


class ModelNet(PointCloudDataset):
    """ [Princeton ModelNet](http://modelnet.cs.princeton.edu/) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None, is_uniform_sampling=False):
        # if you would like to uniformly sampled points from mesh, use this function below
        if is_uniform_sampling:
            loader = mesh.offread_uniformed # used uniformly sampled points.
        else:
            loader = mesh.offread # use the original vertex in the mesh file
        if train > 0:
            pattern = 'train/*.off'
        elif train == 0:
            pattern = 'test/*.off'
        else:
            pattern = ['train/*.off', 'test/*.off']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class ShapeNet2(PointCloudDataset):
    """ [ShapeNet](https://www.shapenet.org/) v2 """

    def __init__(self, dataset_path, transform=None, classinfo=None):
        loader = mesh.objread
        pattern = '*/models/model_normalized.obj'
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class Scene7(PointCloudDataset):
    """ [Scene7 PointCloud](https://github.com/XiaoshuiHuang/fmr) """

    def __init__(self, dataset_path, train=1, transform=None, classinfo=None):
        loader = mesh.plyread
        if train > 0:
            pattern = '*.ply'
        elif train == 0:
            pattern = '*.ply'
        else:
            pattern = ['*.ply', '*.ply']
        super().__init__(dataset_path, pattern, loader, transform, classinfo)


class TransformedDataset(torch.utils.data.Dataset):
    # def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None, resampling=False, duo_mode=False):
    def __init__(self, dataset, rigid_transform, source_modifier=None, template_modifier=None, duo_mode=False):
        self.dataset = dataset
        self.rigid_transform = rigid_transform
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

        # self.rigid_transform_both = transforms.RandomTransformSE3(180, True, 0)
        # self.resampling = resampling
        self.duo_mode = duo_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        pm, _, mat1 = self.dataset[index]

        ### template (target)
        p0 = pm
        if self.duo_mode:
            if index < len(self) - 1:
                p0, _, mat0 = self.dataset[index+1]
            else:
                p0, _, mat0 = self.dataset[index-1]
            assert mat1 is not None and mat0 is not None, f"{mat0},{mat1}"
            mat01 = torch.matmul(torch.inverse(mat1), mat0)
            p0 = torch.matmul(mat01[:3, :3], p0.transpose(0,1)) + mat01[:3, [3]]
            p0 = p0.transpose(0,1)

        if self.template_modifier is not None:
            p0 = self.template_modifier(p0)
                
        ### source
        p1 = pm
        if self.source_modifier is not None:
            p1 = self.source_modifier(p1)

        p1 = self.rigid_transform(p1)
        igt = self.rigid_transform.igt

        # p0: template, p1: source, igt: transform matrix from p0 to p1 (T10)
        ### The official result we want at the end is T01 (the transformation applied on source p1 to align with target p0). 
        ### Therefore this igt (T10) is for conveniently evaluating the error of T01 by multiplication. 
        return p0, p1, igt


class TransformedFixedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, perturbation, source_modifier=None, template_modifier=None):
        self.dataset = dataset
        self.perturbation = numpy.array(perturbation)  # twist (len(dataset), 6)
        self.source_modifier = source_modifier
        self.template_modifier = template_modifier

    def do_transform(self, p0, x):
        # p0: [N, 3]
        # x: [1, 6]
        # x: rotation and translation
        w = x[:, 0:3]
        q = x[:, 3:6]
        R = so3.exp(w).to(p0)  # [1, 3, 3]
        g = torch.zeros(1, 4, 4)
        g[:, 3, 3] = 1
        g[:, 0:3, 0:3] = R  # rotation
        g[:, 0:3, 3] = q  # translation
        p1 = se3.transform(g, p0)
        igt = g.squeeze(0)  # igt: p0 -> p1
        return p1, igt

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        twist = torch.from_numpy(numpy.array(self.perturbation[index])).contiguous().view(1, 6)
        pm, _, _ = self.dataset[index]
        x = twist.to(pm)

        p1 = pm
        if self.source_modifier is not None:
            p1 = self.source_modifier(p1)
            
        p1, igt = self.do_transform(p1, x)

        p0 = pm
        if self.template_modifier is not None:
            p0 = self.template_modifier(p0)

        # p0: template, p1: source, igt: transform matrix from p0 to p1
        return p0, p1, igt


def get_categories(args):
    cinfo = None
    if args.categoryfile:
        categories = [line.rstrip('\n') for line in open(args.categoryfile)]
        categories.sort()
        c_to_idx = {categories[i]: i for i in range(len(categories))}
        cinfo = (categories, c_to_idx)
    return cinfo

def save_transforms(tf_dict, path_out):
    tfstr_dict = dict()
    for key in tf_dict:
        tfstr_dict[key] = [repr(x) for x in tf_dict[key]]
    dir = os.path.dirname(path_out)
    if not os.path.exists(dir):
        os.makedirs(dir)
    with open(path_out, 'w') as f:
        yaml.dump(tfstr_dict, f)
    return

def get_transforms(cfg_yaml, params):
    ### params is used in eval()
    with open(cfg_yaml) as f:
        tfstr_dict = yaml.safe_load(f)
    tf_dict = dict()
    local_scope = locals()
    global_scope = globals()
    global_scope.update(local_scope)
    for key in tfstr_dict:
        tf_dict[key] = [eval("transforms."+x, global_scope) for x in tfstr_dict[key]]
    return tfstr_dict, tf_dict

# def get_params(cfg_yaml):
#     with open(cfg_yaml) as f:
#         params = yaml.safe_load(f)
#     return params

# global dataset function, could call to get dataset
def get_datasets(args, params):
    tfstr_dict, tf_dict = get_transforms(args.tf_cfg, params)
    transform_shared = torchvision.transforms.Compose(tf_dict["shared"])
    transform_source = torchvision.transforms.Compose(tf_dict["source"])
    transform_template = torchvision.transforms.Compose(tf_dict["target"])
    tf_savepath = args.outfile+"_tfs.yaml"
    save_transforms(tf_dict, tf_savepath)

    if args.dataset_type == 'modelnet':
        # download modelnet40 for training
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        DATA_DIR = os.path.join(BASE_DIR, 'ModelNet40')
        if not os.path.exists(DATA_DIR):
            www = 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
            zipfile = os.path.basename(www)
            os.system('wget %s; unzip %s' % (www, zipfile))
            os.system('mv %s %s' % (zipfile[:-4], BASE_DIR))
            os.system('rm %s' % (zipfile))
        if not os.path.exists(DATA_DIR):
            exit(
                "Please download ModelNET40 and put it in the data folder, the download link is http://modelnet.cs.princeton.edu/ModelNet40.zip")

        # if args.mode == 'train':
        # # if True:
        #     # set path and category file for training
        #     args.dataset_path = './data/ModelNet40'
        #     args.categoryfile = './data/categories/modelnet40_half1.txt'      # 1202 models
        #     # args.categoryfile = './data/categories/modelnet40.txt'      # 2468 models
        #     cinfo = get_categories(args)
        #     transform = torchvision.transforms.Compose([ \
        #         transforms.Mesh2Points(), \
        #         transforms.OnUnitCube(), \
        #         transforms.Resampler(args.num_points), \
        #         ])
        #     traindata = ModelNet(args.dataset_path, train=1, transform=transform, classinfo=cinfo, is_uniform_sampling=args.uniformsampling)
        #     testdata = ModelNet(args.dataset_path, train=0, transform=transform, classinfo=cinfo, is_uniform_sampling=args.uniformsampling)
        #     mag_randomly = False #True
        #     trainset = TransformedDataset(traindata, transforms.RandomTransformSE3(args.mag, mag_randomly, args.mag_trans), resampling=args.resampling)
        #     testset = TransformedDataset(testdata, transforms.RandomTransformSE3(args.mag, mag_randomly, args.mag_trans), resampling=args.resampling)
        #     # return trainset, testset
        #     return testset

        if args.mode == 'train':
            # set path and category file for training
            args.dataset_path = './data/ModelNet40'
            args.categoryfile = './data/categories/modelnet40_half1.txt'      # 1202 models
            # args.categoryfile = './data/categories/modelnet40.txt'      # 2468 models
            cinfo = get_categories(args)

            traindata = ModelNet(args.dataset_path, train=1, transform=transform_shared, classinfo=cinfo, is_uniform_sampling=params['uniformsampling'])
            testdata = ModelNet(args.dataset_path, train=0, transform=transform_shared, classinfo=cinfo, is_uniform_sampling=params['uniformsampling'])
            
            trainset = TransformedDataset(traindata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                        transform_source, transform_template)
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                        transform_source, transform_template)
            return trainset, testset
        else:
            # set path and category file for test
            args.dataset_path = './data/ModelNet40'
            args.categoryfile = './data/categories/modelnet40_half1.txt'
            # args.categoryfile = './data/categories/modelnet40.txt'
            cinfo = get_categories(args)

            # get the ground truth perturbation
            fixed_rigid_transform = False
            if args.perturbations:
                perturbations = numpy.loadtxt(args.perturbations, delimiter=',')
                fixed_rigid_transform = True

            testdata = ModelNet(args.dataset_path, train=0, transform=transform_shared, classinfo=cinfo, is_uniform_sampling=params['uniformsampling'])
            if fixed_rigid_transform:
                testset = TransformedFixedDataset(testdata, perturbations, transform_source, transform_template)
            else:
                testset = TransformedDataset(testdata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                            transform_source, transform_template)
            return testset

    elif args.dataset_type == '7scene':
        if args.mode == 'train':
            # set path and category file for training
            args.dataset_path = './data/7scene'
            args.categoryfile = './data/categories/7scene_train.txt'
            cinfo = get_categories(args)

            dataset = Scene7(args.dataset_path, transform=transform_shared, classinfo=cinfo)
            traindata, testdata = dataset.split(0.8)

            trainset = TransformedDataset(traindata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                        transform_source, transform_template, duo_mode=args.duo_mode)
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                        transform_source, transform_template, duo_mode=args.duo_mode)
            return trainset, testset
        else:
            # set path and category file for testing
            args.dataset_path = './data/7scene'
            args.categoryfile = './data/categories/7scene_test.txt'
            cinfo = get_categories(args)

            testdata = Scene7(args.dataset_path, transform=transform_shared, classinfo=cinfo)

            # randomly generate transformation matrix
            testset = TransformedDataset(testdata, transforms.RandomTransformSE3(params['mag_deg'], params['mag_trans'], params['random_deg'], params['random_trans']), \
                        transform_source, transform_template, duo_mode=args.duo_mode)
            return testset
