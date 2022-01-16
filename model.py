"""
create model
Creator: Xiaoshui Huang
Date: 2020-06-19
"""
import torch
import numpy as np
from random import sample

import se_math.se3 as se3
import se_math.invmat as invmat
from tqdm import tqdm

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def visualize_pointcloud(points, normals=None,
                         out_file=None, show=False, 
                         points2=None, info=None, 
                         c1=None, c2=None, cm1='viridis', cm2='viridis', s1=5, s2=5):
    r''' Visualizes point cloud data.

    Args:
        points (tensor): point data
        normals (tensor): normal data (if existing)
        out_file (string): output file
        show (bool): whether the plot should be shown
    '''
    # Use numpy
    points = np.asarray(points)
    # Create plot
    fig = plt.figure()
    ax = fig.gca(projection=Axes3D.name)
    if c1 is not None:
        cmap1 = plt.get_cmap(cm1)   # viridis, magma
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=s1, c=c1, cmap=cmap1)
    else:
        ax.scatter(points[:, 2], points[:, 0], points[:, 1], s=s1)
    if points2 is not None:
        if c2 is not None:
            cmap2 = plt.get_cmap(cm2)
            ax.scatter(points2[:, 2], points2[:, 0], points2[:, 1], s=s2, c=c2, cmap=cmap2, marker='^')
        else:
            ax.scatter(points2[:, 2], points2[:, 0], points2[:, 1], 'r', s=s2)
    if normals is not None:
        ax.quiver(
            points[:, 2], points[:, 0], points[:, 1],
            normals[:, 2], normals[:, 0], normals[:, 1],
            length=0.8, color='gray', linewidth=0.8
        )
    ax.set_xlabel('Z')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    ax.view_init(elev=30, azim=45)
    if info is not None:
        plt.title("{}".format(info))

    if out_file is not None:
        plt.savefig(out_file)
    if show:
        plt.show()
    plt.close(fig)

# a global function to flatten a feature
def flatten(x):
    return x.view(x.size(0), -1)


# a global function to calculate max-pooling
def symfn_max(x):
    # [B, K, N] -> [B, K, 1]
    a = torch.nn.functional.max_pool1d(x, x.size(-1))
    return a


# a global function to generate mlp layers
def _mlp_layers(nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
    """ [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """
    layers = []
    last = nch_input
    for i, outp in enumerate(nch_layers):
        if b_shared:
            weights = torch.nn.Conv1d(last, outp, 1)
        else:
            weights = torch.nn.Linear(last, outp)
        layers.append(weights)
        layers.append(torch.nn.BatchNorm1d(outp, momentum=bn_momentum))
        layers.append(torch.nn.ReLU())
        if b_shared == False and dropout > 0.0:
            layers.append(torch.nn.Dropout(dropout))
        last = outp
    return layers


# a class to generate MLP network
class MLPNet(torch.nn.Module):
    """ Multi-layer perception.
        [B, Cin, N] -> [B, Cout, N] or
        [B, Cin] -> [B, Cout]
    """

    def __init__(self, nch_input, nch_layers, b_shared=True, bn_momentum=0.1, dropout=0.0):
        super().__init__()
        list_layers = _mlp_layers(nch_input, nch_layers, b_shared, bn_momentum, dropout)
        self.layers = torch.nn.Sequential(*list_layers)

    def forward(self, inp):
        out = self.layers(inp)
        return out


# encoder network
class PointNet(torch.nn.Module):
    def __init__(self, dim_k=1024):
        super().__init__()
        scale = 1
        mlp_h1 = [int(64 / scale), int(64 / scale)]
        mlp_h2 = [int(64 / scale), int(128 / scale), int(dim_k / scale)]

        self.h1 = MLPNet(3, mlp_h1, b_shared=True).layers
        self.h2 = MLPNet(mlp_h1[-1], mlp_h2, b_shared=True).layers
        self.sy = symfn_max

    def forward(self, points):
        """ points -> features
            [B, N, 3] -> [B, K]
        """
        # for pointnet feature extraction
        x = points.transpose(1, 2)  # [B, 3, N]
        x = self.h1(x)
        x = self.h2(x)  # [B, K, N]
        x = flatten(self.sy(x))

        return x


# decoder network
class Decoder(torch.nn.Module):
    def __init__(self, num_points=2048, bottleneck_size=1024):
        super(Decoder, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.bn1 = torch.nn.BatchNorm1d(bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(bottleneck_size // 2)
        self.bn3 = torch.nn.BatchNorm1d(bottleneck_size // 4)
        self.fc1 = torch.nn.Linear(self.bottleneck_size, bottleneck_size)
        self.fc2 = torch.nn.Linear(self.bottleneck_size, bottleneck_size // 2)
        self.fc3 = torch.nn.Linear(bottleneck_size // 2, bottleneck_size // 4)
        self.fc4 = torch.nn.Linear(bottleneck_size // 4, self.num_points * 3)
        self.th = torch.nn.Tanh()

    def forward(self, x):
        batchsize = x.size()[0]
        x = torch.nn.functional.relu(self.bn1(self.fc1(x)))
        x = torch.nn.functional.relu(self.bn2(self.fc2(x)))
        x = torch.nn.functional.relu(self.bn3(self.fc3(x)))
        x = self.th(self.fc4(x))
        x = x.view(batchsize, 3, self.num_points).transpose(1, 2).contiguous()
        return x


# the neural network of feature-metric registration
class SolveRegistration(torch.nn.Module):
    def __init__(self, ptnet, decoder=None, isTest=False):
        super().__init__()
        # network
        self.encoder = ptnet
        self.decoder = decoder

        # functions
        self.inverse = invmat.InvMatrix.apply
        self.exp = se3.Exp  # [B, 6] -> [B, 4, 4]
        self.transform = se3.transform  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]

        # initialization for dt: [w1, w2, w3, v1, v2, v3], 3 rotation angles and 3 translation
        delta = 1.0e-2  # step size for approx. Jacobian (default: 1.0e-2)
        dt_initial = torch.autograd.Variable(torch.Tensor([delta, delta, delta, delta, delta, delta]))
        self.dt = torch.nn.Parameter(dt_initial.view(1, 6), requires_grad=True)

        # results
        self.last_err = None
        self.g_series = None  # for debug purpose
        self.prev_r = None
        self.g = None  # estimated transformation T
        self.isTest = isTest # whether it is testing

    # estimate T
    def estimate_t(self, p0, p1, maxiter=5, xtol=1.0e-7, p0_zero_mean=True, p1_zero_mean=True):
        """
        give two point clouds, estimate the T by using IC algorithm
        :param p0: point cloud
        :param p1: point cloud
        :param maxiter: maximum iteration
        :param xtol: a threshold for early stop of transformation estimation
        :param p0_zero_mean: True: normanize p0 before IC algorithm
        :param p1_zero_mean: True: normanize p1 before IC algorithm
        :return: feature-metric projection error (r), encoder-decoder loss (loss_ende)
        """
        a0 = torch.eye(4).view(1, 4, 4).expand(p0.size(0), 4, 4).to(p0)  # [B, 4, 4]
        a1 = torch.eye(4).view(1, 4, 4).expand(p1.size(0), 4, 4).to(p1)  # [B, 4, 4]
        # normalization
        if p0_zero_mean:
            p0_m = p0.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a0 = a0.clone()
            a0[:, 0:3, 3] = p0_m
            q0 = p0 - p0_m.unsqueeze(1)
        else:
            q0 = p0
        if p1_zero_mean:
            p1_m = p1.mean(dim=1)  # [B, N, 3] -> [B, 3]
            a1 = a1.clone()
            a1[:, 0:3, 3] = -p1_m
            q1 = p1 - p1_m.unsqueeze(1)
        else:
            q1 = p1

        # use IC algorithm to estimate the transformation
        g0 = torch.eye(4).to(q0).view(1, 4, 4).expand(q0.size(0), 4, 4).contiguous()
        r, g, loss_ende = self.ic_algo(g0, q0, q1, maxiter, xtol, is_test=self.isTest)
        self.g = g

        # re-normalization
        if p0_zero_mean or p1_zero_mean:
            # output' = trans(p0_m) * output * trans(-p1_m)
            #        = [I, p0_m;] * [R, t;] * [I, -p1_m;]
            #          [0, 1    ]   [0, 1 ]   [0,  1    ]
            est_g = self.g
            if p0_zero_mean:
                est_g = a0.to(est_g).bmm(est_g)
            if p1_zero_mean:
                est_g = est_g.bmm(a1.to(est_g))
            self.g = est_g
            ### g: T01 (apply on 1 to align with 0)

            est_gs = self.g_series  # [M, B, 4, 4]
            if p0_zero_mean:
                est_gs = a0.unsqueeze(0).contiguous().to(est_gs).matmul(est_gs)
            if p1_zero_mean:
                est_gs = est_gs.matmul(a1.unsqueeze(0).contiguous().to(est_gs))
            self.g_series = est_gs

        return r, loss_ende

    # IC algorithm
    def ic_algo(self, g0, p0, p1, maxiter, xtol, is_test=False):
        """
        use IC algorithm to estimate the increment of transformation parameters
        :param g0: initial transformation
        :param p0: point cloud
        :param p1: point cloud
        :param maxiter: maxmimum iteration
        :param xtol: a threashold to check increment of transformation  for early stop
        :return: feature-metric projection error (r), updated transformation (g), encoder-decoder loss
        """
        training = self.encoder.training
        # training = self.decoder.training
        batch_size = p0.size(0)

        self.last_err = None
        g = g0
        self.g_series = torch.zeros(maxiter + 1, *g0.size(), dtype=g0.dtype)
        self.g_series[0] = g0.clone()

        # generate the features
        f0 = self.encoder(p0)
        f1 = self.encoder(p1)

        # task 1
        loss_enco_deco = 0.0
        if not is_test:
            decoder_out_f0 = self.decoder(f0)
            decoder_out_f1 = self.decoder(f1)

            p0_dist1, p0_dist2 = self.chamfer_loss(p0.contiguous(), decoder_out_f0)  # loss function
            loss_net0 = (torch.mean(p0_dist1)) + (torch.mean(p0_dist2))
            p1_dist1, p1_dist2 = self.chamfer_loss(p1.contiguous(), decoder_out_f1)  # loss function
            loss_net1 = (torch.mean(p1_dist1)) + (torch.mean(p1_dist2))
            loss_enco_deco = loss_net0 + loss_net1

        self.encoder.eval()  # and fix them.

        # task 2
        f0 = self.encoder(p0)  # [B, N, 3] -> [B, K]
        # approx. J by finite difference
        dt = self.dt.to(p0).expand(batch_size, 6)  # convert to the type of p0. [B, 6]
        J = self.approx_Jac(p0, f0, dt)
        # compute pinv(J) to solve J*x = -r
        try:
            Jt = J.transpose(1, 2)  # [B, 6, K]
            H = Jt.bmm(J)  # [B, 6, 6]
            # H = H + u_lamda * iDentity
            B = self.inverse(H)
            pinv = B.bmm(Jt)  # [B, 6, K]
        except RuntimeError as err:
            # singular...?
            self.last_err = err
            print(err)
            f1 = self.encoder(p1)  # [B, N, 3] -> [B, K]
            r = f1 - f0
            self.ptnet.train(training)
            return r, g, -1

        itr = 0
        r = None
        for itr in range(maxiter):
            p = self.transform(g.unsqueeze(1), p1)  # [B, 1, 4, 4] x [B, N, 3] -> [B, N, 3]
            f1 = self.encoder(p)  # [B, N, 3] -> [B, K]
            r = f1 - f0  # [B,K]
            dx = -pinv.bmm(r.unsqueeze(-1)).view(batch_size, 6)

            check = dx.norm(p=2, dim=1, keepdim=True).max()
            if float(check) < xtol:
                if itr == 0:
                    self.last_err = 0  # no update.
                break

            g = self.update(g, dx)
            self.g_series[itr + 1] = g.clone()
            self.prev_r = r

        self.encoder.train(training)
        return r, g, loss_enco_deco

    # estimate Jacobian matrix
    def approx_Jac(self, p0, f0, dt):
        # p0: [B, N, 3], Variable
        # f0: [B, K], corresponding feature vector
        # dt: [B, 6], Variable
        # Jk = (ptnet(p(-delta[k], p0)) - f0) / delta[k]
        batch_size = p0.size(0)
        num_points = p0.size(1)

        # compute transforms
        transf = torch.zeros(batch_size, 6, 4, 4).to(p0)
        for b in range(p0.size(0)):
            d = torch.diag(dt[b, :])  # [6, 6]
            D = self.exp(-d)  # [6, 4, 4]
            transf[b, :, :, :] = D[:, :, :]
        transf = transf.unsqueeze(2).contiguous()  # [B, 6, 1, 4, 4]
        p = self.transform(transf, p0.unsqueeze(1))  # x [B, 1, N, 3] -> [B, 6, N, 3]

        f0 = f0.unsqueeze(-1)  # [B, K, 1]
        f1 = self.encoder(p.view(-1, num_points, 3))
        f = f1.view(batch_size, 6, -1).transpose(1, 2)  # [B, K, 6]

        df = f0 - f  # [B, K, 6]
        J = df / dt.unsqueeze(1)  # [B, K, 6]

        return J

    # update the transformation
    def update(self, g, dx):
        # [B, 4, 4] x [B, 6] -> [B, 4, 4]
        dg = self.exp(dx)
        return dg.matmul(g)

    # calculate the chamfer loss
    def chamfer_loss(self, a, b):
        x, y = a, b
        bs, num_points, points_dim = x.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
        diag_ind = torch.arange(0, num_points)
        rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
        ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return torch.min(P, 1)[0], torch.min(P, 2)[0]

    @staticmethod
    def rsq(r):
        # |r| should be 0
        z = torch.zeros_like(r)
        return torch.nn.functional.mse_loss(r, z, reduction='sum')

    @staticmethod
    def comp(g, igt):
        """ |g*igt - I| (should be 0) """
        assert g.size(0) == igt.size(0)
        assert g.size(1) == igt.size(1) and g.size(1) == 4
        assert g.size(2) == igt.size(2) and g.size(2) == 4
        A = g.matmul(igt)
        I = torch.eye(4).to(A).view(1, 4, 4).expand(A.size(0), 4, 4)
        return torch.nn.functional.mse_loss(A, I, reduction='mean') * 16


# main algorithm class
class FMRTrain:
    # def __init__(self, dim_k, num_points, train_type):
    def __init__(self, args, params):
        self.dim_k = args.dim_k
        self.num_points = params['num_points']
        self.max_iter_train = args.max_iter_train
        self.max_iter_val = args.max_iter_val
        # self.max_iter = 10  # max iteration time for IC algorithm
        self._loss_type = args.train_type  # 0: unsupervised, 1: semi-supervised see. self.compute_loss()
        if self._loss_type != 0:
            assert self.max_iter_train > 0, self.max_iter_train

    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points
        decoder = Decoder(num_points=self.num_points)
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder,isTest=False)
        return fmr_solver

    def compute_loss(self, solver, data, device, train_mode=True):
        p0, p1, igt = data
        p0 = p0.to(device)  # template
        p1 = p1.to(device)  # source
        igt = igt.to(device)  # igt: p0 -> p1
        max_iter = self.max_iter_train if train_mode else self.max_iter_val
        r, loss_ende = solver.estimate_t(p0, p1, max_iter)

        loss_dict = dict(loss_ende=loss_ende)
        # unsupervised learning, set max_iter=0
        if max_iter == 0:
            return loss_ende, loss_dict     # autoencoder loss

        loss_r = solver.rsq(r)              # feature residual after registration
        est_g = solver.g
        loss_g = solver.comp(est_g, igt)    # pose residual

        loss_dict.update(loss_feat=loss_r, loss_pose=loss_g)

        # semi-supervised learning, set max_iter>0
        if self._loss_type == 0:
            loss = loss_ende
        elif self._loss_type == 1:
            loss = loss_ende + loss_g
        elif self._loss_type == 2:
            loss = loss_r + loss_g
        else:
            loss = loss_g
        return loss, loss_dict

    def train(self, model, trainloader, optimizer, device):
        model.train()

        Debug = True
        total_loss = 0
        total_loss_dict = dict()
        # count = 0
        if Debug:
            epe = 0
            count_mid = 10
        pbar = tqdm(trainloader)
        for i, data in enumerate(pbar):
            loss, loss_dict = self.compute_loss(model, data, device, True)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_item = loss.item()
            total_loss += loss_item
            for key in loss_dict:
                if key not in total_loss_dict:
                    total_loss_dict[key] = 0
                total_loss_dict[key] += loss_dict[key].item()

            if Debug:
                epe += loss_item
                if (i+1) % count_mid == 0:
                    # print('i=%d, fmr_loss=%f ' % (i, float(epe) / (count_mid + 1)))
                    pbar.set_description('train: %d, fmr_loss=%.4f' % (i+1, float(epe) / count_mid))
                    epe = 0.0
            # count += 1
        count = i + 1
        ave_loss = float(total_loss) / count
        for key in total_loss_dict:
            total_loss_dict[key] = float(total_loss_dict[key]) / count
        return ave_loss, total_loss_dict

    def validate(self, model, testloader, device):
        model.eval()
        Debug = True
        vloss = 0.0
        # count = 0
        total_loss_dict = dict()
        if Debug:
            epe = 0
            count_mid = 10
        with torch.no_grad():
            pbar = tqdm(testloader)
            for i, data in enumerate(pbar):
                loss_net, loss_dict = self.compute_loss(model, data, device, False)
                vloss += loss_net.item()
                for key in loss_dict:
                    if key not in total_loss_dict:
                        total_loss_dict[key] = 0
                    total_loss_dict[key] += loss_dict[key].item()

                if Debug:
                    epe += loss_net.item()
                    if (i+1) % count_mid == 0:
                        pbar.set_description('val: %d, fmr_loss=%.4f' % (i+1, float(epe) / count_mid))
                        epe = 0.0
                
                # count += 1
        count = i + 1
        ave_vloss = float(vloss) / count
        for key in total_loss_dict:
            total_loss_dict[key] = float(total_loss_dict[key]) / count
        return ave_vloss, total_loss_dict

class MetricTracker:
    def __init__(self, keys, angle_180=True) -> None:
        self.d = dict(num=0)
        for k in keys:
            self.d[k] = 0

        self.angle_180 = angle_180
        if self.angle_180:
            assert 'angle' in self.d, self.d
            self.d['num_90-'] = 0
            self.d['num_90+'] = 0
            self.d['angle180'] = 0
            
    def update(self, d):
        for k in d:
            self.d[k] += d[k]
        self.d['num']+= 1

        if self.angle_180:
            angle_diff_90m = d['angle']
            angle_diff_90p = 180-d['angle']
            if angle_diff_90m < angle_diff_90p:
                self.d['angle180'] += angle_diff_90m
                self.d['num_90-'] += 1
            else:
                self.d['angle180'] += angle_diff_90p
                self.d['num_90+'] += 1
        return
    def summary(self):
        for k in self.d:
            if k != 'num':
                self.d[k] /= self.d['num']
        return self.d.copy()

class FMRTest:
    def __init__(self, args, params):
        self.filename = args.outfile + ".csv"
        self.dim_k = args.dim_k
        self.num_points = params['num_points']
        self.max_iter = args.max_iter_val #10  # max iteration time for IC algorithm
        self._loss_type = 1  # see. self.compute_loss()

        # self.noise = args.noise
        # self.density = args.density
    def create_model(self):
        # Encoder network: extract feature for every point. Nx1024
        ptnet = PointNet(dim_k=self.dim_k)
        # Decoder network: decode the feature into points, not used during the evaluation
        decoder = Decoder(num_points=self.num_points)
        # feature-metric ergistration (fmr) algorithm: estimate the transformation T
        fmr_solver = SolveRegistration(ptnet, decoder, isTest=True)
        return fmr_solver

    def evaluate(self, solver, testloader, device):
        solver.eval()
        metric_tracker = MetricTracker(['angle', 'trans'])
        f_err = open(self.filename[:-4]+"_err.txt", "w")
        
        with open(self.filename, 'w') as fout:
            self.eval_1__header(fout)
            f_err.write("dw_res dv_res dw_0 dv_0\n")
            with torch.no_grad():
                # for i, data in enumerate(testloader):
                for i, data in enumerate(tqdm(testloader)):
                    if isinstance(data, dict):
                        p0 = data['inputs_est']
                        p1 = data['inputs_2']

                        scale = max(p0.max() - p0.min(), p1.max() - p1.min())
                        p0 = p0 / scale
                        p1 = p1 / scale
                        
                        igt_raw = data['T21']
                        igt_init_raw = data['T21_est']
                        igt_raw = torch.matmul(igt_raw, igt_init_raw.transpose(-1, -2))
                        
                        igt = torch.eye(4).unsqueeze(0)
                        igt[:, :3, :3] = igt_raw #.transpose(-1, -2)

                    elif isinstance(data, list):
                        p0, p1, igt = data  # igt: p0->p1
                    else:
                        raise ValueError("Not recognized data type {}".format(type(data)))
                    
                    # print("p0", p0.shape, p0.max(), p0.min())
                    # print("p1", p1.shape, p1.max(), p1.min())

                    # ### visualize
                    # print(p0.shape)
                    # visualize_pointcloud(p0.cpu().numpy()[0], points2=p1.cpu().numpy()[0], show=True, s1=1)
                    # category_name = testloader.dataset.models[data['idx'][0]]['category']
                    # if category_name == 'airplane':
                    #     model_name = testloader.dataset.models[data['idx'][0]]['model']
                    #     print(model_name)
                    #     print(p0.shape)
                    #     visualize_pointcloud(p0.cpu().numpy()[0], points2=p1.cpu().numpy()[0], show=True, s1=1)

                    ### ground truth
                    dx0 = se3.log(igt)  # --> [1, 6]
                    dw0 = dx0[:, :3].norm(p=2, dim=1)  # --> [1]
                    dv0 = dx0[:, 3:].norm(p=2, dim=1)  # --> [1]
                    dw0 = dw0.item() / np.pi * 180
                    dv0 = dv0.item()

                    ### estimation
                    p0 = p0.to(device)  # template (1, N, 3)
                    p1 = p1.to(device)  # source (1, M, 3)
                    solver.estimate_t(p0, p1, self.max_iter)
                    est_g = solver.g  # (1, 4, 4)   (p1->p0)

                    ig_gt = igt.cpu().contiguous().view(-1, 4, 4)  # --> [1, 4, 4]
                    g_hat = est_g.cpu().contiguous().view(-1, 4, 4)  # --> [1, 4, 4]
                    self.eval_1__write(fout, ig_gt, g_hat)

                    ### residual
                    dg = g_hat.bmm(ig_gt)  # if correct, dg == identity matrix.
                    dx = se3.log(dg)  # --> [1, 6] (if corerct, dx == zero vector)
                    dn = dx.norm(p=2, dim=1)  # --> [1]
                    dm = dn.mean()
                    dw = dx[:, :3].norm(p=2, dim=1)  # --> [1]
                    dv = dx[:, 3:].norm(p=2, dim=1)  # --> [1]
                    dw = dw.item() / np.pi * 180
                    dv = dv.item()

                    if (not np.isnan(dw)) and (not np.isnan(dv)):
                        metric_dict = dict(angle=dw, trans=dv)
                        metric_tracker.update(metric_dict)
                    f_err.write("{:.4f} {:.4f} {:.4f} {:.4f}\n".format(dw, dv, dw0, dv0)) 

        
            f_err.close()
            ### metric summary:
            avg_d = metric_tracker.summary()
            print("=================================================")
            for k in avg_d:
                print(k, avg_d[k])   
            with open(self.filename[:-4]+"_metric.txt", "w") as f:
                f.write("{:.4f} {:.4f} {:.4f} {:.4f} {:.4f}".format(avg_d['angle'], avg_d['trans'], avg_d['angle_180'], avg_d['num_90-'], avg_d['num_90+']))



    def eval_1__header(self, fout):
        cols = ['h_w1', 'h_w2', 'h_w3', 'h_v1', 'h_v2', 'h_v3', \
                'g_w1', 'g_w2', 'g_w3', 'g_v1', 'g_v2', 'g_v3']  # h: estimated, g: ground-truth twist vectors
        print(','.join(map(str, cols)), file=fout)
        fout.flush()

    def eval_1__write(self, fout, ig_gt, g_hat):
        x_hat = se3.log(g_hat)  # --> [-1, 6]   estimation   p1->p0
        mx_gt = se3.log(ig_gt)  # --> [-1, 6]   ground truth p0->p1
        for i in range(x_hat.size(0)):
            x_hat1 = x_hat[i]  # [6]
            mx_gt1 = mx_gt[i]  # [6]
            vals = torch.cat((x_hat1, -mx_gt1))  # [12]
            valn = vals.cpu().numpy().tolist()
            print(','.join(map(str, valn)), file=fout)
        fout.flush()

