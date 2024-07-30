from __future__ import print_function

import math
import os
import time
import argparse
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torch.backends.cudnn as cudnn
import torch.optim as optim
from future.backports import OrderedDict
import numpy as np
from torch import nn

from models import wideresnet, resnet, densenet, resnet_64x64, resnext_64x64
from utils import nn_util, eval_ood_util

from utils.downsampled_imagenet800_loader import ImageNet800
from utils.downsampled_imagenet_loader import ImageNetEXCIFAR
from utils.tinyimages_80mn_loader import TinyImages, RandomImages

parser = argparse.ArgumentParser(description='Source code of DaYu')
parser.add_argument('--model_name', default='wrn-40-4', help='model name, wrn-40-4 or resnet-34')
parser.add_argument('--dataset', default='cifar10', help='dataset: svhn, cifar10 or cifar100')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--weight_decay', '--wd', default=5e-4, type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train')
parser.add_argument('--schedule', type=int, nargs='+', default=[75, 90],
                    help='decrease learning rate at these epochs.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model_dir', default='dnn_models/cifar/', help='directory of model for saving checkpoint')
parser.add_argument('--topk_cpts', '-s', default=50, type=int, metavar='N', help='save top-k robust checkpoints')
parser.add_argument('--gpuid', type=int, default=0, help='the ID of GPU.')
parser.add_argument('--resume_epoch', type=int, default=0, metavar='N', help='epoch for resuming training')

parser.add_argument('--storage_device', default='cuda', help='device for computing auroc and fpr: cuda or cpu')
parser.add_argument('--save_socre_dir', default='', type=str, help='dir for saving scores')
parser.add_argument('--training_method', default='clean', help='training method: clean')

parser.add_argument('--ood_batch_size', default=256, type=int, help='mini-batch size (default: 256)')
parser.add_argument('--ood_training_method', default='clean', help='out training method: clean')
parser.add_argument('--mine_ood', action='store_true', default=False, help='whether to mine informative oods')
parser.add_argument('--quantile', default=0.125, type=float, help='quantile')
parser.add_argument('--ood_warmup', default=-1, type=int, help='warmup epoch for training on out data.')
parser.add_argument('--ood_file', default='../datasets/80M_Tiny_Images/tiny_images.bin',
                    help='tiny_images file ptah')
parser.add_argument('--ood_beta', default=1.0, type=float, help='beta for ood_loss')
parser.add_argument('--auxiliary_dataset', default='80m_tiny_images',
                    choices=['80m_tiny_images', '300k-random-images', 'downsampled-imagenet',
                             'downsampled-imagenet-800', 'none'], type=str,
                    help='which auxiliary dataset to use')
parser.add_argument('--gpus', type=int, nargs='+', default=[], help='gpus.')

args = parser.parse_args()

use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
if use_cuda:
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)
# torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
# torch.backends.cudnn.deterministic = True
torch.cuda.set_device(int(args.gpuid))
if len(args.gpus) > 1:
    torch.cuda.set_device(int(args.gpus[0]))
device = torch.device('cuda' if torch.cuda.is_available() and use_cuda else 'cpu')
storage_device = torch.device(args.storage_device)

if args.dataset == 'cifar10':
    NUM_IN_CLASSES = 10
    # NUM_EXAMPLES = 50000
elif args.dataset == 'cifar100':
    NUM_IN_CLASSES = 100
    # NUM_EXAMPLES = 50000
elif args.dataset == 'svhn':
    NUM_IN_CLASSES = 10
    # NUM_EXAMPLES = 73257
elif args.dataset == 'tiny-imagenet-200':
    NUM_IN_CLASSES = 200
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

GPUID = args.gpuid
os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)

if args.save_socre_dir == '':
    args.save_socre_dir = args.model_dir
    print(f"INFO, save_socre_dir is not given, I have set it to {args.model_dir}")

# settings
if not os.path.exists(args.model_dir):
    os.makedirs(args.model_dir)

def save_cpt(cur_cpt, model_dir, training_method, epoch):
    path = os.path.join(model_dir, '{0}_model_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['model'].state_dict(), path)
    path = os.path.join(model_dir, '{0}_cpt_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['optimizer'].state_dict(), path)
    path = os.path.join(model_dir, '{0}_trd_epoch{1}.pt'.format(training_method, epoch))
    torch.save(cur_cpt['record'], path)


def del_cpt(model_dir, training_method, epoch):
    path = os.path.join(model_dir, '{0}_model_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(model_dir, '{0}_cpt_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)
    path = os.path.join(model_dir, '{0}_trd_epoch{1}.pt'.format(training_method, epoch))
    if os.path.exists(path):
        os.remove(path)


def resume(epoch, model, optimizer, training_record, print_cpt_info=True):
    path = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
    model.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_cpt_epoch{1}.pt'.format(args.training_method, epoch))
    optimizer.load_state_dict(torch.load(path))
    path = os.path.join(args.model_dir, '{0}_trd_epoch{1}.pt'.format(args.training_method, epoch))
    if os.path.exists(path):
        training_record = torch.load(path)
        print('successfully loaded training_record from {}'.format(path))
        if print_cpt_info:
            print('loaded training_record: {}'.format(training_record))
    else:
        print('I find no training_record from', path)
    return model, optimizer, training_record


def get_all_data(test_loader):
    x_test = None
    y_test = None
    for i, data in enumerate(test_loader):
        batch_x, in_y = data
        if x_test is None:
            x_test = batch_x
        else:
            x_test = torch.cat((x_test, batch_x), 0)

        if y_test is None:
            y_test = in_y
        else:
            y_test = torch.cat((y_test, in_y), 0)
    return x_test, y_test


class OODDataset(torch.utils.data.Dataset):
    def __init__(self, images, labels):
        self.labels = labels
        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        # Load data and get label
        X = self.images[index]
        y = self.labels[index]

        return X, y


def select_ood(ood_loader, model, batch_size, num_in_classes, num_pool_iters, ood_dataset_size, quantile):
    # start at a random point of the outlier dataset; this induces more randomness without obliterating locality
    offset = np.random.randint(len(ood_loader.dataset))
    while offset >= 0 and offset < 10000:
        offset = np.random.randint(len(ood_loader.dataset))

    ood_loader.dataset.offset = offset

    out_iter = iter(ood_loader)
    print('Start selecting OOD samples...')

    start = time.time()
    # select ood samples
    model.eval()

    all_ood_input = []
    all_ood_conf = []
    for k in range(num_pool_iters):

        try:
            out_set = next(out_iter)
        except StopIteration:
            offset = np.random.randint(len(ood_loader.dataset))
            while offset >= 0 and offset < 10000:
                offset = np.random.randint(len(ood_loader.dataset))
            ood_loader.dataset.offset = offset
            out_iter = iter(ood_loader)
            out_set = next(out_iter)

        input = out_set[0]
        with torch.no_grad():
            output = model(input.to(device))
        output = F.softmax(output, dim=1)
        conf = torch.max(output[:, :num_in_classes], dim=1)[0]
        conf = conf.detach().cpu().numpy()

        all_ood_input.append(input)
        all_ood_conf.extend(conf)

    all_ood_input = torch.cat(all_ood_input, 0)[:ood_dataset_size * 4]
    all_ood_conf = np.array(all_ood_conf)[:ood_dataset_size * 4]
    indices = np.argsort(-all_ood_conf)  # large -> small

    if len(all_ood_conf) < ood_dataset_size * 4:
        print('Warning, the num_pool_iters is too small: batch * num_pool_iters should >= ood_dataset_size * 4')

    N = all_ood_input.shape[0]
    selected_indices = indices[int(quantile * N):int(quantile * N) + ood_dataset_size]

    print('Total OOD samples: ', len(all_ood_conf))
    print('Max in-conf: ', np.max(all_ood_conf), 'Min in-Conf: ', np.min(all_ood_conf), 'Average in-conf: ',
          np.mean(all_ood_conf))

    selected_ood_conf = all_ood_conf[selected_indices]
    print('Selected OOD samples: ', len(selected_ood_conf))
    print('Selected Max in-conf: ', np.max(selected_ood_conf), 'Selected Min conf: ', np.min(selected_ood_conf),
          'Selected Average in-conf: ', np.mean(selected_ood_conf))

    ood_images = all_ood_input[selected_indices]
    # ood_labels = (torch.ones(ood_dataset_size) * output.size(1)).long()
    ood_labels = torch.zeros((ood_images.size(0),)).long().to(ood_images.device)

    ood_train_loader = torch.utils.data.DataLoader(
        OODDataset(ood_images, ood_labels),
        batch_size=batch_size, shuffle=True)
    print('Time: ', time.time() - start)

    return ood_train_loader


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    epoch_lr = args.lr
    for i in range(0, len(args.schedule)):
        if epoch > args.schedule[i]:
            epoch_lr = args.lr * np.power(args.gamma, (i + 1))
    for param_group in optimizer.param_groups:
        param_group['lr'] = epoch_lr
    return epoch_lr


def add_noise_to_uniform(unif):
    assert len(unif.size()) == 2
    unif_elem = unif.float().mean()
    new_unif = unif.clone()
    new_unif.uniform_(unif_elem - 0.005 * unif_elem, unif_elem + 0.005 * unif_elem)
    factor = new_unif.sum(dim=1) / unif.sum(dim=1)
    new_unif = new_unif / factor.unsqueeze(dim=1)
    # sum=new_unif.sum(dim=1)
    return new_unif


def cal_cls_results(logits, y, in_classes, num_v_classes, data_type='in'):
    msps, preds = torch.max(F.softmax(logits, dim=1)[:, :in_classes], dim=1)
    if data_type == 'in':
        corr_indcs = preds == y
        corr = corr_indcs.sum().item()
        corr_probs = msps[corr_indcs]
        global_preds = torch.max(F.softmax(logits, dim=1), dim=1)[1]
        located_v_indcs = torch.logical_and(global_preds >= in_classes, global_preds < in_classes + num_v_classes)
        located_v_cls = located_v_indcs.sum().item()
        located_v_corr_indcs = torch.logical_and(corr_indcs, located_v_indcs)
        located_v_corr_cls = located_v_corr_indcs.sum().item()
        return corr, located_v_cls, located_v_corr_cls, msps, corr_probs
    elif data_type == 'out':
        global_preds = torch.max(F.softmax(logits, dim=1), dim=1)[1]
        located_v_cls = torch.logical_and(global_preds >= in_classes, global_preds < in_classes + num_v_classes).sum().item()
        return located_v_cls, msps
    else:
        raise ValueError('un-supported data_type: {}'.format(data_type))


def kl_loss(nat_logits, adv_logits):
    batch_size = nat_logits.size()[0]
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    kl_loss = (1.0 / batch_size) * criterion_kl(F.log_softmax(adv_logits, dim=1), F.softmax(nat_logits, dim=1))
    return kl_loss


def train(model, train_loader, optimizer, test_loader, train_ood_loader):
    def get_in_y_soft(in_y):
        return F.one_hot(in_y, num_classes=NUM_IN_CLASSES).to(in_y.device)

    def get_out_y_soft(num_x):
        return torch.zeros([num_x, NUM_IN_CLASSES]) + 1. / NUM_IN_CLASSES

    def re_process_in_x(org_id_x, org_id_y):
        nat_id_y_soft = get_in_y_soft(org_id_y).to(org_id_y.device)
        return org_id_x, nat_id_y_soft, len(org_id_x)

    def re_process_out_x(org_ood_x):
        len_org_ood = len(org_ood_x)
        nat_ood_y_soft = get_out_y_soft(len_org_ood).to(org_ood_x.device)
        return org_ood_x, nat_ood_y_soft, len_org_ood

    training_record = OrderedDict()
    if args.resume_epoch > 0:
        print('try to resume from epoch', args.resume_epoch)
        model, optimizer, training_record = resume(args.resume_epoch, model, optimizer, training_record)

    for epoch in range(args.resume_epoch + 1, args.epochs + 1):
        print('===================================================================================================')
        if train_ood_loader is not None and epoch > args.ood_warmup:
            if args.mine_ood:
                attack_args = {'attack_eps': args.ood_attack_eps}
                num_ood_candidates = len(train_loader.dataset) * math.ceil(args.ood_batch_size / args.batch_size)
                # 2000 * ood_batch_size >= num_ood_candidates * 4
                selected_ood_loader = select_ood(train_ood_loader, model, args.ood_batch_size, NUM_IN_CLASSES,
                                                 num_pool_iters=2000, ood_dataset_size=num_ood_candidates,
                                                 quantile=args.quantile)
                train_ood_iter = enumerate(selected_ood_loader)
            else:
                train_ood_iter = enumerate(train_ood_loader)

        start_time = time.time()
        epoch_lr = adjust_learning_rate(optimizer, epoch)

        num_nat_ids = 0
        num_adv_ids = 0
        train_nat_id_corr = 0
        train_adv_id_corr = 0
        train_nat_id_msp = torch.tensor([])
        train_adv_id_msp = torch.tensor([])
        
        train_nat_ood_msp = torch.tensor([])
        train_adv_ood_msp = torch.tensor([])
        # train_aa_ood_msp = torch.tensor([])
        for i, data in enumerate(train_loader):
            org_id_x, org_id_y = data

            org_id_x = org_id_x.cuda(non_blocking=True)
            org_id_y = org_id_y.cuda(non_blocking=True)
            processed_id_x, processed_id_y_soft, len_processed_nat_id = re_process_in_x( org_id_x, org_id_y)
            cat_id_x = processed_id_x
            len_cat_id = len(cat_id_x)
            
            if train_ood_loader is not None and epoch > args.ood_warmup:
                if args.otwo_stage and epoch % 2 == 1:
                    cat_x = cat_id_x
                else:
                    j, (org_ood_x, _) = next(train_ood_iter)
                    org_ood_x = org_ood_x.to(device)
                    processed_ood_x, processed_ood_y_soft, len_processed_nat_out = re_process_out_x(org_ood_x)
                    cat_ood_x = processed_ood_x
                    cat_x = torch.cat((cat_id_x, cat_ood_x), dim=0)
            else:
                cat_x = cat_id_x
                
            model.train()
            cat_logits = model(cat_x)
            nat_id_loss = None
            id_loss = nn_util.cross_entropy_soft_target(cat_logits[:len_cat_id], processed_id_y_soft)

            nat_ood_loss, ood_loss = None, None
            if epoch > args.ood_warmup:
                if args.otwo_stage and epoch % 2 == 1:
                    loss = id_loss
                else:
                    ood_loss = nn_util.cross_entropy_soft_target(cat_logits[len_cat_id:], processed_ood_y_soft)
                    loss = id_loss + args.ood_beta * ood_loss
            else:
                loss = id_loss

            # compute output
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss = loss.item()

            # statistic training results on IDs
            id_loss = round(id_loss.item(), 6)
            model.eval()
            with torch.no_grad():
                nat_logits = model(org_id_x)
            nat_corr, _, _, nat_id_msp, _ = cal_cls_results(nat_logits, org_id_y, NUM_IN_CLASSES, 0, data_type='in')
            train_nat_id_corr += nat_corr
            train_nat_id_msp = torch.cat((train_nat_id_msp, nat_id_msp.cpu()), dim=0)

            # statistic training results on OODs
            if ood_loss is not None:
                with torch.no_grad():
                    nat_out_logits = model(org_ood_x)
                # if len_processed_nat_out > 0:
                _, nat_oo_msp = cal_cls_results(nat_out_logits, None, NUM_IN_CLASSES, 0, data_type='out')
                train_nat_ood_msp = torch.cat((train_nat_ood_msp, nat_oo_msp.cpu()), dim=0)

            num_nat_ids += len(org_id_x)
            if i % args.log_interval == 0 or i >= len(train_loader) - 1:
                processed_ratio = round((i / len(train_loader)) * 100, 2)
                print('Train Epoch: {}, Training progress: {}% [{}/{}], In loss: {}, In nat loss: {},'
                      'Out loss: {}, Out nat loss: {}'
                      .format(epoch, processed_ratio, i, len(train_loader), id_loss, nat_id_loss, ood_loss,
                              nat_ood_loss))

        train_nat_id_acc = (float(train_nat_id_corr) / num_nat_ids)
        train_adv_id_acc = None
        if num_adv_ids > 0:
            train_adv_id_acc = (float(train_adv_id_corr) / num_adv_ids)
        batch_time = time.time() - start_time

        message = 'Epoch {}, Time {}, LR: {}, ID loss: {}, OOD loss:{}'.format(epoch, batch_time, epoch_lr, id_loss, ood_loss)
        print(message)
        in_message = 'Training on ID: nat acc: {}, mean of nat-msp: {}, adv acc: {}, mean of adv-msp: {}' \
            .format(train_nat_id_acc, train_nat_id_msp.mean().item(), train_adv_id_acc, train_adv_id_msp.mean().item())
        print(in_message)
        out_message = 'Training on OOD: mean of nat-msp: {}, mean of adv-msp: {}'\
            .format(train_nat_ood_msp.mean().item(), train_adv_ood_msp.mean().item())
        print(out_message)
        print('----------------------------------------------------------------')

        # Evaluation
        socre_save_dir = os.path.join(args.model_dir, 'epoch_{}_scores'.format(epoch))
        if not os.path.exists(socre_save_dir):
            os.makedirs(socre_save_dir)
        all_in_full_scores_file = os.path.join(socre_save_dir, 'id_misc_scores.txt')
        # clean acc
        test_nat_id_acc, _, _, test_nat_id_msp, test_nat_id_corr_prob, test_nat_id_v_msp, test_nat_id_v_ssp = \
            nn_util.eval(model, test_loader, NUM_IN_CLASSES, 0,
                         misc_score_file=all_in_full_scores_file)
        test_nat_id_mmsp = test_nat_id_msp.mean().item()
        test_nat_id_corr_mprob = test_nat_id_corr_prob.mean().item()
        print('Testing ID nat acc: {}, mean of nat-msp: {}, mean of nat corr-prob: {}'
              .format(test_nat_id_acc, test_nat_id_mmsp, test_nat_id_corr_mprob))

        adv_id_auroc = 0.
        adv_id_fpr95 = 1.
        if epoch > args.schedule[0] - 1:
            print('----------------------------------------')
            print('training performance on OODs:')
            # for key_odd, ood_msp in {'train_nat_ood_msp': train_nat_ood_msp, 'train_adv_ood_msp': train_adv_ood_msp,
            #                          'train_aa_ood_msp': train_aa_ood_msp}.items():
            for key_odd, ood_msp in {'train_nat_ood_msp': train_nat_ood_msp,
                                     'train_adv_ood_msp': train_adv_ood_msp}.items():
                if len(ood_msp) == 0:
                    continue
                st = time.time()
                aa_ood_auroc = nn_util.auroc(test_nat_id_msp, ood_msp)
                aa_ood_fpr95, _ = nn_util.fpr_at_tprN(test_nat_id_msp, ood_msp, TPR=95)
                aa_ood_tpr95, _ = nn_util.tpr_at_tnrN(test_nat_id_msp, ood_msp, TNR=95)
                et = time.time()
                print('num of {}: {}, mean of msp: {}, AUROC:{}, FPR@TPR95: {}, TPR@TNR95: {}, evaluation time: {}s'
                      .format(key_odd, ood_msp.size(0), ood_msp.mean().item(), aa_ood_auroc, aa_ood_fpr95, aa_ood_tpr95,
                              et - st))

            print('----------------------------------------')
            print('performance on val OOD data:')
            img_size=[32, 32]
            if args.dataset == 'cifar10' or args.dataset == 'cifar100':
                out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
            elif args.dataset == 'svhn':
                out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
            elif args.dataset == 'tiny-imagenet-200':
                img_size = [64, 64]
                out_datasets = ['places365', 'dtd']
            elif args.dataset == 'tiny-imagenet-200-32x32':
                out_datasets = ['places365', 'dtd']
            id_score_dict = {'in_msp': test_nat_id_msp}
            ood_attack_methods = ['clean', ]
            for o_a_m in ood_attack_methods:
                for step in [100]:
                    st = time.time()
                    _, inputs_num, _, ood_in_msp_dict, _, _, ood_v_ssp_dict \
                        = eval_ood_util.get_ood_scores(model, NUM_IN_CLASSES, 0, socre_save_dir,
                                                       out_datasets, ood_attack_method=o_a_m, ood_batch_size=128,
                                                       device=device, num_oods_per_set=128 * 2, img_size=img_size)
                    ood_vssp_inmsp_dict = {}
                    for temp_key, _ in ood_v_ssp_dict.items():
                        ood_vssp_inmsp_dict[temp_key] = ood_v_ssp_dict[temp_key] - ood_in_msp_dict[temp_key]

                    for sc_key, ood_values in {'in_msp': [ood_in_msp_dict, 'in_msp'],
                                               }.items():
                        if sc_key not in id_score_dict:
                            continue
                        con_f = id_score_dict[sc_key]
                        print('----> using {} as scoring function ---->'.format(sc_key))
                        conf_t = ood_values[0]
                        scoring_func = ood_values[1]
                        (_, _, _, _), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                            = eval_ood_util.eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=[80, 85, 90, 95],
                                                                       scoring_func=scoring_func)
                        print('{}-step {} attacked OODs: with {} scoring function, indiv_auc:{}, indiv_fprN:{}, '
                              'indiv_tprN:{}, indiv_mean_score:{}'.format(step, o_a_m, sc_key, indiv_auc, indiv_fprN,
                                                                          indiv_tprN, indiv_mean_score))
                        print("{}-step {} attacked OODs: mixing_auc: {}, mixing_fprN: {}, mixing_tprN: {}, "
                              "mixing_score: {}, eval time: {}s".format(step, o_a_m, mixing_auc, mixing_fprN,
                                                                        mixing_tprN, mixing_score, time.time() - st))
        # maintain records
        training_record[epoch] = {'loss': loss, 'train_nat_id_acc': train_nat_id_acc,
                                  'train_adv_id_acc': train_adv_id_acc,
                                  'test_nat_id_acc': test_nat_id_acc,
                                  }
        cur_cpt = {'model': model, 'optimizer': optimizer, 'record': training_record}
        if args.epochs - epoch <= args.topk_cpts:
            save_cpt(cur_cpt, args.model_dir, args.training_method, epoch)
    return training_record


def get_model(model_name, num_in_classes=10, num_out_classes=0, num_v_classes=0, normalizer=None, dataset='cifar10'):
    size_3x32x32 = ['svhn', 'cifar10', 'cifar100', 'tiny-imagenet-200-32x32']
    size_3x64x64 = ['tiny-imagenet-200']
    size_3x224x224 = ['imagenet']
    if dataset in size_3x32x32:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-2':
            return wideresnet.WideResNet(depth=40, widen_factor=2, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-18':
            return resnet.ResNet18(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet.ResNet34(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet.ResNet50(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'densenet':
            return densenet.DenseNet3(100, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)
    elif dataset in size_3x64x64:
        if model_name == 'wrn-34-10':
            return wideresnet.WideResNet(depth=34, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-28-10':
            return wideresnet.WideResNet(depth=28, widen_factor=10, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-4':
            return wideresnet.WideResNet(depth=40, widen_factor=4, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'wrn-40-2':
            return wideresnet.WideResNet(depth=40, widen_factor=2, normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-18':
            return resnet_64x64.resnet18(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-34':
            return resnet_64x64.resnet34(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnet-50':
            return resnet_64x64.resnet50(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        elif model_name == 'resnext-50':
            return resnext_64x64.resnext50_32x4d(normalizer=normalizer, num_in_classes=num_in_classes, num_out_classes=num_out_classes, num_v_classes=num_v_classes)
        else:
            raise ValueError('un-supported model: {0}', model_name)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def main():
    # setup data loader
    normalizer = None
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    transform_test = T.Compose([T.ToTensor()])
    aux_id_transform_train = T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor()
    ])
    if args.dataset == 'cifar10':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR10
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        # normalizer = T.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
        #                          std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        dataloader = torchvision.datasets.CIFAR100
        train_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        # normalizer = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        transform_train = T.Compose([
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        svhn_train = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_train,
                                               split='train')
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        train_loader = torch.utils.data.DataLoader(dataset=svhn_train, batch_size=args.batch_size, shuffle=True,
                                                   **kwargs)
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    elif args.dataset == 'tiny-imagenet-200-32x32':
        data_dir = '../datasets/tiny-imagenet-200/'
        # normalizer = T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        transform_train = T.Compose([
            T.RandomResizedCrop(32),
            T.Pad(4, padding_mode='reflect'),
            T.RandomCrop(32),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'tiny-imagenet-200':
        kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        data_dir = '../datasets/tiny-imagenet-200/'
        # normalizer = T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        transform_train = T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
        ])
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        trainset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    ood_mining_batch_size = args.ood_batch_size
    if args.mine_ood:
        ood_mining_batch_size = max(args.ood_batch_size, 400)
    num_fixed_ood = args.num_fixed_ood
    if args.fix_ood and args.num_fixed_ood == -1:
        num_fixed_ood = int(len(train_loader) * (float(ood_mining_batch_size) / float(args.batch_size)) * args.batch_size)
        print('num_fixed_ood:', num_fixed_ood)

    out_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    if args.auxiliary_dataset == '80m_tiny_images':
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        ood_train_loader = torch.utils.data.DataLoader(
            TinyImages(transform=ood_transform_train, num_fixed_x=num_fixed_ood, tiny_file=args.ood_file),
            batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == '300k-random-images':
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        ood_train_loader = torch.utils.data.DataLoader(
            RandomImages(transform=ood_transform_train, tiny_file=args.ood_file), batch_size=ood_mining_batch_size,
            shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == 'downsampled-imagenet':
        ood_mining_batch_size = args.ood_batch_size
        ood_transform_train = T.Compose([
            T.ToTensor(),
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])
        img_loader = ImageNetEXCIFAR(transform=ood_transform_train, id_type=args.dataset, exclude_cifar=True,
                                     excl_simi=0.35, img_size=32, imagenet_dir=args.ood_file)
        if args.mine_ood:
            ood_mining_batch_size = max(args.ood_batch_size, 400)
        print('Info, ood_batch_size is (re-)set to {}'.format(ood_mining_batch_size))
        ood_train_loader = torch.utils.data.DataLoader(img_loader, batch_size=ood_mining_batch_size, shuffle=False, **out_kwargs)
    elif args.auxiliary_dataset == 'downsampled-imagenet-800':
        ood_mining_batch_size = args.ood_batch_size
        out_kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
        if '32x32' in args.dataset:
            ood_transform_train = T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            img_size = 32
        else:
            ood_transform_train = T.Compose([
                T.ToTensor(),
                T.ToPILImage(),
                T.RandomCrop(64, padding=8),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
            ])
            img_size = 64
        img_loader = ImageNet800(transform=ood_transform_train, imagenet_dir=args.ood_file, img_size=img_size)
        if args.mine_ood:
            ood_mining_batch_size = max(args.ood_batch_size, 400)
        print('Info, ood_batch_size is automatically set to {}'.format(ood_mining_batch_size))
        ood_train_loader = torch.utils.data.DataLoader(img_loader, batch_size=ood_mining_batch_size, shuffle=False,
                                                       **out_kwargs)
    else:
        ood_train_loader = None
    cudnn.benchmark = True

    # init model, Net() can be also used here for training
    model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                      num_v_classes=0, normalizer=normalizer, dataset=args.dataset)
    if len(args.gpus) > 1:
        model = nn.DataParallel(model.to(device), device_ids=args.gpus, output_device=args.gpus[0])
    else:
        model = model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    print('========================================================================================================')
    print('args:', args)
    print('========================================================================================================')

    train(model, train_loader, optimizer, test_loader, ood_train_loader)


if __name__ == '__main__':
    main()
