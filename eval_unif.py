from __future__ import print_function
import os
import torch
import argparse
import torchvision.transforms as T
from utils import nn_util, eval_ood_util
from models import wideresnet, resnet, densenet, resnet_64x64, resnext_64x64
import torchvision
import time

parser = argparse.ArgumentParser(description='PyTorch CIFAR OOD Detection Evaluation')
parser.add_argument('--model_name', default='wrn-40-4',
                    help='model name, wrn-28-10, wrn-34-10, wrn-40-4, resnet-18, resnet-50')
parser.add_argument('--alpha', default=0., type=float, help='total confidence of virtual dayu classes')
parser.add_argument('--dataset', default='cifar10', help='dataset: svhn, cifar10 or cifar100')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--no_cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpuid', type=int, default=0, help='The ID of GPU.')
parser.add_argument('--model_file', default='./dnn_models/dataset/model.pt', help='file path of src model')

parser.add_argument('--save_socre_dir', default='', help='dir for saving scores')
parser.add_argument('--ood_batch_size', default=128, type=int, help='mini-batch size (default: 128)')
parser.add_argument('--storage_device', default='cpu', help='device for computing auroc and fpr: cuda or cpu')

parser.add_argument('--model_dir', default='', help='directory of model for saving checkpoint')
parser.add_argument('--training_method', default='clean', help='training method: clean')
parser.add_argument('--st_epoch', default=150, type=int, help='start epoch')
parser.add_argument('--end_epoch', default=201, type=int, help='end epoch')
parser.add_argument('--gpus', type=int, nargs='+', default=[], help='gpus.')

args = parser.parse_args()

# settings
use_cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
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
elif args.dataset == 'svhn':
    NUM_IN_CLASSES = 10
elif args.dataset == 'cifar100':
    NUM_IN_CLASSES = 100
elif args.dataset == 'tiny-imagenet-200':
    NUM_IN_CLASSES = 200
else:
    raise ValueError('error dataset: {0}'.format(args.dataset))

if args.attack_lr > 1:
    args.attack_lr = args.attack_lr / 255
if args.attack_eps > 1:
    args.attack_eps = args.attack_eps / 255
if args.targets < 0:
    args.targets = NUM_IN_CLASSES
    print('INFO, args.targets is not given, I set it to {}'.format(NUM_IN_CLASSES - 1))

if args.save_socre_dir == '' and args.model_file != '':
    args.save_socre_dir = args.model_file + '.eval-scores'
    if not os.path.exists(args.save_socre_dir):
        os.makedirs(args.save_socre_dir)
    print('Warning, args.save_socre_dir is not given, I set it to: {}'.format(args.save_socre_dir))

def filter_state_dict(state_dict):
    from collections import OrderedDict

    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'sub_block' in k:
            continue
        if 'module' in k:
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


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


def eval_main():
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    transform_test = T.Compose([T.ToTensor()])
    if args.dataset == 'cifar10':
        dataloader = torchvision.datasets.CIFAR10
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar10', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'cifar100':
        dataloader = torchvision.datasets.CIFAR100
        test_loader = torch.utils.data.DataLoader(
            dataloader('../datasets/cifar100', train=False, download=True, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False, **kwargs)
    elif args.dataset == 'svhn':
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        svhn_test = torchvision.datasets.SVHN(root='../datasets/svhn/', download=True, transform=transform_test,
                                              split='test')
        test_loader = torch.utils.data.DataLoader(dataset=svhn_test, batch_size=args.batch_size, shuffle=False,
                                                  **kwargs)
    elif args.dataset == 'tiny-imagenet-200-32x32':
        data_dir = '../datasets/tiny-imagenet-200/'
        normalizer = T.Normalize(mean=[0.4802, 0.4481, 0.3975], std=[0.2770, 0.2691, 0.2821])
        transform_test = T.Compose([
            T.Resize(32),
            T.ToTensor(),
        ])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.dataset == 'tiny-imagenet-200':
        data_dir = '../datasets/tiny-imagenet-200/'
        transform_test = T.Compose([
            T.ToTensor(),
        ])
        testset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        raise ValueError('un-supported dataset: {0}'.format(args.dataset))

    print('args:', args)
    print('================================================================')

    model = get_model(args.model_name, num_in_classes=NUM_IN_CLASSES, num_out_classes=0,
                      num_v_classes=args.num_v_classes, dataset=args.dataset).to(device)
    if args.model_dir == '':
        checkpoint = torch.load(args.model_file)
        # model.load_state_dict(checkpoint['state_dict'])
        model.load_state_dict(checkpoint)
        model = model.to(device)
        print('Successfully loaded model from:{}'.format(args.model_file))

        eval_ood_util.eval_in_and_out(model, test_loader, NUM_IN_CLASSES, args, device=device,
                                      storage_device=storage_device)
    else:
        pick_cpts(model, test_loader, args)
        # pick_cpt_and_detecting_adv_ids(model, test_loader, args)


def pick_cpts(model, test_loader, args):
    st_epoch = args.st_epoch
    end_epoch = args.end_epoch
    for epoch in range(st_epoch, end_epoch):
        model_file = os.path.join(args.model_dir, '{0}_model_epoch{1}.pt'.format(args.training_method, epoch))
        if not os.path.exists(model_file):
            print('cpt file is not found: {}'.format(model_file))
            continue
        else:
            print('===============================================================================================')
            print('eval cpt epoch {} from {}'.format(epoch, model_file))
        model.load_state_dict(torch.load(model_file))
        model = model.to(device)
        socre_save_dir = os.path.join(args.model_dir, 'epoch_{}_scores'.format(epoch))
        if not os.path.exists(socre_save_dir):
            os.makedirs(socre_save_dir)
        all_id_misc_scores_file = os.path.join(socre_save_dir, 'id_misc_scores.txt')
        all_id_logits_file = os.path.join(socre_save_dir, 'id_logits.npy')

        # eval clean acc
        nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr, nat_id_msp, nat_id_corr_prob, nat_id_v_msp, nat_id_v_ssp = \
            nn_util.eval(model, test_loader, NUM_IN_CLASSES, num_v_classes=args.num_v_classes,
                         misc_score_file=all_id_misc_scores_file, lotis_file=all_id_logits_file)
        nat_id_mmsp = nat_id_msp.mean().item()
        nat_id_corr_mprob = nat_id_corr_prob.mean().item()
        nat_id_vssp_minus_inmsp = nat_id_v_ssp - nat_id_msp
        nat_id_m_vssp_minus_inmsp = nat_id_vssp_minus_inmsp.mean().item()
        nat_id_mvssp = nat_id_v_ssp.mean().item()
        print('Testing nat ID acc: {}, nat miscls v-classes: {}, nat micls v-classes corr: {}, mean of in-msp: {}, '
              'mean of corr-prob: {}, mean of nat v_ssp: {}, mean of nat (v_ssp - in-msp): {}'
              .format(nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr, nat_id_mmsp, nat_id_corr_mprob, nat_id_mvssp,
                      nat_id_m_vssp_minus_inmsp))
        print('------------------------------------------------------------------------------------------------------')
        print('performance on val OOD data:')
        img_size = [32, 32]
        if args.dataset == 'cifar10' or args.dataset == 'cifar100':
            out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
        elif args.dataset == 'svhn':
            out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
        elif args.dataset == 'tiny-imagenet-200':
            img_size = [64, 64]
            out_datasets = ['places365', 'dtd']
        elif args.dataset == 'tiny-imagenet-200-32x32':
            out_datasets = ['places365', 'dtd']

        id_score_dict = {'in_msp': nat_id_msp}
        ood_attack_methods = ['clean']

        for o_a_m in ood_attack_methods:
            print('-----------------------------------------------')
            st = time.time()
            _, inputs_num, _, ood_in_msp_dict, _, ood_v_msp_dict, ood_v_ssp_dict \
                = eval_ood_util.get_ood_scores(model, NUM_IN_CLASSES, args.num_v_classes, socre_save_dir, out_datasets,
                                               ood_batch_size=args.ood_batch_size, device=device,
                                               num_oods_per_set=args.ood_batch_size * 4, img_size=img_size)
            ood_vssp_minus_inmsp_dict = {}
            ood_vmsp_minus_inmsp_dict = {}
            ood_inmsp_add_vssp_dict = {}
            for temp_key, _ in ood_v_ssp_dict.items():
                ood_vssp_minus_inmsp_dict[temp_key] = ood_v_ssp_dict[temp_key] - ood_in_msp_dict[temp_key]
                ood_vmsp_minus_inmsp_dict[temp_key] = ood_v_msp_dict[temp_key] - ood_in_msp_dict[temp_key]
                ood_inmsp_add_vssp_dict[temp_key] = ood_in_msp_dict[temp_key] + ood_v_ssp_dict[temp_key]

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
                                                               scoring_func=scoring_func, storage_device=storage_device)
                print("Under attack: {}".format(o_a_m))
                print("Performance on Individual OOD set: indiv_auc: {}, indiv_fprN: {}, indiv_tprN: {}, indiv_mean_score: {}, eval time: {}s"
                      .format(indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score, time.time() - st))
                print("Performance on Mixed OOD set: mixing_auc: {}, mixing_fprN: {}, mixing_tprN: {}, mixing_score: {}, eval time: {}s"
                      .format(mixing_auc, mixing_fprN, mixing_tprN, mixing_score, time.time() - st))
            print()


if __name__ == '__main__':
    # test_transfer()
    eval_main()
