import math
import os
import sys
import time

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import utils.svhn_loader as svhn
import numpy as np
from utils import nn_util


def get_ood_abs_dir():
    parent = os.path.dirname(os.path.realpath(__file__))
    parent = os.path.dirname(parent)
    parent = os.path.dirname(parent)
    ood_abs_dir = parent+"/datasets/ood_datasets/"
    return ood_abs_dir


def pick_worst_images(model, num_in_classes, images, adv_images, order='in_msp'):
    assert order in ['in_msp', '-v_ssp']
    with torch.no_grad():
        model.eval()
        inputs = images.detach().clone()
        nat_outputs = F.softmax(model(images), dim=1)
        adv_outputs = F.softmax(model(adv_images), dim=1)
        if order == 'in_msp':
            nat_in_msp, _ = torch.max(nat_outputs[:, :num_in_classes], dim=1)
            adv_in_msp, _ = torch.max(adv_outputs[:, :num_in_classes], dim=1)
            indices = adv_in_msp > nat_in_msp
            inputs[indices] = adv_images[indices]
        else:
            nat_v_ssp =nat_outputs[:, num_in_classes:].sum(dim=1)
            adv_v_ssp =adv_outputs[:, num_in_classes:].sum(dim=1)
            indices = adv_v_ssp > nat_v_ssp
            inputs[indices] = adv_images[indices]
    return inputs


def get_ood_scores(model, num_in_classes, num_v_classes, base_dir, out_datasets, ood_attack_method='clean',
                   ood_batch_size=256, num_oods_per_set=float('inf'), img_size=[32, 32], device=torch.device("cuda")):

    saved_logits_files = []
    all_miscls_v_classes = {}
    all_inputs_num = {}
    all_sum_in_scores = {}
    all_max_in_scores = {}
    all_sum_v_scores = {}
    all_max_v_scores = {}
    all_confidences = []
    
    ood_abs_dir = get_ood_abs_dir()

    for out_dataset in out_datasets:
        out_save_dir = os.path.join(base_dir, out_dataset)
        if not os.path.exists(out_save_dir):
            os.makedirs(out_save_dir)

        if out_dataset == 'svhn':
            testset_out = svhn.SVHN(ood_abs_dir + '/svhn/', split='test', transform=T.ToTensor(), download=True)
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        elif out_dataset == 'cifar10':
            dataloader = torchvision.datasets.CIFAR10
            test_out_loader = torch.utils.data.DataLoader(
                dataloader(ood_abs_dir + '/cifar10/', train=False, download=True, transform=T.ToTensor()),
                batch_size=ood_batch_size, shuffle=False, num_workers=1)
        elif out_dataset == 'cifar100':
            dataloader = torchvision.datasets.CIFAR100
            test_out_loader = torch.utils.data.DataLoader(
                dataloader(ood_abs_dir + '/cifar100/', train=False, download=True, transform=T.ToTensor()),
                batch_size=ood_batch_size, shuffle=False, num_workers=1)
        elif out_dataset == 'dtd':
            testset_out = torchvision.datasets.ImageFolder(root=ood_abs_dir + "/dtd/images", transform=T.Compose(
                [T.Resize(img_size[0]), T.CenterCrop(img_size[1]), T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        elif out_dataset == 'places365':
            testset_out = torchvision.datasets.ImageFolder(root=ood_abs_dir + "/places365/test_subset",
                                                           transform=T.Compose(
                                                               [T.Resize(img_size[0]), T.CenterCrop(img_size[1]),
                                                                T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)
        else:
            testset_out = torchvision.datasets.ImageFolder(ood_abs_dir + "/{}".format(out_dataset), transform=T.Compose(
                [T.Resize(32), T.CenterCrop(32), T.ToTensor()]))
            test_out_loader = torch.utils.data.DataLoader(testset_out, batch_size=ood_batch_size, shuffle=False,
                                                          num_workers=1)

        saved_logits_file = os.path.join(out_save_dir, "{}_test_ood_logits.npy".format(ood_attack_method))

        saved_misc_score_file = os.path.join(out_save_dir, "{}_test_ood_misc_scores.txt".format(ood_attack_method))
        f_misc_scores = open(saved_misc_score_file, 'w')

        # print("Processing out-of-distribution images")
        N = len(test_out_loader.dataset)
        count = 0
        cur_data_sum_in_scores = []
        cur_data_max_in_scores = []
        miscls_v_classes = 0
        cur_data_sum_v_scores = []
        cur_data_max_v_scores = []
        cur_data_logits = []
        for j, data in enumerate(test_out_loader):
            if (j + 1) * ood_batch_size > num_oods_per_set:
                break
            # print('evaluating detection performance on {} {}/{}'.format(out_dataset, j+1, len(test_out_loader)))
            images, labels = data
            images = images.to(device)
            curr_batch_size = images.shape[0]
            if ood_attack_method == 'clean':
                inputs = images
            else:
                raise ValueError('un-supported ood_attack_method: {}'.format(ood_attack_method))

            with torch.no_grad():
                model.eval()
                logits = model(inputs)
                outputs = F.softmax(logits, dim=1)
                all_confidences.append(outputs)
            _, whole_preds = torch.max(outputs, dim=1)

            max_in_scores, _ = torch.max(outputs[:, :num_in_classes], dim=1)
            sum_in_scores = torch.sum(outputs[:, :num_in_classes], dim=1)

            if num_v_classes > 0:
                miscls_v_indices = torch.logical_and((whole_preds >= num_in_classes), (whole_preds < outputs.size(1)))
                miscls_v_classes += miscls_v_indices.sum().item()

                max_v_scores, _ = torch.max(outputs[:, num_in_classes:], dim=1)
                sum_v_scores = torch.sum(outputs[:, num_in_classes:], dim=1)
            else:
                max_v_scores = torch.zeros((outputs.size(0),)).to(outputs.device)
                sum_v_scores = torch.zeros((outputs.size(0),)).to(outputs.device)

            for i in range(0, len(outputs)):
                f_misc_scores.write(
                    "{},{},{},{}\n".format(max_in_scores[i].cpu().numpy(), sum_in_scores[i].cpu().numpy(),
                                           max_v_scores[i].cpu().numpy(), sum_v_scores[i].cpu().numpy()))

            count += curr_batch_size
            cur_data_sum_in_scores.append(sum_in_scores)
            cur_data_max_in_scores.append(max_in_scores)
            cur_data_sum_v_scores.append(sum_v_scores)
            cur_data_max_v_scores.append(max_v_scores)
            cur_data_logits.append(logits)
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - st))

        cur_data_logits = torch.cat(cur_data_logits)
        np.save(saved_logits_file, cur_data_logits.cpu().numpy())
        f_misc_scores.close()

        saved_logits_files.append(saved_logits_file)
        all_inputs_num[out_dataset] = count
        all_sum_in_scores[out_dataset] = torch.cat(cur_data_sum_in_scores)
        all_max_in_scores[out_dataset] = torch.cat(cur_data_max_in_scores)
        all_sum_v_scores[out_dataset] = torch.cat(cur_data_sum_v_scores)
        all_max_v_scores[out_dataset] = torch.cat(cur_data_max_v_scores)
        all_miscls_v_classes[out_dataset] = miscls_v_classes

    return saved_logits_files, all_inputs_num, all_miscls_v_classes, all_max_in_scores, all_sum_in_scores, all_max_v_scores, all_sum_v_scores



def eval_on_signle_ood_dataset(in_scores, out_datasets, each_ood_set_score_dict, ts=[95], scoring_func='in_msp'):
    indiv_auc = {}
    sum_auc = 0

    indiv_fprN = {}
    sum_of_fprN = {}
    indiv_tprN = {}
    sum_of_tprN = {}

    indiv_mean_score = {}
    sum_of_mean_score = 0

    mixing_ood_scores = []

    for out_dataset in out_datasets:
        cur_ood_socres = each_ood_set_score_dict[out_dataset]
        cur_ood_auroc = nn_util.auroc(in_scores, cur_ood_socres, scoring_func=scoring_func)
        indiv_auc[out_dataset] = cur_ood_auroc
        sum_auc += cur_ood_auroc

        for t in ts:
            fprN, tau = nn_util.fpr_at_tprN(in_scores, cur_ood_socres, TPR=t, scoring_func=scoring_func)
            # print('out_dataset:', out_dataset, 'fprN：', fprN, 'tau:', tau)
            if t not in indiv_fprN:
                indiv_fprN[t] = {}
                sum_of_fprN[t] = 0
            indiv_fprN[t][out_dataset] = fprN
            sum_of_fprN[t] += fprN

            tprN, tau = nn_util.tpr_at_tnrN(in_scores, cur_ood_socres, TNR=t, scoring_func=scoring_func)
            # print('out_dataset:', out_dataset, 'tprN：', tprN, 'tau:', tau)
            if t not in indiv_tprN:
                indiv_tprN[t] = {}
                sum_of_tprN[t] = 0
            indiv_tprN[t][out_dataset] = tprN
            sum_of_tprN[t] += tprN

        cur_mean_socre = cur_ood_socres.mean().item()
        indiv_mean_score[out_dataset] = cur_mean_socre
        sum_of_mean_score += cur_mean_socre
        mixing_ood_scores.append(cur_ood_socres)

    avg_auc = sum_auc / len(out_datasets)
    avg_ood_score = sum_of_mean_score / len(out_datasets)

    mixing_ood_scores = torch.cat(mixing_ood_scores)
    mixing_score = mixing_ood_scores.mean().item()
    mixing_auc = nn_util.auroc(in_scores, mixing_ood_scores, scoring_func=scoring_func)

    avg_fprN = {}
    mixing_fprN = {}
    for t in ts:
        avg_fprN[t] = sum_of_fprN[t] / len(out_datasets)
        mixing_fprN[t], _ = nn_util.fpr_at_tprN(in_scores, mixing_ood_scores, TPR=t, scoring_func=scoring_func)

    avg_tprN = {}
    mixing_tprN = {}
    for t in ts:
        avg_tprN[t] = sum_of_tprN[t] / len(out_datasets)
        mixing_tprN[t], _ = nn_util.tpr_at_tnrN(in_scores, mixing_ood_scores, TNR=t, scoring_func=scoring_func)

    return (avg_auc, avg_fprN, avg_tprN, avg_ood_score), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
    mixing_auc, mixing_fprN, mixing_tprN, mixing_score)


def eval_ood_detection(model, num_in_classes, id_score_dict, socre_save_dir, out_datasets, args,
                       ood_attack_methods=['clean'], ts=[95], img_size=[32, 32]):
    storage_device = args.storage_device
    worst_ood_score_dict = {}
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes
    for ood_attack_method in ood_attack_methods:
        st = time.time()
        _, inputs_num, miscls_v_classes, all_max_in_scores_dict, all_sum_in_scores_dict, all_max_v_scores_dict, \
        all_sum_v_scores_dict = get_ood_scores(model, num_in_classes, num_v_classes, socre_save_dir, out_datasets,
                                               ood_attack_method, ood_batch_size=args.ood_batch_size, img_size=img_size)
        ood_vssp_inmsp_dict = {}
        ood_vmsp_inmsp_dict = {}
        for temp_key, _ in all_sum_v_scores_dict.items():
            ood_vssp_inmsp_dict[temp_key] = all_sum_v_scores_dict[temp_key] - all_max_in_scores_dict[temp_key]
            ood_vmsp_inmsp_dict[temp_key] = all_max_v_scores_dict[temp_key] - all_max_in_scores_dict[temp_key]

        for sc_key, ood_values in {'in_msp': [all_max_in_scores_dict, 'in_msp']}.items():
            if sc_key not in id_score_dict:
                continue
            con_f = id_score_dict[sc_key]
            print('=====> using {} as scoring function =====>'.format(sc_key))
            conf_t = ood_values[0]
            scoring_func = ood_values[1]
            (avg_auc, avg_fprN, avg_tprN, avg_ood_score), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
                mixing_auc, mixing_fprN, mixing_tprN, mixing_score) \
                = eval_on_signle_ood_dataset(con_f, out_datasets, conf_t, ts=ts, scoring_func=scoring_func,
                                             storage_device=storage_device)
            total_inputs_num = 0
            sum_miscls_v_classes = 0
            each_outdata_max_in_score = {}
            each_outdata_sum_in_score = {}
            each_outdata_max_v_score = {}
            each_outdata_sum_v_score = {}
            for out_dataset in out_datasets:
                if out_dataset not in worst_ood_score_dict:
                    worst_ood_score_dict[out_dataset] = all_max_in_scores_dict[out_dataset]
                else:
                    indcs = worst_ood_score_dict[out_dataset] < all_max_in_scores_dict[out_dataset]
                    worst_ood_score_dict[out_dataset][indcs] = all_max_in_scores_dict[out_dataset][indcs]
    
                total_inputs_num += inputs_num[out_dataset]
                sum_miscls_v_classes += miscls_v_classes[out_dataset]
    
                each_outdata_max_in_score[out_dataset] = all_max_in_scores_dict[out_dataset].mean().item()
                each_outdata_sum_in_score[out_dataset] = all_sum_in_scores_dict[out_dataset].mean().item()
                each_outdata_max_v_score[out_dataset] = all_max_v_scores_dict[out_dataset].mean().item()
                each_outdata_sum_v_score[out_dataset] = all_sum_v_scores_dict[out_dataset].mean().item()
    
            print('----------------------------------------------------------------')
            print('With "{} attacked" OOD inputs, total miscls v-classes:{}(/{}), '
                  'each_outdata_max_in_score:{}, each_outdata_sum_in_score:{}, '
                  'each_outdata_max_v_score:{}, each_outdata_sum_v_score:{}'
                  .format(ood_attack_method, sum_miscls_v_classes, inputs_num,
                          each_outdata_max_in_score, each_outdata_sum_in_score,
                          each_outdata_max_v_score, each_outdata_sum_v_score))
            print('Individual Performance: indiv_auc:{}, indiv_fprN:{}, indiv_tprN:{}, indiv_mean_score:{}'
                  .format(indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score))
            print('Avg Performance: avg_auc:{}, avg_fprN:{}, avg_tprN:{}, avg_ood_score:{}'
                  .format(avg_auc, avg_fprN, avg_tprN, avg_ood_score))
            print('Performance on Mixing: mixing_auc:{}, mixing_fprN:{}, mixing_tprN:{}, mixing_score:{}'
                  .format(mixing_auc, mixing_fprN, mixing_tprN, mixing_score))
            print('eval time: {}s'.format(time.time() - st))
        print()


def get_all_data(test_loader, max_num=float('inf')):
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
        if len(y_test) >= max_num:
            return x_test, y_test
    return x_test, y_test


def eval_in_and_out(model, test_loader, num_in_classes, args):
    socre_save_dir = args.save_socre_dir
    if not os.path.exists(socre_save_dir):
        os.makedirs(socre_save_dir)

    # # clean IDs
    clean_id_misc_scores_file = os.path.join(socre_save_dir, 'clean_id_misc_scores.txt')
    clean_id_logits_file = os.path.join(socre_save_dir, 'clean_id_logits.npy')
    nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr, nat_id_in_msp, nat_id_corr_prob, nat_id_v_msp, nat_id_v_ssp = \
        nn_util.eval(model, test_loader, num_in_classes, args.num_v_classes, misc_score_file=clean_id_misc_scores_file,
                     lotis_file=clean_id_logits_file)
    nat_id_mmsp = nat_id_in_msp.mean().item()
    nat_id_corr_mprob = nat_id_corr_prob.mean().item()
    nat_id_vssp_minus_inmsp = nat_id_v_ssp - nat_id_in_msp
    nat_id_m_vssp_minus_inmsp = nat_id_vssp_minus_inmsp.mean().item()
    print('Testing nat ID acc: {}, nat miscls v-classes: {}, nat micls v-classes corr: {}, mean of nat-mmsp: {}, '
          'mean of corr-prob: {}, mean of (v-ssp - in-msp):{}'.format(nat_id_acc, nat_id_v_cls, nat_id_v_cls_corr,
                                            nat_id_mmsp, nat_id_corr_mprob, nat_id_m_vssp_minus_inmsp))

    # detecting OODs
    ts = [80, 85, 90, 95]
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
    elif args.dataset == 'svhn':
        out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
    print('=============================== OOD Detection performance =====================================')
    st = time.time()
    id_score_dict = {'in_msp': nat_id_in_msp}
    ood_attack_methods = ['clean']
    eval_ood_detection(model, num_in_classes, id_score_dict=id_score_dict, socre_save_dir=socre_save_dir,
                       out_datasets=out_datasets, args=args, ood_attack_methods=ood_attack_methods, ts=ts)
    print('evaluation time: {}s'.format(time.time() - st))
    print()