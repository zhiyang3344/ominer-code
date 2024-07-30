import os
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
    ood_abs_dir = parent + "/datasets/ood_datasets/"
    return ood_abs_dir


def pick_worst_images(model, num_in_classes, images, adv_images):
    with torch.no_grad():
        model.eval()
        inputs = images.detach().clone()
        nat_outputs = F.softmax(model(images), dim=1)
        # nat_max_in_scores, _ = torch.max(nat_outputs[:, :num_in_classes], dim=1)
        nat_ood_scores = torch.sum(nat_outputs[:, num_in_classes:], dim=1)

        adv_outputs = F.softmax(model(adv_images), dim=1)
        # adv_max_in_scores, _ = torch.max(adv_outputs[:, :num_in_classes], dim=1)
        adv_ood_scores = torch.sum(adv_outputs[:, num_in_classes:], dim=1)

        indices = adv_ood_scores < nat_ood_scores
        inputs[indices] = adv_images[indices]
    return inputs


def get_ood_scores(model, num_in_classes, num_out_classes, num_v_classes, base_dir, out_datasets,
                   ood_attack_method='clean', ood_batch_size=256, num_oods_per_set=float('inf'), img_size=[32, 32],
                   device=torch.device("cuda")):

    saved_logits_files = []
    all_in_msps = {}
    all_out_msps = {}
    all_out_ssps = {}
    all_v_msps = {}
    all_v_ssps = {}
    all_miscls_in_classes = {}
    all_miscls_v_classes = {}
    all_inputs_num = {}

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
        cur_in_msps = []
        cur_out_msps = []
        cur_out_ssps = []
        cur_v_msps = []
        cur_v_ssps = []
        miscls_in_classes = 0
        miscls_v_classes = 0
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
                probs = F.softmax(model(inputs), dim=1)

            whole_scores, whole_preds = torch.max(probs, dim=1)
            miscls_in_indices = (whole_preds < num_in_classes)
            miscls_in_classes += miscls_in_indices.sum().item()

            if num_out_classes > 0:
                max_in_scores, _ = torch.max(probs[:, :num_in_classes], dim=1)
                sum_in_scores = torch.sum(probs[:, :num_in_classes], dim=1)

                max_out_scores, _ = torch.max(probs[:, num_in_classes:num_in_classes + num_out_classes], dim=1)
                sum_out_scores = torch.sum(probs[:, num_in_classes:num_in_classes + num_out_classes], dim=1)
            else:
                raise ValueError('num_out_classes should be > 0 !')

            # num_v_classes = probs.size(1) - num_in_classes - num_out_classes
            if num_v_classes > 0:
                miscls_v_indices = torch.logical_and((whole_preds >= num_in_classes + num_out_classes),
                                                     (whole_preds < num_in_classes + num_out_classes + num_v_classes))
                miscls_v_classes += miscls_v_indices.sum().item()
                max_v_scores, _ = torch.max(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes], dim=1)
                sum_v_scores = torch.sum(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes], dim=1)
            else:
                max_v_scores = torch.zeros((probs.size(0), )).to(device)
                sum_v_scores = torch.zeros((probs.size(0), )).to(device)

            for i in range(0, len(probs)):
                f_misc_scores.write(
                    "{},{},{},{},{},{}\n".format(max_in_scores[i].cpu().numpy(), sum_in_scores[i].cpu().numpy(),
                                                 max_out_scores[i].cpu().numpy(), sum_out_scores[i].cpu().numpy(),
                                                 max_v_scores[i].cpu().numpy(), sum_v_scores[i].cpu().numpy()))

            count += curr_batch_size
            cur_in_msps.append(max_in_scores)
            cur_out_msps.append(probs[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0])
            cur_out_ssps.append(probs[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1))
            if num_v_classes > 0:
                cur_v_msps.append(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].max(dim=1)[0])
                cur_v_ssps.append(probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].sum(dim=1))
            else:
                cur_v_msps.append(torch.zeros((probs.size(0), ), device=probs.device))
                cur_v_ssps.append(torch.zeros((probs.size(0),), device=probs.device))
            cur_data_logits.append(logits)
            # print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, N, time.time() - st))
            # print('batch:', j, 'msp:', max_in_scores.mean().item())
            # exit()

        cur_data_logits = torch.cat(cur_data_logits)
        np.save(saved_logits_file, cur_data_logits.cpu().numpy())
        f_misc_scores.close()
        saved_logits_files.append(saved_logits_file)
        all_in_msps[out_dataset] = torch.cat(cur_in_msps)
        all_out_msps[out_dataset] = torch.cat(cur_out_msps)
        all_out_ssps[out_dataset] = torch.cat(cur_out_ssps)
        all_v_msps[out_dataset] = torch.cat(cur_v_msps)
        all_v_ssps[out_dataset] = torch.cat(cur_v_ssps)
        all_inputs_num[out_dataset] = count
        all_miscls_in_classes[out_dataset] = miscls_in_classes
        all_miscls_v_classes[out_dataset] = miscls_v_classes
    return saved_logits_files, all_inputs_num, all_miscls_in_classes, all_miscls_v_classes, all_in_msps, all_out_msps, all_out_ssps, all_v_msps, all_v_ssps


def eval_on_signle_ood_dataset(id_scores, out_datasets, indiv_ood_score_dict, ts=[95], scoring_func='in_msp'):
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
        cur_ood_socres = indiv_ood_score_dict[out_dataset]
        cur_ood_auroc = nn_util.auroc(id_scores, cur_ood_socres, scoring_func=scoring_func)
        indiv_auc[out_dataset] = cur_ood_auroc
        sum_auc += cur_ood_auroc

        for t in ts:
            fprN, _ = nn_util.fpr_at_tprN(id_scores, cur_ood_socres, TPR=t, scoring_func=scoring_func)
            if t not in indiv_fprN:
                indiv_fprN[t] = {}
                sum_of_fprN[t] = 0
            indiv_fprN[t][out_dataset] = fprN
            sum_of_fprN[t] += fprN

            tprN, _ = nn_util.tpr_at_tnrN(id_scores, cur_ood_socres, TNR=t, scoring_func=scoring_func)
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
    mixing_auc = nn_util.auroc(id_scores, mixing_ood_scores, scoring_func=scoring_func)

    avg_fprN = {}
    mixing_fprN = {}
    for t in ts:
        avg_fprN[t] = sum_of_fprN[t] / len(out_datasets)
        mixing_fprN[t], _ = nn_util.fpr_at_tprN(id_scores, mixing_ood_scores, TPR=t, scoring_func=scoring_func)

    avg_tprN = {}
    mixing_tprN = {}
    for t in ts:
        avg_tprN[t] = sum_of_tprN[t] / len(out_datasets)
        mixing_tprN[t], _ = nn_util.tpr_at_tnrN(id_scores, mixing_ood_scores, TNR=t, scoring_func=scoring_func)

    return (avg_auc, avg_fprN, avg_tprN, avg_ood_score), (indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score), (
        mixing_auc, mixing_fprN, mixing_tprN, mixing_score)


def eval_ood_detection(model, num_in_classes, id_score_dict, socre_save_dir, out_datasets, args,
                       ood_attack_methods=['clean'], ts=[95], img_size=[32, 32]):
    storage_device = args.storage_device
    worst_ood_score_dict = {}
    num_out_classes = args.num_out_classes
    num_v_classes = 0
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes
    for ood_attack_method in ood_attack_methods:
        st=time.time()
        saved_logits_files, inputs_num, miscls_in_classes, miscls_v_classes, in_msp_dict, out_msp_dict, out_ssp_dict, _, v_ssp_dict \
            = get_ood_scores(model, num_in_classes, num_out_classes, num_v_classes, socre_save_dir, out_datasets,
                             ood_attack_method, ood_batch_size=args.ood_batch_size, img_size=img_size)
        v_and_out_minus_in_dict = {}
        v_and_out_dict = {}
        out_minus_in_dict = {}
        for temp_key, _ in out_ssp_dict.items():
            v_and_out_minus_in_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key] - in_msp_dict[temp_key]
            v_and_out_dict[temp_key] = v_ssp_dict[temp_key] + out_ssp_dict[temp_key]
            out_minus_in_dict[temp_key] = out_ssp_dict[temp_key] - in_msp_dict[temp_key]

        for sc_key, ood_values in {'in_msp': [in_msp_dict, 'in_msp'],
                                   'v-out_ssp_minus_in_msp': [v_and_out_minus_in_dict, 'r_ssp'],
                                   'v-out_ssp': [v_and_out_dict, 'r_ssp'],
                                   'out_ssp_minus_in_msp': [out_minus_in_dict, 'r_ssp'],
                                   'out_msp': [out_msp_dict, 'r_ssp'],
                                   'out_ssp': [out_ssp_dict, 'r_ssp']}.items():
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
            sum_miscls_in_classes = 0
            for out_dataset in out_datasets:
                if out_dataset not in worst_ood_score_dict:
                    worst_ood_score_dict[out_dataset] = conf_t[out_dataset]
                else:
                    indcs = worst_ood_score_dict[out_dataset] < conf_t[out_dataset]
                    worst_ood_score_dict[out_dataset][indcs] = conf_t[out_dataset][indcs]

                total_inputs_num += inputs_num[out_dataset]
                sum_miscls_v_classes += miscls_v_classes[out_dataset]
                sum_miscls_in_classes += miscls_in_classes[out_dataset]
            print('----------------------------------------------------------------')
            print('With "{} (attacked)" OOD inputs, total miscls in-classes:{}(/{}), total miscls v-classes:{}(/{})'
                  .format(ood_attack_method, sum_miscls_in_classes, inputs_num, sum_miscls_v_classes, inputs_num, ))
            print('Individual Performance: avg_auc:{}, avg_fprN:{}, avg_tprN:{}, avg_ood_score:{}'
                  .format(indiv_auc, indiv_fprN, indiv_tprN, indiv_mean_score))
            print('Avg Performance: indiv_auc:{}, indiv_fprN:{}, indiv_tprN:{}, indiv_mean_score:{}'
                  .format(avg_auc, avg_fprN, avg_tprN, avg_ood_score))
            print('Performance on Mixing: mixing_auc:{}, mixing_fprN:{}, mixing_tprN:{}, mixing_score:{}'.format(
                mixing_auc, mixing_fprN, mixing_tprN, mixing_score))
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
    num_out_classes = 0
    if 'num_out_classes' in args:
        num_out_classes = args.num_out_classes
    elif 'advid_out_classes' in args and 'ood_out_classes' in args:
        num_out_classes = args.advid_out_classes + args.ood_out_classes
    if 'num_v_classes' in args:
        num_v_classes = args.num_v_classes
    else:
        num_v_classes = 0

    # clean IDs
    print('----------------------------------------------------------------')
    clean_id_misc_scores_file = os.path.join(socre_save_dir, 'clean_id_misc_scores.txt')
    clean_id_logits_file = os.path.join(socre_save_dir, 'clean_id_logits.npy')
    nat_id_acc, nat_miscls_out_cls, nat_miscls_v_cls, nat_id_msp, nat_id_corr_prob, nat_id_out_msp, nat_id_out_ssp, _, nat_id_v_ssp = \
        nn_util.eval_with_out_classes(model, test_loader, num_in_classes, num_out_classes, num_v_classes=num_v_classes,
                                      misc_score_file=clean_id_misc_scores_file, lotis_file = clean_id_logits_file)
    nat_id_mmsp = nat_id_msp.mean().item()
    nat_id_corr_mprob = nat_id_corr_prob.mean().item()
    nat_id_out_mmsp = nat_id_out_msp.mean().item()
    nat_id_out_mssp = nat_id_out_ssp.mean().item()
    nat_id_vout = nat_id_v_ssp + nat_id_out_ssp
    nat_id_m_vout = nat_id_vout.mean().item()
    nat_id_vout_in = nat_id_v_ssp + nat_id_out_ssp - nat_id_msp
    nat_id_m_vout_in = nat_id_vout_in.mean().item()
    print('Testing nat ID acc: {}, miscls into num_out_classes: {}, miscls into num_v_classes: {}, mean of in-msp: {}, '
          'mean of nat corr-prob: {}, mean of out-msp: {}, mean of out-ssp: {}, mean of (v_ssp + out-ssp): {}, '
          'mean of (v_ssp + out-ssp - in-msp): {}'.format(nat_id_acc, nat_miscls_out_cls, nat_miscls_v_cls, nat_id_mmsp,
                                                          nat_id_corr_mprob, nat_id_out_mmsp, nat_id_out_mssp,
                                                          nat_id_m_vout, nat_id_m_vout_in))
    print()

    # detecting OODs
    ts = [80, 85, 90, 95]
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        out_datasets = ['places365', 'svhn', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']
    elif args.dataset == 'svhn':
        out_datasets = ['cifar10', 'cifar100', 'places365', 'LSUN', 'LSUN_resize', 'iSUN', 'dtd']

    print('=============================== OOD Detection performance =====================================')
    st = time.time()
    id_score_dict = {'in_msp': nat_id_msp, 'out_msp': nat_id_out_msp, 'out_ssp': nat_id_out_ssp}
    ood_attack_methods = ['clean']
    eval_ood_detection(model, num_in_classes, id_score_dict, socre_save_dir=socre_save_dir,
                       out_datasets=out_datasets, args=args, ood_attack_methods=ood_attack_methods, ts=ts)
    print('evaluation time: {}s'.format(time.time() - st))
    print()
