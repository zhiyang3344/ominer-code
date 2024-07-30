import torch
import torch.nn.functional as F
import numpy as np


# def label_smoothing_without_vnodes(true_label, alpha=0.5, num_classes=10):
#     y_ls = F.one_hot(true_label, num_classes=num_classes) * (1 - alpha) + alpha / (num_classes - 1)
#     y_ls = y_ls.to(true_label.device)
#     return y_ls
#
#
# def constuct_soft_label(true_label, alpha, num_classes, vnodes):
#     if alpha == 0:
#         return F.one_hot(true_label, num_classes=num_classes + vnodes).to(true_label.device)
#     y_ls = torch.zeros((len(true_label), num_classes + vnodes), device=true_label.device)
#     indexes = [i for i in range(len(true_label))]
#     y_ls[indexes, true_label] += (1 - alpha)
#     y_ls[:, num_classes:num_classes + vnodes] = alpha / vnodes
#     return y_ls


# def constuct_soft_label_with_uniform(true_label, alpha, num_classes, c_nodes):
#     # one-hot
#     if alpha == 0:
#         return F.one_hot(true_label, num_classes=num_classes + c_nodes).to(true_label.device)
#     # standard label smoothing
#     elif alpha != 0 and c_nodes == 0:
#         y_ls = F.one_hot(true_label, num_classes=num_classes) * (1 - alpha) + alpha / (num_classes - 1)
#         y_ls = y_ls.to(true_label.device)
#         return y_ls
#     # with additional virtuall classes
#     elif alpha != 0 and c_nodes != 0:
#         y_vc = torch.zeros((len(true_label), num_classes + c_nodes), device=true_label.device)
#         indexes = [i for i in range(len(true_label))]
#         y_vc[indexes, true_label] += (1 - alpha)
#         y_vc[:, num_classes:num_classes + c_nodes] = alpha / c_nodes
#         return y_vc
#     else:
#         raise ValueError('error alpha:{0} or vnodes:{1}'.format(alpha, c_nodes))


def kl_loss_from_prob(student_prob, teacher_prob):
    batch_size = student_prob.size()[0]
    criterion_kl = torch.nn.KLDivLoss(size_average=False)
    log_student_prob = torch.log(student_prob)
    kl_loss = (1.0 / batch_size) * criterion_kl(log_student_prob, teacher_prob)
    return kl_loss


# def virtual_self_distillion_loss(student_logits, num_classes, temperature, alpha, y_true):
#     # batch_size = student_logits.size()[0]
#     student_prob = F.softmax(student_logits, dim=1)
#
#     teacher_logits = torch.zeros_like(student_logits)
#     teacher_logits[:, num_classes:] = student_logits[:, :num_classes]
#     indexes = [i for i in range(len(student_logits))]
#     teacher_logits[indexes, num_classes + y_true] = 0.0
#     teacher_logits[indexes, y_true] = student_logits[indexes, y_true]
#     teacher_prob = F.softmax(teacher_logits, dim=1)
#
#     # KL divergence * (temperature^2)
#     kl_loss = temperature * temperature * kl_loss_from_prob(student_prob / temperature, teacher_prob / temperature)
#     xent = torch.nn.CrossEntropyLoss()
#     ce_loss = xent(student_logits, y_true)
#     loss = alpha * kl_loss + (1 - alpha) * ce_loss
#     return loss


# def rob_distillion_loss(nat_student_logits, adv_student_logits, nat_teacher_logits, temperature, alpha, y_true):
#     KL_loss = torch.nn.KLDivLoss()
#     XENT_loss = torch.nn.CrossEntropyLoss()
#     kl_loss = KL_loss(F.log_softmax(adv_student_logits / temperature, dim=1),
#                       F.softmax(nat_teacher_logits / temperature, dim=1))
#     ce_loss = XENT_loss(nat_student_logits, y_true)
#     loss = temperature * temperature * alpha * kl_loss + (1 - alpha) * ce_loss
#     return loss


# def virtual_self_distillion_loss(student_logits, num_classes, temperature, alpha, y_true):
#     # batch_size = student_logits.size()[0]
#     student_prob = F.softmax(student_logits, dim=1)
#
#     teacher_prob = torch.zeros_like(student_logits)
#     teacher_prob[:, num_classes:] = student_prob[:, :num_classes]
#     indexes = [i for i in range(len(student_logits))]
#     teacher_prob[indexes, num_classes + y_true] = 0.0
#     teacher_prob[indexes, y_true] = student_prob[indexes, y_true]
#
#     # KL divergence * (temperature^2)
#     kl_loss = temperature * temperature * kl_loss_from_prob(student_prob / temperature, teacher_prob / temperature)
#     xent = torch.nn.CrossEntropyLoss()
#     ce_loss = xent(student_logits, y_true)
#     loss = alpha * kl_loss + (1 - alpha) * ce_loss
#     return loss


def cross_entropy_soft_target(logit, y_soft, reduction='mean'):
    batch_size = logit.size()[0]
    log_prob = F.log_softmax(logit, dim=1)
    if reduction == 'none':
        loss = -torch.sum(log_prob * y_soft, dim=1)
    elif reduction == 'mean':
        loss = -torch.sum(log_prob * y_soft) / batch_size
    else:
        print('un-supported reduction: {}'.format(reduction))
    return loss


# def cross_entropy_out_classes(logits, y_soft, num_in_classes, num_out_classes, normlization=True, reduction='mean'):
#     if num_out_classes <= 0:
#         raise ValueError('num_out_classes should be >= 0 !')
#     batch_size = logits.size()[0]
#     log_prob = F.log_softmax(logits, dim=1)
#     tmp_prob = log_prob[:, num_in_classes:num_in_classes + num_out_classes]
#
#     tmp_target = y_soft[:, num_in_classes:num_in_classes + num_out_classes]
#     if normlization:
#         tmp_target = tmp_target / torch.sum(tmp_target, dim=1).unsqueeze(dim=1)
#     if reduction == 'none':
#         loss = -torch.sum(tmp_prob * tmp_target, dim=1)
#     elif reduction == 'mean':
#         loss = -torch.sum(tmp_prob * tmp_target) / batch_size
#     else:
#         print('un-supported reduction: {}'.format(reduction))
#     return loss


# def cross_entropy_specified_dims(logits, y_soft, st_dim, end_dim, normlization=True, reduction='mean'):
#     batch_size = logits.size()[0]
#     log_prob = F.log_softmax(logits, dim=1)
#     tmp_prob = log_prob[:, st_dim:end_dim]
#     tmp_target = y_soft[:, st_dim:end_dim]
#     if normlization:
#         tmp_target = tmp_target / torch.sum(tmp_target, dim=1).unsqueeze(dim=1)
#
#     if reduction == 'none':
#         loss = -torch.sum(tmp_prob * tmp_target, dim=1)
#     elif reduction == 'mean':
#         loss = -torch.sum(tmp_prob * tmp_target) / batch_size
#     else:
#         print('un-supported reduction: {}'.format(reduction))
#
#     return loss


def eval(model, test_loader, num_in_classes, num_v_classes, misc_score_file='', lotis_file=''):
    model.eval()
    correct = 0
    total = 0
    located_in_vcls = 0
    located_in_vcls_corr = 0
    all_corr_prob = []
    all_in_msp = []
    all_v_ssp = []
    all_v_msp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for i, data in enumerate(test_loader):
        inputs, targets = data
        targets = targets.cuda(non_blocking=True)
        inputs = inputs.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)

        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        correct_indices = in_preds == targets
        correct += correct_indices.sum().item()
        all_corr_prob.append(in_msp[correct_indices])
        all_in_msp.append(in_msp)
        if num_v_classes > 0:
            v_ssp = probs[:, num_in_classes:num_in_classes + num_v_classes].sum(dim=1)
            v_msp = probs[:, num_in_classes:num_in_classes + num_v_classes].max(dim=1)[0]
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_v_ssp.append(v_ssp)
        all_v_msp.append(v_msp)

        _, whole_preds = torch.max(probs, dim=1)
        located_in_vclass_indices = torch.logical_and((whole_preds >= num_in_classes),
                                                      (whole_preds < num_in_classes + num_v_classes))
        located_in_vcls += located_in_vclass_indices.sum().item()
        located_in_vcls_corr += (
            torch.logical_and(correct_indices, located_in_vclass_indices)).sum().item()
        total += targets.size(0)

        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{},{},{},{}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                          v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
        if lotis_file != '':
            all_logits.append(logits)

    all_corr_prob = torch.cat(all_corr_prob)
    all_in_msp = torch.cat(all_in_msp)
    all_v_ssp = torch.cat(all_v_ssp)
    all_v_msp = torch.cat(all_v_msp)
    acc = (float(correct) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, located_in_vcls, located_in_vcls_corr, all_in_msp, all_corr_prob, all_v_msp, all_v_ssp


def eval_from_data(model, x, y, batch_size, num_in_classes, num_v_classes, misc_score_file='', lotis_file=''):
    model.eval()
    num_examples = len(x)
    correct = 0
    total = 0
    located_in_vcls = 0
    located_in_vcls_corr = 0
    all_corr_prob = []
    all_in_msp = []
    all_v_ssp = []
    all_v_msp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_x = x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]
        # compute output
        with torch.no_grad():
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        correct_indices = in_preds == batch_y
        correct += correct_indices.sum().item()
        all_corr_prob.append(in_msp[correct_indices])
        all_in_msp.append(in_msp)
        if num_v_classes > 0:
            v_ssp = probs[:, num_in_classes:num_in_classes + num_v_classes].sum(dim=1)
            v_msp = probs[:, num_in_classes:num_in_classes + num_v_classes].max(dim=1)[0]
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_v_ssp.append(v_ssp)
        all_v_msp.append(v_msp)

        _, whole_preds = torch.max(probs, dim=1)
        located_in_vclass_indices = torch.logical_and((whole_preds >= num_in_classes),
                                                      (whole_preds < num_in_classes + num_v_classes))
        located_in_vcls += located_in_vclass_indices.sum().item()
        located_in_vcls_corr += (
            torch.logical_and(correct_indices, located_in_vclass_indices)).sum().item()
        total += batch_y.size(0)

        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{},{},{},{}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                          v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
        if lotis_file != '':
            all_logits.append(logits)

    all_corr_prob = torch.cat(all_corr_prob)
    all_in_msp = torch.cat(all_in_msp)
    all_v_ssp = torch.cat(all_v_ssp)
    all_v_msp = torch.cat(all_v_msp)
    acc = (float(correct) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, located_in_vcls, located_in_vcls_corr, all_in_msp, all_corr_prob, all_v_msp, all_v_ssp


# def auroc(in_conf, out_conf):
#     # calculate the AUROC
#     min_conf = np.min([np.min(in_conf), np.min(out_conf)])
#     max_conf = np.max([np.max(in_conf), np.max(out_conf)])
#     gap = (max_conf - min_conf) / 100000
#     aurocBase = 0.0
#     fprTemp = 1.0
#     for t in np.arange(min_conf, max_conf, gap):
#         tpr = np.sum(np.sum(in_conf >= t)) / np.float(len(in_conf))
#         fpr = np.sum(np.sum(out_conf > t)) / np.float(len(out_conf))
#         aurocBase += (-fpr + fprTemp) * tpr
#         fprTemp = fpr
#     aurocBase += fpr * tpr
#     return aurocBase


def eval_with_out_classes(model, test_loader, num_in_classes, num_out_classes, num_v_classes=0, misc_score_file='',
                          lotis_file=''):
    model.eval()
    correct = 0
    total = 0
    miscls_out_cls = 0
    miscls_v_cls = 0
    all_corr_prob = []
    all_in_msp = []
    all_out_msp = []
    all_out_ssp = []
    all_v_msp = []
    all_v_ssp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for i, data in enumerate(test_loader):
        batch_x, batch_y = data
        batch_x = batch_x.cuda(non_blocking=True)
        batch_y = batch_y.cuda(non_blocking=True)

        with torch.no_grad():
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

        # num_v_classes = probs.size(1) - num_in_classes - num_out_classes
        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        correct_indcs = in_preds == batch_y
        correct += correct_indcs.sum().item()
        all_corr_prob.append(in_msp[correct_indcs])
        all_in_msp.append(in_msp)
        if num_out_classes > 0:
            out_msp = probs[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
            out_ssp = probs[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1)
        else:
            out_msp = torch.zeros((probs.size(0),), device=probs.device)
            out_ssp = torch.zeros((probs.size(0),), device=probs.device)
        if num_v_classes > 0:
            v_msp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].max(dim=1)[0]
            v_ssp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].sum(dim=1)
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_out_msp.append(out_msp)
        all_out_ssp.append(out_ssp)
        all_v_msp.append(v_msp)
        all_v_ssp.append(v_ssp)

        whole_probs, whole_preds = torch.max(probs, dim=1)
        if num_out_classes > 0:
            miscls_out_indcs = torch.logical_and((whole_preds < num_in_classes + num_out_classes),
                                                          (whole_preds >= num_in_classes))
            miscls_out_cls += (miscls_out_indcs).sum().item()
        if num_v_classes > 0:
            miscls_v_indcs = torch.logical_and((whole_preds >= num_in_classes + num_out_classes),
                                                        (whole_preds < probs.size(1)))
            miscls_v_cls += (miscls_v_indcs).sum().item()

        total += batch_x.size(0)
        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{}, {}, {}, {}, {}, {}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                                     out_msp[i].cpu().numpy(), out_ssp[i].cpu().numpy(),
                                                                     v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
        if lotis_file != '':
            all_logits.append(logits)

    all_in_msp = torch.cat(all_in_msp)
    all_corr_prob = torch.cat(all_corr_prob)
    all_out_msp = torch.cat(all_out_msp)
    all_out_ssp = torch.cat(all_out_ssp)
    all_v_msp = torch.cat(all_v_msp)
    all_v_ssp = torch.cat(all_v_ssp)
    acc = (float(correct) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, miscls_out_cls, miscls_v_cls, all_in_msp, all_corr_prob, all_out_msp, all_out_ssp, all_v_msp, all_v_ssp


def eval_from_data_with_out_classes(model, x, y, batch_size, num_in_classes, num_out_classes, num_v_classes=0,
                                    misc_score_file='', lotis_file=''):
    model.eval()
    num_examples = len(x)
    correct = 0
    total = 0
    miscls_out_cls = 0
    miscls_v_cls = 0
    
    all_corr_prob = []
    all_in_msp = []
    all_out_msp = []
    all_out_ssp = []
    all_v_msp = []
    all_v_ssp = []
    all_logits = []

    if misc_score_file != '':
        f_misc_score = open(misc_score_file, 'w')

    for idx in range(0, len(x), batch_size):
        st_idx = idx
        end_idx = min(idx + batch_size, num_examples)
        batch_x = x[st_idx:end_idx]
        batch_y = y[st_idx:end_idx]

        with torch.no_grad():
            logits = model(batch_x)
            probs = F.softmax(logits, dim=1)

        # num_v_classes = probs.size(1) - num_in_classes - num_out_classes
        in_msp, in_preds = torch.max(probs[:, :num_in_classes], dim=1)
        correct_indcs = in_preds == batch_y
        correct += correct_indcs.sum().item()
        all_corr_prob.append(in_msp[correct_indcs])
        all_in_msp.append(in_msp)

        if num_out_classes > 0:
            out_msp = probs[:, num_in_classes:num_in_classes + num_out_classes].max(dim=1)[0]
            out_ssp = probs[:, num_in_classes:num_in_classes + num_out_classes].sum(dim=1)
        else:
            out_msp = torch.zeros((probs.size(0),), device=probs.device)
            out_ssp = torch.zeros((probs.size(0),), device=probs.device)
        if num_v_classes > 0:
            v_msp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].max(dim=1)[0]
            v_ssp = probs[:, num_in_classes + num_out_classes:num_in_classes + num_out_classes + num_v_classes].sum(dim=1)
        else:
            v_msp = torch.zeros((probs.size(0),), device=probs.device)
            v_ssp = torch.zeros((probs.size(0),), device=probs.device)
        all_out_msp.append(out_msp)
        all_out_ssp.append(out_ssp)
        all_v_msp.append(v_msp)
        all_v_ssp.append(v_ssp)

        whole_probs, whole_preds = torch.max(probs, dim=1)
        if num_out_classes > 0:
            miscls_out_indcs = torch.logical_and((whole_preds < num_in_classes + num_out_classes),
                                                 (whole_preds >= num_in_classes))
            miscls_out_cls += (miscls_out_indcs).sum().item()
        if num_v_classes > 0:
            miscls_v_indcs = torch.logical_and((whole_preds >= num_in_classes + num_out_classes),
                                               (whole_preds < probs.size(1)))
            miscls_v_cls += (miscls_v_indcs).sum().item()

        total += batch_x.size(0)
        if misc_score_file != '':
            in_ssp = torch.sum(probs[:, :num_in_classes], dim=1)
            for i in range(0, len(probs)):
                f_misc_score.write("{}, {}, {}, {}, {}, {}\n".format(in_msp[i].cpu().numpy(), in_ssp[i].cpu().numpy(),
                                                                     out_msp[i].cpu().numpy(), out_ssp[i].cpu().numpy(),
                                                                     v_msp[i].cpu().numpy(), v_ssp[i].cpu().numpy()))
        if lotis_file != '':
            all_logits.append(logits)

    all_in_msp = torch.cat(all_in_msp)
    all_corr_prob = torch.cat(all_corr_prob)
    all_out_msp = torch.cat(all_out_msp)
    all_out_ssp = torch.cat(all_out_ssp)
    all_v_msp = torch.cat(all_v_msp)
    all_v_ssp = torch.cat(all_v_ssp)

    acc = (float(correct) / total)
    if misc_score_file != '':
        f_misc_score.close()
    if lotis_file != '':
        all_logits = torch.cat(all_logits)
        np.save(lotis_file, all_logits.cpu().numpy())

    return acc, miscls_out_cls, miscls_v_cls, all_in_msp, all_corr_prob, all_out_msp, all_out_ssp, all_v_msp, all_v_ssp


def auroc(conf_f, conf_t, scoring_func='in_msp'):
    # calculate the AUROC
    if torch.is_tensor(conf_f):
        conf_f = np.array(conf_f.cpu())
    if torch.is_tensor(conf_t):
        conf_t = np.array(conf_t.cpu())
    min_conf = np.min(np.concatenate((conf_f, conf_t)))
    max_conf = np.max(np.concatenate((conf_f, conf_t)))
    gap = (max_conf - min_conf) / 100000
    auroc_base = 0.0
    fpr_temp = 1.0
    if scoring_func == 'in_msp':
        for t in np.arange(max_conf, min_conf, -gap):
            fpr = np.sum(conf_f <= t) / len(conf_f)
            tpr = np.sum(conf_t <= t) / len(conf_t)
            auroc_base += (fpr_temp - fpr) * tpr
            fpr_temp = fpr
    elif scoring_func in ['r_ssp']:
        for t in np.arange(min_conf, max_conf, gap):
            fpr = np.sum(conf_f >= t) / len(conf_f)
            tpr = np.sum(conf_t >= t) / len(conf_t)
            auroc_base += (fpr_temp - fpr) * tpr
            fpr_temp = fpr
    else:
        raise ValueError('un-supported scoring_func:', scoring_func)
    auroc_base += fpr * tpr

    return auroc_base


# def tpr95(in_conf, out_conf):
#     # calculate the falsepositive error when tpr is 95%
#     min_conf = np.min([np.min(in_conf), np.min(out_conf)])
#     max_conf = np.max([np.max(in_conf), np.max(out_conf)])
#     gap = (max_conf - min_conf) / 100000
#     total = 0.0
#     fpr = 0.0
#
#     for tau in np.arange(min_conf, max_conf, gap):
#         tpr = np.sum(np.sum(in_conf >= tau)) / np.float(len(in_conf))
#         out_fpr = np.sum(np.sum(out_conf > tau)) / np.float(len(out_conf))
#         if tpr <= 0.9505 and tpr >= 0.9495:
#             fpr += out_fpr
#             total += 1
#     if total == 0:
#         taus = np.arange(min_conf, max_conf, gap)
#         taus = taus[::-1]  # reverse taus
#         for tau in taus:
#             tpr = np.sum(np.sum(in_conf >= tau)) / np.float(len(in_conf))
#             out_fpr = np.sum(np.sum(out_conf > tau)) / np.float(len(out_conf))
#             if tpr >= 0.9495:
#                 print('Warning, no TPR locate in 95%, return OUT FPR at TPR{0}%'.format(tpr))
#                 return out_fpr
#         return 0
#     else:
#         fprBase = fpr / total
#
#     return fprBase


# def ood_tpr_at_id_tnrN(in_conf, out_conf, N=0.95):
#     # calculate the fnr of OODs when the fpr of IDs is 5%
#     # 接受对id样本的5%误检率，检查对OOD样本的误检率，越小越好，tau与待检测的OOD无关
#     min_conf = torch.min(torch.cat((in_conf, out_conf)))
#     max_conf = torch.max(torch.cat((in_conf, out_conf)))
#     gap = (max_conf - min_conf) / 100000
#     total = 0.0
#     fnr = 0.0
#
#     for tau in torch.arange(min_conf, max_conf, gap):
#         id_tnr = torch.sum(in_conf >= tau) / len(in_conf)
#         ood_tpr = torch.sum(out_conf < tau) / len(out_conf)
#         if id_tnr <= (N + 0.005) and id_tnr >= (N - 0.005):
#             fnr += ood_tpr
#             total += 1
#         # print('tau: {}, tpr:{} ,out_fpr:{}'.format(tau, tpr, out_fpr))
#     if total == 0:
#         taus = torch.arange(min_conf, max_conf, gap)
#         taus = taus[::-1]  # reverse taus
#         for tau in taus:
#             id_tnr = torch.sum(in_conf >= tau) / len(in_conf)
#             ood_tpr = torch.sum(out_conf < tau) / len(out_conf)
#             if id_tnr >= (N - 0.0005):
#                 print('###### Warning, no id_tnr = {}, return ood_tpr ({}) at id_tnr = {} ######'.format(int(N * 100), ood_tpr, tau))
#                 return ood_tpr.item()
#         return -1
#     else:
#         fnrBase = fnr / total
#         if torch.is_tensor(fnrBase):
#             fnrBase = fnrBase.item()
#
#     return fnrBase


# conf_t: confidences of OODs
# conf_f: condfienced of IDs
def tpr_at_tnrN(conf_f, conf_t, TNR=95, scoring_func='in_msp'):
    assert TNR >= 1
    if torch.is_tensor(conf_f):
        conf_f = np.array(conf_f.cpu())
    if torch.is_tensor(conf_t):
        conf_t = np.array(conf_t.cpu())

    if scoring_func in ['in_msp']:
        PERC = np.percentile(conf_f, 100 - TNR)
        TPR = np.sum(conf_t <= PERC) / len(conf_t)
        return TPR, PERC
    elif scoring_func in ['r_ssp']:
        PERC = np.percentile(conf_f, TNR)
        TPR = np.sum(conf_t >= PERC) / len(conf_t)
        return TPR, PERC
    else:
        raise ValueError('un-supported scoring_func:', scoring_func)


# conf_t: confidences of OODs
# conf_f: condfienced of IDs
def fpr_at_tprN(conf_f, conf_t, TPR=95, scoring_func='in_msp'):
    assert TPR >= 1
    if torch.is_tensor(conf_f):
        conf_f = np.array(conf_f.cpu())
    if torch.is_tensor(conf_t):
        conf_t = np.array(conf_t.cpu())

    if scoring_func in ['in_msp']:
        PERC = np.percentile(conf_t, TPR)
        FPR = np.sum(conf_f <= PERC) / len(conf_f)
        return FPR, PERC
    elif scoring_func in ['r_ssp']:
        PERC = np.percentile(conf_t, 100 - TPR)
        FPR = np.sum(conf_f >= PERC) / len(conf_f)
        return FPR, PERC
    else:
        raise ValueError('un-supported scoring_func:', scoring_func)


# def id_fpr_at_ood_tprN(in_conf, out_conf, N=0.95):
#     # calculate the fpr of IDs when the tpr of OODs is 95%
#     # 当对oods的检测正确类为95%时，把id误检为ood的概率，越小越好，tau与待检测的OOD相关
#     min_conf = torch.min(torch.cat((in_conf, out_conf)))
#     max_conf = torch.max(torch.cat((in_conf, out_conf)))
#     gap = (max_conf - min_conf) / 100000
#     total = 0.0
#     fpr = 0.0
#
#     for tau in torch.arange(min_conf, max_conf, gap):
#         tpr_ood = torch.sum(out_conf <= tau) / len(out_conf)
#         fpr_id = torch.sum(in_conf < tau) / len(in_conf)
#         if tpr_ood <= (N + 0.005) and tpr_ood >= (N - 0.005):
#             fpr += fpr_id
#             total += 1
#         # print('tau: {}, tpr:{} ,out_fpr:{}'.format(tau, tpr, out_fpr))
#     if total == 0:
#         taus = torch.arange(min_conf, max_conf, gap)
#         # taus = taus[::-1]  # reverse taus
#         for tau in taus:
#             tpr_ood = torch.sum(out_conf <= tau) / len(out_conf)
#             fpr_id = torch.sum(in_conf < tau) / len(in_conf)
#             if tpr_ood >= (N - 0.0005):
#                 print('###### Warning, no tpr_ood = {}, return fpr_id ({}) at tpr_ood = {} ########'.format(N, fpr_id, tpr_ood))
#                 return fpr_id.item()
#         return -1
#     else:
#         fprBase = fpr / total
#         if torch.is_tensor(fprBase):
#             fprBase = fprBase.item()
#
#     return fprBase


if __name__ == '__main__':
    pass


