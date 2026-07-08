#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import copy, sys
import time
import numpy as np
from tqdm import tqdm
import torch
from tensorboardX import SummaryWriter
import random
import torch.utils.model_zoo as model_zoo
from pathlib import Path

lib_dir = (Path(__file__).parent / ".." / "lib").resolve()
if str(lib_dir) not in sys.path:
    sys.path.insert(0, str(lib_dir))
mod_dir = (Path(__file__).parent / ".." / "lib" / "models").resolve()
if str(mod_dir) not in sys.path:
    sys.path.insert(0, str(mod_dir))

from resnet import resnet18
from options import args_parser
from update import LocalUpdate, save_protos, LocalTest, test_inference_new_het_lt
from models import CNNMnist, CNNFemnist
# from utils import get_dataset, average_weights, exp_details, proto_aggregation, agg_func, average_weights_per, average_weights_sem
# from utils import (get_dataset, average_weights, exp_details,
#                    proto_aggregation, agg_func, average_weights_per, average_weights_sem,
#                    get_client_class_freq, acmp_proto_aggregation, get_topk_protos_for_clients)
from utils import (get_dataset, average_weights, exp_details,
                   proto_aggregation, agg_func, average_weights_per, average_weights_sem,
                   get_client_class_freq, acmp_proto_aggregation, get_topk_protos_for_clients,
                   server_update_boundary_teacher)
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):
#     """
#     Unified FedProto_taskheter:
#     - Same training logic as FedProto_modelheter
#     - Only model architecture is different (task_heter uses uniform CNN)
#     """

#     summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_taskheter_' +
#                                    str(args.ways) + 'w' + str(args.shots) + 's' +
#                                    str(args.stdev) + 'e_' + str(args.num_users) +
#                                    'u_' + str(args.rounds) + 'r')
    
#     # global protos: ACMP-FL stores dict or list based on acmp settings
#     global_protos = {}
#     train_loss, train_accuracy = [], []
#     # ACMP-FL: each client maintains top-k clusters
#     topk_dict = {i: None for i in range(args.num_users)}
#     idxs_users = np.arange(args.num_users)

#     print(">>> Initial Accuracy Test (Before Training)")
#     acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
#         args,
#         local_model_list,
#         test_dataset,
#         classes_list,
#         user_groups_lt,
#         global_protos={}
#     )

#     print('For all users (with protos), mean test acc: {:.5f}, std: {:.5f}'
#           .format(np.mean(acc_list_g), np.std(acc_list_g)))
#     print('For all users (w/o protos), mean test acc: {:.5f}, std: {:.5f}'
#           .format(np.mean(acc_list_l), np.std(acc_list_l)))
#     print("===============================================================\n")

#     # ============================
#     # Training Loop
#     # ============================
#     for round in tqdm(range(args.rounds)):
#         local_weights, local_losses, local_protos = [], [], {}
#         local_losses_dict = []
#         local_acc_list = []

#         print(f'\n | Global Training Round : {round + 1} |\n')

#         # Dynamic rotation for rotated MNIST
#         if args.dataset == 'mnist_rotated_dynamic':
#             try:
#                 from sampling import apply_dynamic_rotation
#             except ImportError:
#                 from lib.sampling import apply_dynamic_rotation

#             train_dataset = apply_dynamic_rotation(train_dataset, round, user_groups, args.num_users)
#             print(f"🔄 Dynamic rotation applied at round {round + 1}.")

#         proto_loss = 0
#         # -----------------------------
#         # ⭐ 使用 frac 随机选取部分客户端参与训练
#         # -----------------------------
#         # m = max(int(args.frac * args.num_users), 1)
#         # selected_users = np.random.choice(idxs_users, m, replace=False)

#         # for idx in selected_users:
#         for idx in idxs_users:

#             # Use identical setting to model_heter
#             local_model = LocalUpdate(
#                 args=args,
#                 dataset=train_dataset,
#                 idxs=user_groups[idx],
#                 class_freq_i=client_class_freq[idx]   # <-- IMPORTANT for ACMP-FL
#             )

#             # ACMP-FL local update
#             if args.alg == 'acmpfl':
#                 topk = topk_dict.get(idx, None)
#                 # w, loss, acc, protos = local_model.update_weights_acmp(
#                 #     args, idx, global_protos, topk,
#                 #     model=copy.deepcopy(local_model_list[idx]),
#                 #     global_round=round
#                 # )
#                 w, loss, acc, protos = local_model.update_weights_acmp(
#                 args=args,
#                 idx=idx,
#                 global_protos=global_protos,
#                 topk_protos=topk,
#                 model=copy.deepcopy(local_model_list[idx]),
#                 global_round=round
#                 )

#                 # w, loss, acc, protos = local_model.update_weights_het(
#                 #     args, idx, global_protos, 
#                 #     model=copy.deepcopy(local_model_list[idx]), 
#                 #     global_round=round)
#             else:
#                 w, loss, acc, protos = local_model.update_weights_het(
#                     args, idx, global_protos,
#                     model=copy.deepcopy(local_model_list[idx]),
#                     global_round=round
#                 )

#             local_protos[idx] = protos   # protos = {cls : prototype_vec_dim50}

#             local_weights.append(copy.deepcopy(w))
#             local_losses.append(copy.deepcopy(loss["total"]))
#             # 统计到列表中，用于 Round Summary
#             local_losses_dict.append(loss)
#             local_acc_list.append(acc)

#             # 打印每个用户的 loss 和 acc
#             print(f"  |- User {idx}: total={loss['total']:.4f}, "
#                   f"task={loss.get('task',0):.4f}, "
#                   f"align={loss.get('align',0):.4f}, "
#                   f"tail={loss.get('tail',0):.4f}, "
#                   f"acc={acc:.4f}")

#             # summary_writer.add_scalar('Train/Loss/user'+str(idx+1), loss['total'], round)
#             # summary_writer.add_scalar('Train/Loss1/user'+str(idx+1), loss['1'], round)
#             # summary_writer.add_scalar('Train/Loss2/user'+str(idx+1), loss['2'], round)
#             # summary_writer.add_scalar('Train/Acc/user'+str(idx+1), acc, round)

#             summary_writer.add_scalar(f'Train/Loss_total/user{idx+1}', loss['total'], round)
#             summary_writer.add_scalar(f'Train/Loss_task/user{idx+1}', loss['task'], round)
#             summary_writer.add_scalar(f'Train/Loss_align/user{idx+1}', loss['align'], round)
#             summary_writer.add_scalar(f'Train/Loss_tail/user{idx+1}', loss['tail'], round)
#             summary_writer.add_scalar(f'Train/Loss_boundary/user{idx+1}', loss['boundary'], round)


#         # update local models
#         for idx in idxs_users:
#             lm = copy.deepcopy(local_model_list[idx])
#             lm.load_state_dict(local_weights[idx], strict=True)
#             local_model_list[idx] = lm
#         # 只更新参与训练的客户端
#         # for i, idx in enumerate(selected_users):
#         #     lm = copy.deepcopy(local_model_list[idx])
#         #     lm.load_state_dict(local_weights[i], strict=True)
#         #     local_model_list[idx] = lm

#         # global proto aggregation EXACT SAME AS model_heter
#         if args.alg == 'acmpfl':
#             global_protos, W = acmp_proto_aggregation(
#                 args, global_protos, local_protos, client_class_freq
#             )
#             topk_dict = get_topk_protos_for_clients(args, W)
#         else:
#             global_protos = proto_aggregation(local_protos)

#             # ===== Global Loss Summary =====
#         loss_avg = np.mean(local_losses)
#         train_loss.append(loss_avg)

#         print(f"  >> Avg Global Training Loss (Round {round+1}) = {loss_avg:.4f}\n")
#         # ===== Round Summary =====
#         mean_total = np.mean([d['total'] for d in local_losses_dict])
#         mean_task = np.mean([d.get('task',0) for d in local_losses_dict])
#         mean_align = np.mean([d.get('align',0) for d in local_losses_dict])
#         mean_tail  = np.mean([d.get('tail',0) for d in local_losses_dict])
#         mean_acc   = np.mean(local_acc_list)
        
#         print("\n====== Round {} Summary ======".format(round + 1))
#         print(f"  Avg Total Loss : {mean_total:.4f}")
#         print(f"    - Task Loss  : {mean_task:.4f}")
#         print(f"    - Align Loss : {mean_align:.4f}")
#         print(f"    - Tail Loss  : {mean_tail:.4f}")
#         print(f"  Avg Local Acc  : {mean_acc:.4f}")
#         print("================================\n")

        
#         save_intervals = [1, 20, 40, 60, 80, args.rounds]
#         if (round + 1) in save_intervals:
#             save_dir = f"./saved_protos/round_{round+1}"
#             os.makedirs(save_dir, exist_ok=True)

#             local_proto_list, local_label_list, local_client_idx = [], [], []
#             for i in local_protos.keys():
#                 for c in local_protos[i].keys():
#                     tensor = local_protos[i][c]
#                     if isinstance(tensor, list):
#                         for p in tensor:
#                             local_proto_list.append(p.detach().cpu().numpy())
#                             local_label_list.append(c)
#                             local_client_idx.append(i)
#                     else:
#                         local_proto_list.append(tensor.detach().cpu().numpy())
#                         local_label_list.append(c)
#                         local_client_idx.append(i)

#             np.save(os.path.join(save_dir, "acmpfl_local_protos.npy"), local_proto_list)
#             np.save(os.path.join(save_dir, "acmpfl_labels.npy"), local_label_list)
#             np.save(os.path.join(save_dir, "acmpfl_client_idx.npy"), local_client_idx)

#             # save W only for ACMP-FL
#             if args.alg == "acmpfl" and 'W' in locals():
#                 np.save(os.path.join(save_dir, "acmpfl_w_matrix.npy"), W)
#                 print(f"Saved W matrix for round {round+1}")
#             else:
#                 print(f"No W matrix saved (alg={args.alg})")

#     # ======================
#     # Final Testing
#     # ======================
#     acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
#         args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

#     # global_proto_acc = np.mean(acc_list_g)
#     # local_model_acc  = np.mean(acc_list_l)
    
#     # print(f"[Round {round+1}] Global Proto ACC = {global_proto_acc:.4f}  |  Local Model ACC = {local_model_acc:.4f}")
    
#     # # ======================================================
#     # # 🔑 Save Accuracy Curves for Later Plotting
#     # # ======================================================
#     # if round == 0:
#     #     global_acc_curve = []
#     #     local_acc_curve = []
    
#     # global_acc_curve.append(global_proto_acc)
#     # local_acc_curve.append(local_model_acc)
    
#     # np.save(f'./acc_global_proto_{args.alg}.npy', np.array(global_acc_curve))
#     # np.save(f'./acc_local_model_{args.alg}.npy', np.array(local_acc_curve))

#     print("====================================================")
#     print('For all users (with protos), mean acc = {:.5f}, std = {:.5f}'
#           .format(np.mean(acc_list_g), np.std(acc_list_g)))
#     print('For all users (w/o protos), mean acc = {:.5f}, std = {:.5f}'
#           .format(np.mean(acc_list_l), np.std(acc_list_l)))
#     print('For all users (proto loss), mean = {:.5f}, std = {:.5f}'
#           .format(np.mean(loss_list), np.std(loss_list)))

def FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list):

    summary_writer = SummaryWriter('../tensorboard/'+ args.dataset +'_fedproto_taskheter_' +
                                   str(args.ways) + 'w' + str(args.shots) + 's' +
                                   str(args.stdev) + 'e_' + str(args.num_users) +
                                   'u_' + str(args.rounds) + 'r')
    
    global_protos = {}
    train_loss, train_accuracy = [], []
    topk_dict = {i: None for i in range(args.num_users)}
    idxs_users = np.arange(args.num_users)

    # ============================================================
    # ⭐ 新增：保存每一轮 ACC 的列表
    # ============================================================
    acc_curve = []

    print(">>> Initial Accuracy Test (Before Training)")
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args,
        local_model_list,
        test_dataset,
        classes_list,
        user_groups_lt,
        global_protos={}
    )

    print('For all users (with protos), mean test acc: {:.5f}, std: {:.5f}'
          .format(np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean test acc: {:.5f}, std: {:.5f}'
          .format(np.mean(acc_list_l), np.std(acc_list_l)))
    print("===============================================================\n")

    # ⭐ 保存初始 ACC
    acc_curve.append(np.mean(acc_list_g))

    # ============================
    # Training Loop
    # ============================
    for round in tqdm(range(args.rounds)):
        local_weights, local_losses, local_protos = [], [], {}
        local_losses_dict = []
        local_acc_list = []

        print(f'\n | Global Training Round : {round + 1} |\n')

        # Dynamic rotation for rotated MNIST
        if args.dataset == 'mnist_rotated_dynamic':
            try:
                from sampling import apply_dynamic_rotation
            except ImportError:
                from lib.sampling import apply_dynamic_rotation

            train_dataset = apply_dynamic_rotation(train_dataset, round, user_groups, args.num_users)
            print(f"🔄 Dynamic rotation applied at round {round + 1}.")

        proto_loss = 0

        for idx in idxs_users:

            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                class_freq_i=client_class_freq[idx]
            )

            if args.alg == 'acmpfl':
                topk = topk_dict.get(idx, None)
                w, loss, acc, protos = local_model.update_weights_acmp(
                    args=args,
                    idx=idx,
                    global_protos=global_protos,
                    topk_protos=topk,
                    model=copy.deepcopy(local_model_list[idx]),
                    global_round=round
                )
            else:
                w, loss, acc, protos = local_model.update_weights_het(
                    args, idx, global_protos, 
                    model=copy.deepcopy(local_model_list[idx]), 
                    global_round=round
                )

            local_protos[idx] = protos

            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss["total"]))
            local_losses_dict.append(loss)
            local_acc_list.append(acc)

            print(f"  |- User {idx}: total={loss['total']:.4f}, "
                  f"task={loss.get('task',0):.4f}, "
                  f"align={loss.get('align',0):.4f}, "
                  f"tail={loss.get('tail',0):.4f}, "
                  f"boundary={loss.get('boundary',0):.4f}, "
                  f"acc={acc:.4f}")

            summary_writer.add_scalar(f'Train/Loss_total/user{idx+1}', loss['total'], round)
            summary_writer.add_scalar(f'Train/Loss_task/user{idx+1}', loss['task'], round)
            summary_writer.add_scalar(f'Train/Loss_align/user{idx+1}', loss['align'], round)
            summary_writer.add_scalar(f'Train/Loss_tail/user{idx+1}', loss['tail'], round)
            summary_writer.add_scalar(f'Train/Loss_boundary/user{idx+1}', loss['boundary'], round)

        # ======================================================
        # Update local models
        # ======================================================
        for idx in idxs_users:
            lm = copy.deepcopy(local_model_list[idx])
            lm.load_state_dict(local_weights[idx], strict=True)
            local_model_list[idx] = lm

        # ======================================================
        # Global prototype aggregation
        # ======================================================
        if args.alg == 'acmpfl':
            global_protos, W = acmp_proto_aggregation(
                args, global_protos, local_protos, client_class_freq
            )
            topk_dict = get_topk_protos_for_clients(args, W)
        else:
            global_protos = proto_aggregation(local_protos)

        loss_avg = np.mean(local_losses)
        train_loss.append(loss_avg)

        print(f"  >> Avg Global Training Loss (Round {round+1}) = {loss_avg:.4f}\n")

        # ======================================================
        # ⭐ 每一轮 round 测试 ACC（with global protos）
        # ======================================================
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
            args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

        round_acc = np.mean(acc_list_g)
        acc_curve.append(round_acc)

        print(f"  >> Test ACC (with protos) = {round_acc:.4f}")

        # ======================================================
        # 原型保存（保持原样）
        # ======================================================
        save_intervals = [1, 20, 40, 60, 80, args.rounds]
        if (round + 1) in save_intervals:
            save_dir = f"./saved_protos/round_{round+1}"
            os.makedirs(save_dir, exist_ok=True)

            local_proto_list, local_label_list, local_client_idx = [], [], []
            for i in local_protos.keys():
                for c in local_protos[i].keys():
                    tensor = local_protos[i][c]
                    if isinstance(tensor, list):
                        for p in tensor:
                            local_proto_list.append(p.detach().cpu().numpy())
                            local_label_list.append(c)
                            local_client_idx.append(i)
                    else:
                        local_proto_list.append(tensor.detach().cpu().numpy())
                        local_label_list.append(c)
                        local_client_idx.append(i)

            np.save(os.path.join(save_dir, "acmpfl_local_protos.npy"), local_proto_list)
            np.save(os.path.join(save_dir, "acmpfl_labels.npy"), local_label_list)
            np.save(os.path.join(save_dir, "acmpfl_client_idx.npy"), local_client_idx)

            if args.alg == "acmpfl" and 'W' in locals():
                np.save(os.path.join(save_dir, "acmpfl_w_matrix.npy"), W)

    # ===========================
    # Final Print
    # ===========================
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos)

    print("====================================================")
    print('For all users (with protos), mean acc = {:.5f}, std = {:.5f}'
          .format(np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean acc = {:.5f}, std = {:.5f}'
          .format(np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (proto loss), mean = {:.5f}, std = {:.5f}'
          .format(np.mean(loss_list), np.std(loss_list)))

    # ============================================================
    # ⭐ 保存每轮 ACC 到 npy 文件（唯一新增的输出）
    # ============================================================
    np.save("fedproto_taskheter_acc.npy", np.array(acc_curve))
    print("🔥 Saved per-round ACC → fedproto_taskheter_acc.npy")




def FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt,
                        local_model_list, classes_list):

    summary_writer = SummaryWriter(
        '../tensorboard/' + args.dataset + '_fedproto_mh_' +
        str(args.ways) + 'w' + str(args.shots) + 's' +
        str(args.stdev) + 'e_' + str(args.num_users) +
        'u_' + str(args.rounds) + 'r'
    )

    # ✅ 与 taskheter 一致：全局原型用 dict
    global_protos = {}
    teacher, teacher_ema = None, None
    # ACMP-FL: 为每个客户端维护 Top-k 原型簇
    topk_dict = {i: None for i in range(args.num_users)}
    idxs_users = np.arange(args.num_users)

    train_loss, train_accuracy = [], []

    # ============================================================
    # ⭐ 新增：保存每一轮 ACC 的列表（与 taskheter 一致）
    # ============================================================
    acc_curve = []

    # ================================================================
    #   Initial Accuracy (Before Any Training)
    # ================================================================
    print(">>> Initial Accuracy Test (Before Training)")

    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args,
        local_model_list,
        test_dataset,
        classes_list,
        user_groups_lt,
        global_protos={}  # 初始无全局原型
    )

    print('For all users (with protos), mean test acc: {:.5f}, std: {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o  protos), mean test acc: {:.5f}, std: {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print("===============================================================\n")

    # ⭐ 保存初始 ACC（with protos）
    acc_curve.append(np.mean(acc_list_g))

    # ============================
    # Training Loop
    # ============================
    for round in tqdm(range(args.rounds)):

        local_weights, local_losses, local_protos = [], [], {}
        local_losses_dict = []
        local_acc_list = []

        print(f'\n | Global Training Round : {round + 1} |\n')

        # ================================================
        # 🔁 Dynamic Rotated MNIST（每轮概念漂移）
        # ================================================
        if args.dataset == 'mnist_rotated_dynamic':
            try:
                from sampling import apply_dynamic_rotation
            except ImportError:
                from lib.sampling import apply_dynamic_rotation

            train_dataset = apply_dynamic_rotation(train_dataset, round, user_groups, args.num_users)
            print(f"🔄 Dynamic rotation applied at round {round + 1}.")

        proto_loss = 0

        # ============================
        # Local Updates (all clients)
        # ============================
        for idx in idxs_users:

            local_model = LocalUpdate(
                args=args,
                dataset=train_dataset,
                idxs=user_groups[idx],
                class_freq_i=client_class_freq[idx]
            )

            # ✅ 与 taskheter 对齐：acmpfl 走 update_weights_acmp，并用 topk_protos
            if args.alg == 'acmpfl':
                topk = topk_dict.get(idx, None)
                w, loss, acc, protos = local_model.update_weights_acmp(
                    args=args,
                    idx=idx,
                    global_protos=global_protos,
                    topk_protos=topk,
                    model=copy.deepcopy(local_model_list[idx]),
                    global_round=round,
                    teacher_ema=teacher_ema
                )
            else:
                w, loss, acc, protos = local_model.update_weights_het(
                    args, idx, global_protos,
                    model=copy.deepcopy(local_model_list[idx]),
                    global_round=round
                )

            # 你原来这里做了 agg_func(protos) 再上传
            # agg_protos = agg_func(protos)
            # local_protos[idx] = agg_protos
            if args.alg == 'acmpfl':
               local_protos[idx] = protos          # ✅ 直接上传
            else:
               local_protos[idx] = agg_func(protos) # FedProto 的旧逻辑才需要


            local_weights.append(copy.deepcopy(w))
            local_losses.append(copy.deepcopy(loss.get("total", 0.0)))
            local_losses_dict.append(loss)
            local_acc_list.append(acc)

            # 打印（尽量兼容不同 loss key）
            print(f"  |- User {idx}: total={loss.get('total',0):.4f}, "
                  f"task={loss.get('task', loss.get('1',0)):.4f}, "
                  f"align={loss.get('align',0):.4f}, "
                  f"tail={loss.get('tail',0):.4f}, "
                  f"proto={loss.get('2',0):.4f}, "
                  f"boundary={loss.get('boundary',0):.4f}, "
                  f"acc={acc:.4f}")

            # ✅ TensorBoard：按 taskheter 的命名写（缺失就写 0）
            summary_writer.add_scalar(f'Train/Loss_total/user{idx+1}', loss.get('total', 0.0), round)
            summary_writer.add_scalar(f'Train/Loss_task/user{idx+1}',  loss.get('task', loss.get('1', 0.0)), round)
            summary_writer.add_scalar(f'Train/Loss_align/user{idx+1}', loss.get('align', 0.0), round)
            summary_writer.add_scalar(f'Train/Loss_tail/user{idx+1}',  loss.get('tail', 0.0), round)
            summary_writer.add_scalar(f'Train/Loss_boundary/user{idx+1}', loss.get('boundary', 0.0), round)
            # 兼容你 modelheter 里原来的 proto loss=loss['2']
            summary_writer.add_scalar(f'Train/Loss_proto/user{idx+1}', loss.get('2', 0.0), round)
            summary_writer.add_scalar(f'Train/Acc/user{idx+1}', acc, round)

            proto_loss += loss.get('2', 0.0)

        # ======================================================
        # Update local models
        # ======================================================
        for idx in idxs_users:
            lm = copy.deepcopy(local_model_list[idx])
            lm.load_state_dict(local_weights[idx], strict=True)
            local_model_list[idx] = lm

        # ======================================================
        # Global prototype aggregation
        # ======================================================
        if args.alg == 'acmpfl':
            global_protos, W = acmp_proto_aggregation(
                args, global_protos, local_protos, client_class_freq
            )
            topk_dict = get_topk_protos_for_clients(args, W)
        else:
            global_protos = proto_aggregation(local_protos)

        teacher, teacher_ema = server_update_boundary_teacher(
            args, teacher, teacher_ema, global_protos
        )

        loss_avg = np.mean(local_losses)
        train_loss.append(loss_avg)
        print(f"  >> Avg Global Training Loss (Round {round+1}) = {loss_avg:.4f}\n")

        # ======================================================
        # ⭐ 每一轮 round 测试 ACC（with global protos）——与 taskheter 一致
        # ======================================================
        acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
            args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos
        )

        round_acc = np.mean(acc_list_g)
        acc_curve.append(round_acc)
        print(f"  >> Test ACC (with protos) = {round_acc:.4f}")

        # ====================================================
        # 原型保存（基本保持原样）
        # ====================================================
        save_intervals = [1, 20, 40, 60, 80, args.rounds]
        if (round + 1) in save_intervals:
            save_dir = f'./saved_protos/round_{round+1}'
            os.makedirs(save_dir, exist_ok=True)

            local_proto_list = []
            local_label_list = []
            local_client_idx = []

            for i in local_protos.keys():
                for c in local_protos[i].keys():
                    proto_tensor = local_protos[i][c]
                    if isinstance(proto_tensor, list):
                        for p in proto_tensor:
                            local_proto_list.append(p.detach().cpu().numpy())
                            local_label_list.append(c)
                            local_client_idx.append(i)
                    else:
                        local_proto_list.append(proto_tensor.detach().cpu().numpy())
                        local_label_list.append(c)
                        local_client_idx.append(i)

            np.save(os.path.join(save_dir, 'acmpfl_local_protos.npy'), np.array(local_proto_list))
            np.save(os.path.join(save_dir, 'acmpfl_labels.npy'), np.array(local_label_list))
            np.save(os.path.join(save_dir, 'acmpfl_client_idx.npy'), np.array(local_client_idx))
            print(f"✅ [Round {round+1}] Saved {len(local_proto_list)} local prototypes to {save_dir}/")

            # 保存 W（更稳健一点）
            if args.alg == 'acmpfl' and 'W' in locals() and W is not None:
                np.save(os.path.join(save_dir, 'acmpfl_w_matrix.npy'), W)
                try:
                    shape_info = getattr(W, "shape", None)
                    if shape_info is None and isinstance(W, dict):
                        shape_info = (len(W), len(next(iter(W.values()))))
                    print(f"✅ [Round {round+1}] Saved W matrix, shape = {shape_info}")
                except Exception:
                    print(f"✅ [Round {round+1}] Saved W matrix.")
            else:
                print(f"⚠️ [Round {round+1}] No W matrix to save for {args.alg}")

    # ===========================
    # Final Print
    # ===========================
    acc_list_l, acc_list_g, loss_list = test_inference_new_het_lt(
        args, local_model_list, test_dataset, classes_list, user_groups_lt, global_protos
    )

    print("====================================================")
    print('For all users (with protos), mean acc = {:.5f}, std = {:.5f}'.format(
        np.mean(acc_list_g), np.std(acc_list_g)))
    print('For all users (w/o protos), mean acc = {:.5f}, std = {:.5f}'.format(
        np.mean(acc_list_l), np.std(acc_list_l)))
    print('For all users (proto loss), mean = {:.5f}, std = {:.5f}'.format(
        np.mean(loss_list), np.std(loss_list)))

    # ============================================================
    # ⭐ 保存每轮 ACC 到 npy 文件（与 taskheter 同逻辑）
    # ============================================================
    np.save("fedproto_modelheter_acc.npy", np.array(acc_curve))
    print("🔥 Saved per-round ACC → fedproto_modelheter_acc.npy")


if __name__ == '__main__':
    start_time = time.time()

    args = args_parser()
    exp_details(args)

    # set random seeds
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.device == 'cuda':
        torch.cuda.set_device(args.gpu)
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # ============================================
    # load dataset and user groups (robust version)
    # ============================================
    n_list = np.random.randint(
        max(2, args.ways - args.stdev),
        min(args.num_classes, args.ways + args.stdev +1),
        args.num_users
    )
   


    # 动态安全生成 k_list，防止 low >= high 报错
    def safe_randint(low, high, size, default_val):
        low = max(1, low)
        if low >= high:
            print(f"⚠️ [Warning] Invalid randint range: low={low}, high={high}. Using fixed value = {default_val}")
            return np.full(size, default_val, dtype=int)
        return np.random.randint(low, high, size)

    if args.dataset == 'mnist':
        k_list = safe_randint(args.shots - args.stdev + 1, args.shots + args.stdev - 1, args.num_users, args.shots)
    elif args.dataset == 'cifar10':
        k_list = safe_randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users, args.shots)
    elif args.dataset == 'cifar100':
        k_list = np.full(args.num_users, args.shots, dtype=int)  # 固定shots
    elif args.dataset == 'femnist':
        k_list = safe_randint(args.shots - args.stdev + 1, args.shots + args.stdev + 1, args.num_users, args.shots)
    else:
        k_list = np.full(args.num_users, args.shots, dtype=int)


    # train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = get_dataset(args, n_list, k_list)
    # train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt = \
    #     get_dataset(args, n_list, k_list)
    train_dataset, test_dataset, user_groups, user_groups_lt, classes_list, classes_list_gt, client_class_freq = \
        get_dataset(args, n_list, k_list)
        
    # ====================================================
    # 🔹 Rotated MNIST: 概念漂移（Concept Drift）增强模式
    # ====================================================
    # if args.dataset == 'mnist_rotated':
    #     try:
    #         from lib.sampling import get_rotated_mnist
    #     except ImportError:
    #         from sampling import get_rotated_mnist  # 如果 sampling.py 在同级目录
    #     print("🌀 Applying Rotated MNIST transformation ...")
    #     train_dataset, test_dataset = get_rotated_mnist(
    #             train_dataset,
    #             test_dataset,
    #             user_groups,
    #             user_groups_lt,
    #             args.num_users
    #     )
    # ====================================================
    # 🔹 Rotated MNIST（Static + Dynamic）
    # ====================================================
    if args.dataset == 'mnist_rotated':
        # —— 静态旋转 ——
        try:
            from lib.sampling import get_rotated_mnist
        except ImportError:
            from sampling import get_rotated_mnist
        print("🌀 Applying STATIC Rotated MNIST ...")
        train_dataset, test_dataset = get_rotated_mnist(
            train_dataset,
            test_dataset,
            user_groups,
            user_groups_lt,
            args.num_users
            # round,              
            # delta=15
        )

    elif args.dataset == 'mnist_rotated_dynamic':
        # —— 动态旋转（每轮概念漂移） ——
        print("🌀 Dynamic Rotated MNIST enabled (semantic drift each round).")
        # 动态旋转将在训练循环中执行，这里不进行旋转


    # ACMP-FL: 预先统计每个客户端的本地类别频率 f_{i,c}
    # client_class_freq = get_client_class_freq(args, train_dataset, user_groups)

    

    # =======================================
    # Build models for all users
    # =======================================
    local_model_list = []
    for i in range(args.num_users):

        # ---- MNIST / Rotated MNIST ----
        if args.dataset in ['mnist', 'mnist_rotated', 'mnist_rotated_dynamic']:
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNMnist(args=args)

        # ---- FEMNIST ----
        elif args.dataset == 'femnist':
            if args.mode == 'model_heter':
                if i < 7:
                    args.out_channels = 18
                elif i >= 7 and i < 14:
                    args.out_channels = 20
                else:
                    args.out_channels = 22
            else:
                args.out_channels = 20
            local_model = CNNFemnist(args=args)

        # ---- CIFAR10 / CIFAR100 ----
        elif args.dataset in ['cifar10', 'cifar100']:
            if args.mode == 'model_heter':
                if i < 10:
                    args.stride = [1, 4]
                else:
                    args.stride = [2, 2]
            else:
                args.stride = [2, 2]
            resnet = resnet18(args, pretrained=True, num_classes=1000)

            # Step 2：替换 FC 层（1000 → CIFAR10 ）
            resnet.fc = torch.nn.Linear(resnet.fc.in_features, args.num_classes)

            # Step 3：初始化新 FC（必须，否则FC部分随机）
            torch.nn.init.xavier_uniform_(resnet.fc.weight)
            torch.nn.init.zeros_(resnet.fc.bias)

            local_model = resnet

        # ✅ 所有数据集通用的后处理
        local_model.to(args.device)
        local_model.train()
        local_model_list.append(local_model)

    # local_model_list = []
    # for i in range(args.num_users):

    #     # ---- MNIST ----
    #     if args.dataset in ['mnist', 'mnist_rotated', 'mnist_rotated_dynamic']:
    #         args.out_channels = 20
    #         local_model = CNNMnist(args=args)

    #     # ---- FEMNIST ----
    #     elif args.dataset == 'femnist':
    #         args.out_channels = 20
    #         local_model = CNNFemnist(args=args)

    #     # ---- CIFAR10 / CIFAR100 ----
    #     elif args.dataset in ['cifar10', 'cifar100']:

    #         # ★★★ 区分 task_heter / model_heter ★★★
    #         if args.mode == 'model_heter':
    #             if i < 10:
    #                 args.stride = [1, 4]
    #             else:
    #                 args.stride = [2, 2]
    #         else:
    #             # task_heter → 所有客户端相同模型（推荐 stride=[2,2]）
    #             args.stride = [2, 2]

    #         # 统一使用 ResNet18
    #         resnet = resnet18(args, pretrained=True, num_classes=1000)

    #         # 替换 FC 头
    #         resnet.fc = torch.nn.Linear(resnet.fc.in_features, args.num_classes)
    #         torch.nn.init.xavier_uniform_(resnet.fc.weight)
    #         torch.nn.init.zeros_(resnet.fc.bias)

    #         local_model = resnet

    #     # -------- finalize --------
    #     local_model.to(args.device)
    #     local_model.train()
    #     local_model_list.append(local_model)


        # ===== 验证预训练 ResNet18 是否正确加载 =====
        if i == 0:   # 只检查第一个 client 的模型就够了
            conv1_weight = local_model.conv1.weight.detach().cpu()
            print(">>> Pretrained Check: conv1 weight mean = {:.6f}, std = {:.6f}".format(
                conv1_weight.mean().item(),
                conv1_weight.std().item()
            ))

        # ======================================================
        # ✅ DEBUG: 训练前检查每个客户端模型是否“异构”
        # ======================================================
        import torch
        from torch.utils.data import DataLoader
        
        def inspect_model_heter(local_model_list, args):
            print("\n================ Model Heterogeneity Check ================")
        
            # 造一个 dummy 输入（不依赖 DataLoader，更稳）
            if args.dataset in ['mnist', 'mnist_rotated', 'mnist_rotated_dynamic', 'femnist']:
                x = torch.randn(2, 1, 28, 28).to(args.device)
            else:  # cifar10 / cifar100
                x = torch.randn(2, 3, 32, 32).to(args.device)
        
            for i, m in enumerate(local_model_list):
                m = m.to(args.device)
                m.eval()
        
                n_params = sum(p.numel() for p in m.parameters())
        
                # 找最后一个 Linear 层（大概率是分类头）
                last_linear = None
                for name, module in m.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        last_linear = (name, module.in_features, module.out_features)
        
                # 尝试 forward 看 logits shape（兼容 forward 返回 tuple/list）
                try:
                    with torch.no_grad():
                        out = m(x)
                    if isinstance(out, (tuple, list)):
                        logits = out[0]
                    else:
                        logits = out
                    logits_shape = tuple(logits.shape)
                except Exception as e:
                    logits_shape = f"ForwardError: {type(e).__name__}: {e}"
        
                print(f"[Client {i:02d}] params={n_params:,}  last_linear={last_linear}  logits={logits_shape}")
        
            print("===========================================================\n")
        
        inspect_model_heter(local_model_list, args)
        

    if args.mode == 'task_heter':
        FedProto_taskheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)
    else:
        FedProto_modelheter(args, train_dataset, test_dataset, user_groups, user_groups_lt, local_model_list, classes_list)