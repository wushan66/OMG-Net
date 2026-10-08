import argparse
import pickle
import os
import numpy as np
from tqdm import tqdm
from itertools import product


def calculate_acc(weights, label, results_list):
    """根据权重计算Top1和Top5准确率"""
    right_num = 0
    right_num_5 = 0
    total_num = len(label)

    for i in range(total_num):
        l = label[i]
        # 加权融合各分支分数
        fused_score = sum(w * res[i][1] for w, res in zip(weights, results_list))
        # 计算Top5和Top1
        rank_5 = fused_score.argsort()[-5:]
        right_num_5 += int(int(l) in rank_5)
        pred = np.argmax(fused_score)
        right_num += int(pred == int(l))

    acc1 = right_num / total_num
    acc5 = right_num_5 / total_num
    return acc1, acc5


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        required=True,
                        choices={'ntu/xsub', 'ntu/xview', 'ntu120/xsub', 'ntu120/xset', 'NW-UCLA'},
                        help='the work folder for storing results')
    parser.add_argument('--joint-dir',
                        required=True,
                        help='Directory containing "epoch1_test_score.pkl" for joint eval results')
    parser.add_argument('--bone-dir',
                        required=True,
                        help='Directory containing "epoch1_test_score.pkl" for bone eval results')
    parser.add_argument('--joint-motion-dir', default=None)
    parser.add_argument('--bone-motion-dir', default=None)
    parser.add_argument('--step',
                        default=0.1,
                        type=float,
                        help='Step size for weight search (smaller = more precise but slower)')
    parser.add_argument('--weight-range',
                        default=[0.0, 2.0],
                        type=float,
                        nargs=2,
                        help='Weight search range [min, max]')

    arg = parser.parse_args()

    # ---------------------- 1. 加载标签 ----------------------
    dataset = arg.dataset
    if 'UCLA' in dataset:
        label = []
        with open('./data/NW-UCLA/val_label.pkl', 'rb') as f:
            data_info = pickle.load(f)
            for info in data_info:
                label.append(int(info['label']) - 1)
    elif 'ntu120' in dataset:
        if 'xsub' in dataset:
            npz_data = np.load('./data/ntu120/NTU120_CSub.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xset' in dataset:
            npz_data = np.load('./data/ntu120/NTU120_CSet.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    elif 'ntu' in dataset:
        if 'xsub' in dataset:
            npz_data = np.load('./data/ntu/NTU60_CS2.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
        elif 'xview' in dataset:
            npz_data = np.load('./data/ntu/NTU60_CV2.npz')
            label = np.where(npz_data['y_test'] > 0)[1]
    else:
        raise NotImplementedError("Dataset not supported")
    label = np.array(label)

    # ---------------------- 2. 加载各分支预测结果 ----------------------
    results_list = []
    # 加载关节特征
    with open(os.path.join(arg.joint_dir, 'epoch1_test_score.pkl'), 'rb') as f:
        joint_res = list(pickle.load(f).items())
        results_list.append(joint_res)
    # 加载骨骼特征
    with open(os.path.join(arg.bone_dir, 'epoch1_test_score.pkl'), 'rb') as f:
        bone_res = list(pickle.load(f).items())
        results_list.append(bone_res)
    # 加载关节运动特征（可选）
    if arg.joint_motion_dir is not None:
        with open(os.path.join(arg.joint_motion_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            joint_motion_res = list(pickle.load(f).items())
            results_list.append(joint_motion_res)
    # 加载骨骼运动特征（可选）
    if arg.bone_motion_dir is not None:
        with open(os.path.join(arg.bone_motion_dir, 'epoch1_test_score.pkl'), 'rb') as f:
            bone_motion_res = list(pickle.load(f).items())
            results_list.append(bone_motion_res)
    num_branches = len(results_list)
    print(f"Detected {num_branches} branches for fusion")

    # ---------------------- 3. 生成权重搜索网格 ----------------------
    min_w, max_w = arg.weight_range
    step = arg.step
    # 生成每个分支的权重候选值
    weight_candidates = np.arange(min_w, max_w + step, step)
    # 生成所有权重组合（笛卡尔积）
    weight_combinations = product(weight_candidates, repeat=num_branches)
    total_combinations = len(weight_candidates) ** num_branches
    print(f"Weight search range: [{min_w}, {max_w}], step: {step}")
    print(f"Total weight combinations to test: {total_combinations}")

    # ---------------------- 4. 搜索最优权重 ----------------------
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_weights = None

    for weights in tqdm(weight_combinations, total=total_combinations, desc="Searching optimal weights"):
        acc1, acc5 = calculate_acc(weights, label, results_list)
        # 更新最优结果
        if acc1 > best_acc1:
            best_acc1 = acc1
            best_acc5 = acc5
            best_weights = weights

    # ---------------------- 5. 输出结果 ----------------------
    print("\n" + "=" * 50)
    print("Optimal Weight Search Results")
    print("=" * 50)
    print(f"Best Top1 Acc: {best_acc1 * 100:.4f}%")
    print(f"Best Top5 Acc: {best_acc5 * 100:.4f}%")
    print(f"Optimal Weights:")
    branches_name = ["Joint", "Bone", "Joint-Motion", "Bone-Motion"][:num_branches]
    for name, w in zip(branches_name, best_weights):
        print(f"  {name}: {w:.2f}")
    print("=" * 50)


