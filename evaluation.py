import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from main import TSFool


if __name__ == '__main__':
    from models.models_structure.PenDigits import RNN
    model = torch.load('models/PenDigits.pkl')
    dataset_name = 'PenDigits'
    X = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_X.npy')
    Y = np.load(f'datasets/preprocessed/{dataset_name}/{dataset_name}_TEST_Y.npy')
    time_start = time.time()
    adv_X, adv_Y, target_X = TSFool(model, X, Y, K=2, T=10, F=1, eps=0.1, N=20, P=0.9, C=1, target=-1, details=False)
    time_end = time.time()
    np.save(f'datasets/adversarial/{dataset_name}/{dataset_name}_TEST_ADV_X.npy', adv_X)
    np.save(f'datasets/adversarial/{dataset_name}/{dataset_name}_TEST_ADV_Y.npy', adv_Y)

    # original accuracy
    X_torch = torch.from_numpy(X).to(torch.float32)
    output, _ = model(X_torch)
    model_pred_y = torch.max(output, 1)[1].data.numpy()
    model_accuracy = float((model_pred_y == Y).astype(int).sum()) / float(Y.size)

    # attacked accuracy
    adv_X_torch = torch.from_numpy(adv_X).to(torch.float32)
    adv_output, _ = model(adv_X_torch)
    adv_model_pred_y = torch.max(adv_output, 1)[1].data.numpy()
    adv_model_accuracy = float((adv_model_pred_y == adv_Y).astype(int).sum()) / float(adv_Y.size)

    # average perturbation amount (percentage)
    l_1_distance = 0
    l_2_distance = 0
    l_inf_distance = 0
    group_size = int(adv_X.shape[0] / target_X.shape[0])
    for i in range(target_X.shape[0]):
        current_target_x = target_X[i]
        for j in range(group_size):
            current_adv_x = adv_X[i * group_size + j]
            x_perturbation = abs(current_target_x - current_adv_x)
            for k in range(target_X.shape[1]):
                l_1_distance += np.sum(x_perturbation[k])
                l_2_distance += np.sqrt(np.sum(x_perturbation[k] ** 2))
                l_inf_distance += np.max(x_perturbation[k])
    l_1_perturbation_percentage = 100 * l_1_distance / (adv_X.shape[0] * adv_X.shape[1] * adv_X.shape[2])
    l_2_perturbation_percentage = 100 * l_2_distance / (adv_X.shape[0] * adv_X.shape[1] * np.sqrt(np.sum([1 for i in range(adv_X.shape[2])])))
    l_inf_perturbation_percentage = 100 * l_inf_distance / (adv_X.shape[0] * adv_X.shape[1])

    # average camouflage coefficient
    classes_mean = []
    for i in range(int(np.max(Y) + 1)):
        current_sample_number = 0
        current_sample_sum = np.zeros((X.shape[1], X.shape[2]))
        for j in range(X.shape[0]):
            if Y[j] == i:
                current_sample_number += 1
                current_sample_sum += X[j]
        current_sample_mean = current_sample_sum / current_sample_number
        classes_mean.append(current_sample_mean)

    classes_distance_mean = []
    for i in range(int(np.max(Y) + 1)):
        current_sample_number = 0
        current_classes_distance_sum = 0
        current_classes_mean = classes_mean[i]
        for j in range(X.shape[0]):
            if Y[j] == i:
                current_sample_number += 1
                for k in range(X.shape[1]):
                    current_classes_distance_sum += np.sqrt(np.sum(abs(X[j][k] - current_classes_mean[k]) ** 2))
        current_classes_distance_mean = current_classes_distance_sum / current_sample_number
        classes_distance_mean.append(current_classes_distance_mean)

    camouflage_coefficient_count = 0
    sum_camouflage_coefficient = 0
    for i in range(adv_X.shape[0]):
        if adv_model_pred_y[i] != adv_Y[i]:
            camouflage_coefficient_count += 1
            current_classes_distance_original = classes_distance_mean[int(adv_Y[i])]
            current_classes_distance_adv = classes_distance_mean[int(adv_model_pred_y[i])]
            current_classes_mean_original = classes_mean[int(adv_Y[i])]
            current_classes_mean_adv = classes_mean[int(adv_model_pred_y[i])]
            current_distance_original = 0
            current_distance_adv = 0
            for j in range(adv_X.shape[1]):
                current_distance_original += np.sqrt(np.sum(abs(adv_X[i][j] - current_classes_mean_original[j]) ** 2))
                current_distance_adv += np.sqrt(np.sum(abs(adv_X[i][j] - current_classes_mean_adv[j]) ** 2))
            sum_camouflage_coefficient += (current_distance_original / current_classes_distance_original) / \
                                          (current_distance_adv / current_classes_distance_adv)
    avg_camouflage_coefficient = sum_camouflage_coefficient / camouflage_coefficient_count

    print('\nTSFool Attack:')
    print(f' - Dataset: UEA-{dataset_name}')
    print(' - Original Model Accuracy: %.4f' % model_accuracy)
    print(' - Attack Success Rate: %.2f%%' % (100 * (1 - adv_model_accuracy)))
    print(f' - The Number of Generated Adversarial Samples: {len(adv_X)}')
    print(' - Average Time Cost (s per sample): %.4f' % ((time_end - time_start) / len(adv_X)))
    print(' - Average Perturbation Amount (L1 / L2 / Linf): %.2f%% / %.2f%% / %.2f%%'
          % (l_1_perturbation_percentage, l_2_perturbation_percentage, l_inf_perturbation_percentage))
    print(' - Average Camouflage Coefficient: %.4f' % avg_camouflage_coefficient)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    timestep_record = [i for i in range(1, X.shape[1] + 1)]
    plt_sample_idx = 0
    plt_sample_idx_adv = plt_sample_idx * group_size

    proj_2 = [1.05] * adv_X.shape[1]
    ax.plot(timestep_record, proj_2, target_X[plt_sample_idx].copy()[:, 1], color='#1f77b4')
    ax.plot(timestep_record, proj_2, adv_X[plt_sample_idx_adv].copy()[:, 1], color='red')
    proj_3 = [-0.05] * adv_X.shape[1]
    ax.plot(timestep_record, target_X[plt_sample_idx].copy()[:, 0], proj_3, color='#1f77b4')
    ax.plot(timestep_record, adv_X[plt_sample_idx_adv].copy()[:, 0], proj_3, color='red')
    ax.plot3D(timestep_record, target_X[plt_sample_idx].copy()[:, 0], target_X[plt_sample_idx].copy()[:, 1], '-',
              linewidth=3, alpha=0.55, color='#1f77b4', label='Original')
    ax.plot3D(timestep_record, adv_X[plt_sample_idx_adv].copy()[:, 0], adv_X[plt_sample_idx_adv].copy()[:, 1], '-',
              linewidth=3, alpha=0.55, color='red', label='Attacked')

    plt.title("Multi-TSFool")
    ax.set_zlabel("feature 2")
    ax.set_ylabel("feature 1")
    ax.set_xlabel("time step")
    ax.set_zlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_box_aspect((2, 1, 1))
    plt.legend()
    plt.savefig(f"figures/{dataset_name}.png", bbox_inches='tight', pad_inches=0.1, dpi=600)
    plt.show()
