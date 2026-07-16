import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")
import torch.nn as nn
import pandas as pd
import datetime
import argparse
import os, random, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_curve, confusion_matrix, cohen_kappa_score,
                             accuracy_score, roc_auc_score, precision_score,
                             recall_score, balanced_accuracy_score, f1_score,
                             precision_recall_curve, auc)
from dataprocess.creat_data_DC import  creat_data
from dataprocess.utils_test import *
from frame.model import SynergyPredictionModel
import warnings
warnings.filterwarnings("ignore")


# ---



def set_seed(seed: int = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset


def split_dataset(dataset, ratio):
    n = int(len(dataset) * ratio)
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2


# ---


# ---
class EarlyStopper:
    def __init__(self, patience: int = 20, mode: str = 'max', delta: float = 0.0):
        self.patience = patience
        self.counter = 0
        self.best = None
        self.mode = mode
        self.delta = delta
        self.stop = False

    def __call__(self, metric):
        if self.best is None:
            self.best = metric
            return False
        if (metric > self.best + self.delta) if self.mode == 'max' else (metric < self.best - self.delta):
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop
# ---


# ---

def train(model, device, drug1_loader_train, drug2_loader_train, optimizer, loss_fn, epoch, train_batch_size,
          log_interval):
    model.train()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()
    lambda_sym = 0.01

    zipped = zip(drug1_loader_train, drug2_loader_train)
    for batch_idx, data in enumerate(zipped):
        data1 = data[0].to(device)
        data2 = data[1].to(device)
        y = data[0].y.view(-1, 1).float().to(device).squeeze(1)

        optimizer.zero_grad()
        output= model(data1, data2)
        loss = loss_fn(output, y)

        loss.backward()

        # total_norm = 0
        # if batch_idx % log_interval == 0:
        #     for p in model.parameters():
        #         if p.grad is not None:
        #             param_norm = p.grad.dataprocess.norm(2)
        #             total_norm += param_norm.item() ** 2
        #     total_norm = total_norm ** (1. / 2)


        # torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

        optimizer.step()

        if batch_idx % log_interval == 0:
            processed_samples = min((batch_idx + 1) * train_batch_size, len(drug1_loader_train.dataset))
            dataset_size = len(drug1_loader_train.dataset)
            progress = 100. * (batch_idx + 1) / len(drug1_loader_train)
            print(
                f'Train epoch: {epoch} [{processed_samples}/{dataset_size} ({progress:.0f}%)]\tLoss: {loss.item():.6f}')

        ys = output.to('cpu').data.numpy()
        predicted_labels = list(map(lambda x: int(x > 0.5), ys))
        total_preds = torch.cat((total_preds, torch.Tensor(ys)), 0)
        total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
        total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def predicting(model, device, drug1_loader_test, drug2_loader_test):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_prelabels = torch.Tensor()

    print(f'Predicting on {len(drug1_loader_test.dataset)} samples...')
    with torch.no_grad():
        for data in zip(drug1_loader_test, drug2_loader_test):
            data1 = data[0].to(device)
            data2 = data[1].to(device)
            output = model(data1, data2)

            ys = output.to('cpu').data.numpy()
            predicted_labels = list(map(lambda x: int(x > 0.5), ys))
            total_preds = torch.cat((total_preds, torch.Tensor(ys)), 0)
            total_prelabels = torch.cat((total_prelabels, torch.Tensor(predicted_labels)), 0)
            total_labels = torch.cat((total_labels, data1.y.view(-1, 1).cpu()), 0)

    return total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_prelabels.numpy().flatten()


def save_AUCs(AUCs, file_name):
    with open(file_name, 'a') as f:
        if isinstance(AUCs, list):
            f.write('\t'.join(map(str, AUCs)) + '\n')
        else:
            f.write(str(AUCs) + '\n')


def calculate_metrics_original(T, S, Y):
    #
    try:
        AUC = roc_auc_score(T, S)
    except ValueError:
        AUC = 0.5
    precision, recall, _ = precision_recall_curve(T, S)
    PR_AUC = auc(recall, precision)
    ACC = accuracy_score(T, Y)
    BACC = balanced_accuracy_score(T, Y)
    PREC = precision_score(T, Y, zero_division=0)
    RECALL = recall_score(T, Y, zero_division=0)
    F1 = f1_score(T, Y, zero_division=0)
    KAPPA = cohen_kappa_score(T, Y)
    try:
        tn, fp, fn, tp = confusion_matrix(T, Y).ravel()
        TPR = tp / (tp + fn)
    except ValueError:
        TPR = 0.0
    return AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL, F1




if __name__ == '__main__':

    # --- 1. ---
    config = {
        "seed": 1,
        "device": "cuda:0" if torch.cuda.is_available() else "cpu",
        "result_name": "DB_model1",
        "modeling": SynergyPredictionModel,

        # 数据路径
        # drgucombdb
        "cellfile": "data/drugcombdb/dbcellfeature.csv",
        "drug_smiles_file": "data/drugcombdb/dbdrugsmiles.csv",
        "datafile": "data/drugcombdb/drugcombdb.csv",
        "dataset_name": "drugcombdb",
        # merck
        # "cellfile": "data/merck/cell_features.csv",
        # "drug_smiles_file": "data/merck/smiles.csv",
        # "datafile": "data/merck/merck.csv",
        # "dataset_name": "merck",


        # 训练参数
        "TRAIN_BATCH_SIZE": 1024,
        "TEST_BATCH_SIZE":1024,
        "NUM_EPOCHS": 200,
        "LOG_INTERVAL": 5,
        "EARLY_STOP_PATIENCE": 30,

        # 优化器与调度器参数
        "INITIAL_LR": 0.0005,
        "WEIGHT_DECAY": 1e-4,
        "WARMUP_EPOCHS": 0,
    }
    #
    METRIC_HEADERS = ["AUC_dev", "PR_AUC", "ACC", "BACC", "PREC", "TPR", "KAPPA", "RECALL", "F1"]

    # --- 2.
    set_seed(config['seed'])
    device = torch.device(config['device'])
    print(f"Using device: {device}")

    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d-%H-%M-%S")

    folder_path = f'./result/{config["result_name"]}'
    os.makedirs(folder_path, exist_ok=True)
    print(f"Results will be saved in: {folder_path}")

    # --- 3.
    print('开始处理源文件....')
    drug1, drug2, cell, label, smile_graph, cell_features = creat_data(
        config['datafile'], config['drug_smiles_file'], config['cellfile']
    )
    print('从源文件提取特征成功！')

    print('载入数据...')
    drug1_data = TestbedDataset(dataset=config['dataset_name'] + '_drug1', xd=drug1, xt=cell, y=label,
                                smile_graph=smile_graph, xt_featrue=cell_features)
    drug2_data = TestbedDataset(dataset=config['dataset_name'] + '_drug2', xd=drug2, xt=cell, y=label,
                                smile_graph=smile_graph, xt_featrue=cell_features)
    print('载入数据完成！')


    lenth = len(drug1_data)
    pot = int(lenth / 5)
    print(f'Total dataprocess length: {lenth}, Fold size (pot): {pot}')

    rng = random.Random(config['seed'])
    random_num = rng.sample(range(lenth), lenth)

    all_folds_best_auc_metrics = []
    all_folds_best_auc_lines = []
    header_for_summary = ('Fold\tEpoch\t' + '\t'.join(METRIC_HEADERS))


    summary_log_file = os.path.join(folder_path, f'summary_5-fold-db1-k=128-seed:1_{time_str}.txt')
    with open(summary_log_file, 'w') as f:
        f.write(f"--- 5-Fold Cross-Validation Summary ---\n")
        f.write(f"Model: {config['modeling'].__name__}\n")
        f.write(f"Result Name: {config['result_name']}\n")
        f.write(f"Timestamp: {time_str}\n\n")
        f.write("--- Best AUC Epoch Details per Fold ---\n")
        f.write(header_for_summary + "\n")

    for i in range(5):
        print("\n" + "=" * 30)
        print(f"--- Starting Fold {i + 1}/5 ---")
        print("=" * 30)

        # 1.
        test_num = random_num[pot * i: pot * (i + 1)]
        train_num = random_num[:pot * i] + random_num[pot * (i + 1):]
        drug1_data_train, drug1_data_test = drug1_data[train_num], drug1_data[test_num]
        drug2_data_train, drug2_data_test = drug2_data[train_num], drug2_data[test_num]

        # 2. DataLoaders
        drug1_loader_train = DataLoader(drug1_data_train, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=None)
        drug1_loader_test = DataLoader(drug1_data_test, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=None)
        drug2_loader_train = DataLoader(drug2_data_train, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=None)
        drug2_loader_test = DataLoader(drug2_data_test, batch_size=config['TRAIN_BATCH_SIZE'], shuffle=None)

        # 3.
        model = config['modeling']().to(device)
        loss_fn = nn.BCELoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['INITIAL_LR'], weight_decay=config['WEIGHT_DECAY'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=config['NUM_EPOCHS'] - config['WARMUP_EPOCHS'],
                                                               eta_min=0)

        early_stopper = EarlyStopper(patience=config['EARLY_STOP_PATIENCE'])

        file_AUCs = os.path.join(folder_path,
                                 f'Model-k=128 seed:1_DB1_{i + 1}--_{time_str}.txt')
        header = ('Epoch\t' + '\t'.join(METRIC_HEADERS))
        with open(file_AUCs, 'w') as f:
            f.write(header + '\n')

        #
        best_auc_tracker = 0.0
        best_auc_epoch = -1
        best_auc_line = ""
        best_auc_metrics_list = []

        # 4.
        for epoch in range(1, config['NUM_EPOCHS'] + 1):

            if epoch <= config['WARMUP_EPOCHS'] and config['WARMUP_EPOCHS'] > 0:
                lr_scale = epoch / config['WARMUP_EPOCHS']
                for param_group in optimizer.param_groups:
                    param_group['lr'] = config['INITIAL_LR'] * lr_scale

            train_T, train_S, train_Y = train(
                model, device, drug1_loader_train, drug2_loader_train,
                optimizer, loss_fn, epoch,
                config['TRAIN_BATCH_SIZE'], config['LOG_INTERVAL']
            )

            if epoch > config['WARMUP_EPOCHS']:
                scheduler.step()

            T, S, Y = predicting(model, device, drug1_loader_test, drug2_loader_test)

            # ---  ---
            train_metrics_auc = roc_auc_score(train_T, train_S)
            train_metrics_acc = accuracy_score(train_T, train_Y)

            AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL, F1 = calculate_metrics_original(T, S, Y)
            current_test_metrics_values = [AUC, PR_AUC, ACC, BACC, PREC, TPR, KAPPA, RECALL, F1]

            #
            print(f"--- Epoch {epoch}/{config['NUM_EPOCHS']} ---")
            print(f"Train: AUC={train_metrics_auc:.4f}, ACC={train_metrics_acc:.4f}")
            print(f"Test:  AUC={AUC:.4f}, ACC={ACC:.4f}, PR_AUC={PR_AUC:.4f}")

            #
            metrics_values_with_epoch = [epoch] + current_test_metrics_values
            #
            metrics_str_list = [str(epoch)] + [f"{m:.6f}" for m in current_test_metrics_values]

            is_new_best_auc = AUC > best_auc_tracker

            #
            if is_new_best_auc:
                save_AUCs(metrics_str_list, file_AUCs)
                best_auc_tracker = AUC
                best_auc_epoch = epoch
                best_auc_line = "\t".join(metrics_str_list)
                best_auc_metrics_list = metrics_values_with_epoch
                model_save_path = os.path.join(folder_path, f'best_model_fold_{i + 1}.pth')
                torch.save(model.state_dict(), model_save_path)

            if early_stopper(AUC):
                break

        # --- 5.---
        print(f"--- Fold {i + 1} Finished ---")
        print(f"Best Test AUC: {best_auc_tracker:.4f} at epoch {best_auc_epoch}")


        all_folds_best_auc_metrics.append(best_auc_metrics_list)

        #
        summary_line_list = [f"Fold {i + 1}"] + [f"{m:.6f}" if isinstance(m, float) else str(m) for m in
                                                 best_auc_metrics_list]
        all_folds_best_auc_lines.append("\t".join(summary_line_list))

        #
        with open(file_AUCs, 'a') as f:
            f.write("\n" + "=" * 20 + " Best Epoch Summary " + "=" * 20 + "\n")
            f.write(f"Best AUC Epoch [{best_auc_epoch}]:\n{best_auc_line}\n")

    # --- 6. ---
    print("\n" + "=" * 30)
    print("--- 5-Fold Cross-Validation Finished ---")

    with open(summary_log_file, 'a') as f:
        for line in all_folds_best_auc_lines:
            f.write(line + "\n")

    #

    metrics_array = np.array([metrics[1:] for metrics in all_folds_best_auc_metrics])
    mean_metrics = np.mean(metrics_array, axis=0)
    std_metrics = np.std(metrics_array, axis=0)

    #
    print("\n--- Overall Performance (Mean ± Std Dev) ---")
    for i, header in enumerate(METRIC_HEADERS):
        print(f"{header}: {mean_metrics[i]:.6f} ± {std_metrics[i]:.6f}")

    #
    with open(summary_log_file, 'a') as f:
        f.write("\n--- Overall Performance (Mean ± Std Dev) ---\n")
        f.write("Metric\tMean\tStd Dev\tReport\n")
        for i, header in enumerate(METRIC_HEADERS):
            mean_val = mean_metrics[i]
            std_val = std_metrics[i]
            report_str = f"{mean_val:.6f} ± {std_val:.6f}"
            f.write(f"{header}\t{mean_val:.6f}\t{std_val:.6f}\t{report_str}\n")

        f.write(f"\nReport as (AUC): {mean_metrics[0]:.6f} ± {std_metrics[0]:.6f}\n")

    print(f"Summary report saved to: {summary_log_file}")
    print("Training complete.")