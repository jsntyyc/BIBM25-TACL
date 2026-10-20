import gc
from collections import defaultdict
from pathlib import Path
from tempfile import mkdtemp
from typing import Optional, Dict
from collections import deque
import pandas as pd
import torch
from omegaconf import OmegaConf
from rich.pretty import  pprint
from rich.progress import track
from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data import ConcatDataset
from tasks.icd.src.data.datatypes import Data, Lookups
from tasks.icd.src.metrics import  MetricCollection
from tasks.icd.src.models import BaseModel
from tasks.icd.src.settings import ID_COLUMN, TARGET_COLUMN, create_run_directory
from tasks.icd.src.trainer.callbacks import BaseCallback
from tasks.icd.src.utils.decision_boundary import f1_score_db_tuning


class Trainer:
    def __init__(
        self,
        config: OmegaConf,
        data: Data,
        model: BaseModel,
        optimizer: torch.optim.Optimizer,
        dataloaders: dict[str, DataLoader],
        metric_collections: dict[str, dict[str, MetricCollection]],
        callbacks: BaseCallback,
        lr_scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        lookups: Optional[Lookups] = None,
        accumulate_grad_batches: int = 1,
    ) -> None:
        self.config = config
        self.data = data
        self.model = model
        self.optimizer = optimizer
        self.dataloaders = dataloaders
        self.callbacks = callbacks
        self.device = "cpu"
        self.metric_collections = metric_collections
        self.lr_scheduler = lr_scheduler
        self.lookups = lookups
        self.accumulate_grad_batches = accumulate_grad_batches
        pprint(f"Accumulating gradients over {self.accumulate_grad_batches} batch(es).")
        # 设置初始训练数据为 train_easy
        self.dataloaders["train"] = self.dataloaders["train_easy"]
        print(f"Initialized training with {len(self.dataloaders['train'].dataset)} examples (easy)")
        self.validate_on_training_data = config.trainer.validate_on_training_data
        self.print_metrics = config.trainer.print_metrics
        self.epochs = config.trainer.epochs
        self.epoch = 0
        self.use_amp = config.trainer.use_amp
        self.threshold_tuning = config.trainer.threshold_tuning
        self.gradient_scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.experiment_path = create_run_directory(getattr(config, "experiment_dir", None))
        self.current_val_results = None
        self.stop_training = False
        self.best_db = 0.5
        self.on_initialisation_end()

    # def mix_dataloaders(*dataloaders, ratios=None):
    #     """
    #     合并多个 DataLoader，并按给定的比例（ratios）进行混合。
    #     :param dataloaders: 传入的多个 DataLoader
    #     :param ratios: 数据集的混合比例，默认为均匀混合
    #     :return: 合并后的 DataLoader
    #     """
    #
    #     # 如果没有指定比例，则默认为均匀混合
    #     if ratios is None:
    #         ratios = [1.0] * len(dataloaders)
    #
    #     # 确保比例总和为 1
    #     assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios should sum to 1"
    #
    #     # 获取每个数据集的大小
    #     dataset_sizes = [len(dl.dataset) for dl in dataloaders]
    #
    #     # 根据比例调整每个数据集的大小
    #     weighted_datasets = []
    #     for i, dl in enumerate(dataloaders):
    #         subset_size = int(ratios[i] * sum(dataset_sizes))  # 计算每个数据集应占的样本数
    #         subset = torch.utils.data.Subset(dl.dataset, range(subset_size))  # 根据大小选择子集
    #         weighted_datasets.append(subset)
    #
    #     # 合并所有子数据集
    #     combined_dataset = ConcatDataset(weighted_datasets)
    #
    #     # 创建新的 DataLoader
    #     mixed_dataloader = DataLoader(combined_dataset, batch_size=dataloaders[0].batch_size, shuffle=True)
    #
    #     return mixed_dataloader
    #
    # def concat_dataloaders(*dataloaders):
    #     """
    #     拼接多个 DataLoader，将其数据合并成一个新的 DataLoader。
    #     :param dataloaders: 传入的多个 DataLoader
    #     :return: 拼接后的 DataLoader
    #     """
    #     # 使用 ConcatDataset 来拼接多个数据集
    #     combined_dataset = ConcatDataset([dl.dataset for dl in dataloaders])
    #
    #     # 创建新的 DataLoader
    #     combined_dataloader = DataLoader(combined_dataset, batch_size=dataloaders[0].batch_size, shuffle=True)
    #
    #     return combined_dataloader


    # def fit(self) -> None:#3
    #     """Train and validate the model."""
    #     try:
    #         self.on_fit_begin()
    #
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集
    #
    #             #easy-middle-hard
    #             if self.epoch <= 5:
    #                 self.dataloaders["train"] = self.dataloaders["train_easy"]
    #                 num_train_examples = len(self.dataloaders["train"].dataset)
    #                 print(f"Training with {num_train_examples} examples (easy)")
    #             elif 5 < self.epoch <= 10:
    #                 self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                 num_train_examples = len(self.dataloaders["train"].dataset)
    #                 print(f"Training with {num_train_examples} examples (middle)")
    #             else:
    #                 self.dataloaders["train"] = self.dataloaders["train_hard"]
    #                 num_train_examples = len(self.dataloaders["train"].dataset)
    #                 print(f"Training with {num_train_examples} examples (hard)")
    #
    #             # if self.epoch < self.epochs * 0.25:
    #             #     self.dataloaders["train"] = self.dataloaders["train_hard"]
    #             #     num_train_examples = len(self.dataloaders["train"].dataset)
    #             #     print(f"Training with {num_train_examples} examples (hard)")
    #             # elif self.epoch < self.epochs * 0.50:
    #             #     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #             #     num_train_examples = len(self.dataloaders["train"].dataset)
    #             #     print(f"Training with {num_train_examples} examples (middle)")
    #             # else:
    #             #     self.dataloaders["train"] = self.dataloaders["train_easy"]
    #             #     num_train_examples = len(self.dataloaders["train"].dataset)
    #             #     print(f"Training with {num_train_examples} examples (easy)")
    #
    #
    #             self.train_one_epoch(self.epoch)
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #             self.val(self.epoch, "val")
    #             self.on_epoch_end()
    #             self.epoch += 1
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    def fit(self) -> None:
        """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
        try:
            self.on_fit_begin()
            f1_scores = deque(maxlen=3)  # 保存最近 3 个 epoch 的 F1 分数
            self.dataloaders["train"] = self.dataloaders["train_easy"]
            easy_to_middle_threshold = 0.1  # middle 到 hard 的平均增长阈值(可以作为超参数)
            middle_to_hard_threshold = 0.05 # middle 到 hard 的平均增长阈值(可以作为超参数)

            hard_to_middle_threshold = 0.05#(可以作为超参数)
            middle_to_easy_threshold = 0.1#(可以作为超参数)
            for _ in range(self.epoch, self.epochs):
                if self.stop_training:
                    break
                self.on_epoch_begin()

                # 动态切换训练数据集(easy->middle->hard)
                if self.dataloaders["train"] != self.dataloaders["train_hard"] and len(f1_scores) > 1:
                    # 计算最近几个 epoch 的平均增长率
                    f1_growth_rates = [
                        f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
                    ]
                    avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
                    print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")

                    # 根据增长率切换数据集
                    if (
                            self.dataloaders["train"] == self.dataloaders["train_easy"]
                            and avg_growth_rate < easy_to_middle_threshold
                    ):
                        self.dataloaders["train"] = self.dataloaders["train_middle"]
                        print("Switched to middle difficulty data")
                    elif (
                            self.dataloaders["train"] == self.dataloaders["train_middle"]
                            and avg_growth_rate < middle_to_hard_threshold
                    ):
                        self.dataloaders["train"] = self.dataloaders["train_hard"]
                        print("Switched to hard difficulty data")

                # 动态切换训练数据集(hard->middle->easy)
                # if self.dataloaders["train"] != self.dataloaders["train_easy"] and len(f1_scores) > 1:
                #     # 计算最近几个 epoch 的平均增长率
                #     f1_growth_rates = [
                #         f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
                #     ]
                #     avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
                #     print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
                #
                #     # 根据增长率切换数据集
                #     if (
                #             self.dataloaders["train"] == self.dataloaders["train_hard"]
                #             and avg_growth_rate < hard_to_middle_threshold
                #     ):
                #         self.dataloaders["train"] = self.dataloaders["train_middle"]
                #         print("Switched to middle difficulty data")
                #     elif (
                #             self.dataloaders["train"] == self.dataloaders["train_middle"]
                #             and avg_growth_rate < middle_to_easy_threshold
                #     ):
                #         self.dataloaders["train"] = self.dataloaders["train_easy"]
                #         print("Switched to easy difficulty data")

                self.train_one_epoch(self.epoch)

                if self.validate_on_training_data:
                    self.train_val(self.epoch, "train_val")

                # 验证并记录 F1 分数
                val_results = self.val(self.epoch, "val")
                f1_score = val_results["val"]["all"]["f1_micro"].item()
                f1_scores.append(f1_score)
                print(f"Validation F1 score: {f1_score}")

                self.on_epoch_end()
                self.epoch += 1

            self.on_fit_end()
            self.val(self.epoch, "val", evaluating_best_model=True)
            self.val(self.epoch, "test", evaluating_best_model=True)
            self.save_final_model()
        except KeyboardInterrupt:
            pprint("Training interrupted by user. Stopping training")
        del self.model
        gc.collect()
        torch.cuda.empty_cache()
        self.on_end()

    # def fit(self) -> None: #2
    #     """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
    #     try:
    #         self.on_fit_begin()
    #         f1_scores = deque(maxlen=3)  # 保存最近3个epoch的F1分数
    #         self.dataloaders["train"] = self.dataloaders["train_easy"]
    #         # self.dataloaders["train"] = self.dataloaders["train_middle"]
    #         easy_to_middle_threshold = 0.1  # easy到middle的平均增长阈值(可以作为超参数)
    #         middle_to_easy_threshold = 0.1
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集(easy -> middle)
    #             if len(f1_scores) > 1:
    #                 # 计算最近几个epoch的平均增长率
    #                 f1_growth_rates = [
    #                     f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #                 ]
    #                 avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #                 print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #
    #                 # 根据增长率切换数据集
    #                 if (
    #                         self.dataloaders["train"] == self.dataloaders["train_easy"]
    #                         and avg_growth_rate < easy_to_middle_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                     print("Switched to middle difficulty data")
    #
    #             # 动态切换训练数据集(middle -> easy)
    #             # if len(f1_scores) > 1:
    #             #     # 计算最近几个epoch的平均增长率
    #             #     f1_growth_rates = [
    #             #         f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #             #     ]
    #             #     avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #             #     print(
    #             #         f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #             #
    #             #     # 根据增长率切换数据集
    #             #     if (
    #             #             self.dataloaders["train"] == self.dataloaders["train_middle"]
    #             #             and avg_growth_rate < middle_to_easy_threshold
    #             #     ):
    #             #         self.dataloaders["train"] = self.dataloaders["train_easy"]
    #             #         print("Switched to easy difficulty data")
    #
    #             self.train_one_epoch(self.epoch)
    #
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #
    #             # 验证并记录F1分数
    #             val_results = self.val(self.epoch, "val")
    #             f1_score = val_results["val"]["all"]["f1_micro"].item()
    #             f1_scores.append(f1_score)
    #             print(f"Validation F1 score: {f1_score}")
    #
    #             self.on_epoch_end()
    #             self.epoch += 1
    #
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    # def fit(self) -> None:#4正
    #     """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
    #     try:
    #         self.on_fit_begin()
    #         f1_scores = deque(maxlen=3)  # 保存最近3个epoch的F1分数
    #         self.dataloaders["train"] = self.dataloaders["train_easy"]
    #         # self.dataloaders["train"] = self.dataloaders["train_hard_hard"]
    #         # 定义阈值(可作为超参数调整)
    #         easy_to_middle_threshold = 0.1
    #         middle_to_hard_threshold = 0.05
    #         hard_to_hard_hard_threshold = 0.02
    #
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集
    #             if len(f1_scores) > 1:
    #                 # 计算最近几个epoch的平均增长率
    #                 f1_growth_rates = [
    #                     f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #                 ]
    #                 avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #                 print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #
    #                 # 根据增长率切换数据集
    #                 if (
    #                         self.dataloaders["train"] == self.dataloaders["train_easy"]
    #                         and avg_growth_rate < easy_to_middle_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                     print("Switched to middle difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_middle"]
    #                         and avg_growth_rate < middle_to_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard"]
    #                     print("Switched to hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard"]
    #                         and avg_growth_rate < hard_to_hard_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard_hard"]
    #                     print("Switched to very hard difficulty data")
    #
    #             self.train_one_epoch(self.epoch)
    #
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #
    #             # 验证并记录F1分数
    #             val_results = self.val(self.epoch, "val")
    #             f1_score = val_results["val"]["all"]["f1_micro"].item()
    #             f1_scores.append(f1_score)
    #             print(f"Validation F1 score: {f1_score}")
    #
    #             self.on_epoch_end()
    #             self.epoch += 1
    #
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    # def fit(self) -> None:#4逆
    #     """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
    #     try:
    #         self.on_fit_begin()
    #         f1_scores = deque(maxlen=3)  # 保存最近3个epoch的F1分数
    #         self.dataloaders["train"] = self.dataloaders["train_hard_hard"]
    #
    #         # 定义阈值(可作为超参数调整)
    #         hard_hard_to_hard_threshold = 0.02
    #         hard_to_middle_threshold = 0.05
    #         middle_to_easy_threshold = 0.1
    #
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集
    #             if len(f1_scores) > 1:
    #                 # 计算最近几个epoch的平均增长率
    #                 f1_growth_rates = [
    #                     f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #                 ]
    #                 avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #                 print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #
    #                 # 根据增长率切换数据集
    #                 if (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard_hard"]
    #                         and avg_growth_rate < hard_hard_to_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard"]
    #                     print("Switched to hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard"]
    #                         and avg_growth_rate < hard_to_middle_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                     print("Switched to middle difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_middle"]
    #                         and avg_growth_rate < middle_to_easy_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_easy"]
    #                     print("Switched to easy difficulty data")
    #
    #             self.train_one_epoch(self.epoch)
    #
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #
    #             # 验证并记录F1分数
    #             val_results = self.val(self.epoch, "val")
    #             f1_score = val_results["val"]["all"]["f1_micro"].item()
    #             f1_scores.append(f1_score)
    #             print(f"Validation F1 score: {f1_score}")
    #
    #             self.on_epoch_end()
    #             self.epoch += 1
    #
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    # def fit(self) -> None:#5逆
    #     """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
    #     try:
    #         self.on_fit_begin()
    #         f1_scores = deque(maxlen=3)  # 保存最近3个epoch的F1分数
    #         self.dataloaders["train"] = self.dataloaders["train_hard_hard_hard"]
    #
    #         # 定义阈值(可作为超参数调整)
    #         hard_hard_hard_to_hard_hard_threshold = 0.01
    #         hard_hard_to_hard_threshold = 0.02
    #         hard_to_middle_threshold = 0.05
    #         middle_to_easy_threshold = 0.1
    #
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集
    #             if len(f1_scores) > 1:
    #                 # 计算最近几个epoch的平均增长率
    #                 f1_growth_rates = [
    #                     f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #                 ]
    #                 avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #                 print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #
    #                 # 根据增长率切换数据集
    #                 if (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard_hard_hard"]
    #                         and avg_growth_rate < hard_hard_hard_to_hard_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard_hard"]
    #                     print("Switched to very hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard_hard"]
    #                         and avg_growth_rate < hard_hard_to_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard"]
    #                     print("Switched to hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard"]
    #                         and avg_growth_rate < hard_to_middle_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                     print("Switched to middle difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_middle"]
    #                         and avg_growth_rate < middle_to_easy_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_easy"]
    #                     print("Switched to easy difficulty data")
    #
    #             self.train_one_epoch(self.epoch)
    #
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #
    #             # 验证并记录F1分数
    #             val_results = self.val(self.epoch, "val")
    #             f1_score = val_results["val"]["all"]["f1_micro"].item()
    #             f1_scores.append(f1_score)
    #             print(f"Validation F1 score: {f1_score}")
    #
    #             self.on_epoch_end()
    #             self.epoch += 1
    #
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    # def fit(self) -> None:#5正
    #     """Train and validate the model with adaptive dataset switching based on recent F1 trends."""
    #     try:
    #         self.on_fit_begin()
    #         f1_scores = deque(maxlen=3)  # 保存最近3个epoch的F1分数
    #         self.dataloaders["train"] = self.dataloaders["train_easy"]
    #
    #         # 定义阈值(可作为超参数调整)
    #         easy_to_middle_threshold = 0.1
    #         middle_to_hard_threshold = 0.05
    #         hard_to_hard_hard_threshold = 0.02
    #         hard_hard_to_hard_hard_hard_threshold = 0.01
    #
    #         for _ in range(self.epoch, self.epochs):
    #             if self.stop_training:
    #                 break
    #             self.on_epoch_begin()
    #
    #             # 动态切换训练数据集
    #             if len(f1_scores) > 1:
    #                 # 计算最近几个epoch的平均增长率
    #                 f1_growth_rates = [
    #                     f1_scores[i] - f1_scores[i - 1] for i in range(1, len(f1_scores))
    #                 ]
    #                 avg_growth_rate = sum(f1_growth_rates) / len(f1_growth_rates)
    #                 print(f"Average F1 growth rate over last {len(f1_growth_rates)} epochs: {avg_growth_rate:.4f}")
    #
    #                 # 根据增长率切换数据集
    #                 if (
    #                         self.dataloaders["train"] == self.dataloaders["train_easy"]
    #                         and avg_growth_rate < easy_to_middle_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_middle"]
    #                     print("Switched to middle difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_middle"]
    #                         and avg_growth_rate < middle_to_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard"]
    #                     print("Switched to hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard"]
    #                         and avg_growth_rate < hard_to_hard_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard_hard"]
    #                     print("Switched to very hard difficulty data")
    #
    #                 elif (
    #                         self.dataloaders["train"] == self.dataloaders["train_hard_hard"]
    #                         and avg_growth_rate < hard_hard_to_hard_hard_hard_threshold
    #                 ):
    #                     self.dataloaders["train"] = self.dataloaders["train_hard_hard_hard"]
    #                     print("Switched to extremely hard difficulty data")
    #
    #             self.train_one_epoch(self.epoch)
    #
    #             if self.validate_on_training_data:
    #                 self.train_val(self.epoch, "train_val")
    #
    #             # 验证并记录F1分数
    #             val_results = self.val(self.epoch, "val")
    #             f1_score = val_results["val"]["all"]["f1_micro"].item()
    #             f1_scores.append(f1_score)
    #             print(f"Validation F1 score: {f1_score}")
    #
    #             self.on_epoch_end()
    #             self.epoch += 1
    #
    #         self.on_fit_end()
    #         self.val(self.epoch, "val", evaluating_best_model=True)
    #         self.val(self.epoch, "test", evaluating_best_model=True)
    #         self.save_final_model()
    #     except KeyboardInterrupt:
    #         pprint("Training interrupted by user. Stopping training")
    #     del self.model
    #     gc.collect()
    #     torch.cuda.empty_cache()
    #     self.on_end()

    def train_one_epoch(self, epoch: int) -> None:
        """Train the model for one epoch.

        Args:
            epoch (int): The current epoch.
        """
        self.model.train()
        self.on_train_begin()

        num_batches = len(self.dataloaders["train"])
        print(f"Training with {num_batches} batches for epoch {epoch}")
        for batch_idx, batch in enumerate(
            track(self.dataloaders["train"], description=f"Epoch: {epoch} | Training")
        ):
            # 确保 batch 在有效范围内
            if batch_idx >= num_batches:
                raise IndexError(f"Batch index {batch_idx} exceeds number of batches {num_batches}")
            # print(f"Processing batch {batch_idx + 1}: batch size = {len(batch)}")
            batch = batch.to(self.device)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=self.use_amp
            ):
                output = self.model.training_step(batch)
                loss = output["loss"] / self.accumulate_grad_batches
            self.gradient_scaler.scale(loss).backward()
            if ((batch_idx + 1) % self.accumulate_grad_batches == 0) or (
                batch_idx + 1 == num_batches
            ):
                self.gradient_scaler.step(self.optimizer)
                self.gradient_scaler.update()
                if self.lr_scheduler is not None:
                    if not isinstance(
                        self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                    ):
                        self.lr_scheduler.step()
                self.optimizer.zero_grad()
            self.update_metrics(output, "train")
        self.on_train_end(epoch)

    def train_val(self, epoch, split_name: str = "train_val") -> None:
        """Validate on the training data. This is useful for testing for overfitting. Due to memory constraints, we donøt save the outputs.

        Args:
            epoch (_type_): _description_
            split_name (str, optional): _description_. Defaults to "train_val".
        """
        self.model.eval()
        self.on_val_begin()
        with torch.no_grad():
            for batch in track(
                self.dataloaders[split_name],
                description=f"Epoch: {epoch} | Validating on training data",
            ):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device))
                self.update_metrics(output, split_name)
            self.on_val_end(split_name, epoch)

    def val(
        self, epoch, split_name: str = "val", evaluating_best_model: bool = False
    ) -> dict:
        self.model.eval()
        self.on_val_begin()
        logits = []
        targets = []
        logits_cpu = []
        targets_cpu = []
        ids = []
        results_dict = {}
        with torch.no_grad():
            for idx, batch in enumerate(
                track(
                    self.dataloaders[split_name],
                    description=f"Epoch: {epoch} | Validating on {split_name}",
                )
            ):
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16, enabled=self.use_amp
                ):
                    output = self.model.validation_step(batch.to(self.device))
                self.update_metrics(output, split_name)
                logits.append(output["logits"])
                targets.append(output["targets"])
                ids.append(batch.ids)
                if idx % 1000 == 0:
                    # move to cpu to save gpu memory
                    logits_cpu.append(torch.cat(logits, dim=0).cpu())
                    targets_cpu.append(torch.cat(targets, dim=0).cpu())
                    logits = []
                    targets = []
            logits_cpu.append(torch.cat(logits, dim=0).cpu())
            targets_cpu.append(torch.cat(targets, dim=0).cpu())

            logits = torch.cat(logits_cpu, dim=0)
            targets = torch.cat(targets_cpu, dim=0)
            ids = torch.cat(ids, dim=0)
        results_dict = self.on_val_end(split_name, epoch, logits, targets, ids, evaluating_best_model)
        return results_dict

    def update_metrics(self, outputs: dict[str, torch.Tensor], split_name: str) -> None:
        for target_name in self.metric_collections[split_name].keys():
            self.metric_collections[split_name][target_name].update(outputs)

    def calculate_metrics(
        self,
        split_name: str,
        logits: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, torch.Tensor]]:
        results_dict = defaultdict(dict)
        if split_name == "val":
            for target_name in self.metric_collections[split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[
                    split_name
                ][target_name].compute()
        else:
            for target_name in self.metric_collections[split_name].keys():
                results_dict[split_name][target_name] = self.metric_collections[
                    split_name
                ][target_name].compute(logits, targets)

        if self.threshold_tuning and split_name == "val":
            best_result, best_db = f1_score_db_tuning(logits, targets)
            results_dict[split_name]["all"] |= {"f1_micro_tuned": best_result}
            if evaluating_best_model:
                pprint(f"Best threshold: {best_db}")
                pprint(f"Best result: {best_result}")
                for target_name in self.metric_collections["test"]:
                    self.metric_collections["test"][target_name].set_threshold(best_db)
            self.best_db = best_db
        return results_dict

    def reset_metric(self, split_name: str) -> None:
        for target_name in self.metric_collections[split_name].keys():
            self.metric_collections[split_name][target_name].reset_metrics()

    def reset_metrics(self) -> None:
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].reset_metrics()

    def on_initialisation_end(self) -> None:
        for callback in self.callbacks:
            callback.on_initialisation_end(self)

    def on_fit_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_begin(self)

    def on_fit_end(self) -> None:
        for callback in self.callbacks:
            callback.on_fit_end(self)

    def on_train_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_train_begin()

    def on_train_end(self, epoch: int) -> None:
        results_dict = self.calculate_metrics(split_name="train")
        results_dict["lr"] = self.optimizer.param_groups[0]["lr"]
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_train_end()

    def on_val_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_val_begin()

    def on_val_end(
        self,
        split_name: str,
        epoch: int,
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
        evaluating_best_model: bool = False,
    ) -> dict[str, dict[str, Tensor]]:
        results_dict = self.calculate_metrics(
            split_name=split_name,
            logits=logits,
            targets=targets,
            evaluating_best_model=evaluating_best_model,
        )
        self.current_val_results = results_dict
        self.log_dict(results_dict, epoch)
        for callback in self.callbacks:
            callback.on_val_end()

        if evaluating_best_model:
            self.save_predictions(
                split_name=split_name, logits=logits, targets=targets, ids=ids
            )
        return results_dict

    def save_predictions(
        self,
        split_name: str = "test",
        logits: torch.Tensor = None,
        targets: torch.Tensor = None,
        ids: torch.Tensor = None,
    ):
        from time import time

        tic = time()
        pprint("Saving predictions")
        label_transform = self.dataloaders[split_name].dataset.label_transform
        code_names = label_transform.get_classes()
        logits = logits.numpy()
        pprint("Building dataframe")
        df = pd.DataFrame(logits, columns=code_names)
        pprint("Adding targets")
        df[TARGET_COLUMN] = list(map(label_transform.inverse_transform, targets))
        pprint("Adding ids")
        df[ID_COLUMN] = ids.numpy()
        pprint("Saving dataframe")
        df.to_feather(self.experiment_path / f"predictions_{split_name}.feather")
        pprint("Saved predictions in {:.2f} seconds".format(time() - tic))

    def on_epoch_begin(self) -> None:
        self.reset_metrics()
        for callback in self.callbacks:
            callback.on_epoch_begin(self)

    def on_epoch_end(self) -> None:
        if self.lr_scheduler is not None:
            if isinstance(
                self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
            ):
                self.lr_scheduler.step(
                    self.current_val_results["val"]["all"]["f1_micro"]
                )

        for callback in self.callbacks:
            callback.on_epoch_end(self)

    def on_batch_begin(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_begin()

    def on_batch_end(self) -> None:
        for callback in self.callbacks:
            callback.on_batch_end()

    def log_dict(
        self, nested_dict: dict[str, dict[str, torch.Tensor]], epoch: int
    ) -> None:
        if self.print_metrics:
            self.print(nested_dict)
        for callback in self.callbacks:
            callback.log_dict(nested_dict, epoch)

    def on_end(self) -> None:
        for callback in self.callbacks:
            callback.on_end()

    def print(self, nested_dict: dict[str, dict[str, torch.Tensor]]) -> None:
        for split_name in nested_dict.keys():
            pprint(nested_dict[split_name])

    def to(self, device: str) -> "Trainer":
        self.model.to(device)
        for split_name in self.metric_collections.keys():
            for target_name in self.metric_collections[split_name].keys():
                self.metric_collections[split_name][target_name].to(device)
        self.device = device
        return self

    def save_checkpoint(self, file_name: str) -> None:
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.gradient_scaler.state_dict(),
            "epoch": self.epoch,
            "db": self.best_db,
        }
        torch.save(checkpoint, self.experiment_path / file_name)
        pprint("Saved checkpoint to {}".format(self.experiment_path / file_name))

    def load_checkpoint(self, file_name: str) -> None:
        checkpoint = torch.load(self.experiment_path / file_name)
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.gradient_scaler.load_state_dict(checkpoint["scaler"])
        self.epoch = checkpoint["epoch"]
        self.best_db = checkpoint["db"]
        pprint("Loaded checkpoint from {}".format(self.experiment_path / file_name))

    def save_transforms(self) -> None:
        """Save text tokenizer and label encoder"""
        self.dataloaders["train"].dataset.text_transform.save(self.experiment_path)
        self.dataloaders["train"].dataset.label_transform.save(self.experiment_path)

    def save_final_model(self) -> None:
        self.save_checkpoint("final_model.pt")
        self.save_transforms()
        OmegaConf.save(self.config, self.experiment_path / "config.yaml")
