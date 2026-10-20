"""Lazy CLI for the preserved bert_apw_kmeans experiment. See README for prerequisites."""
import argparse
from tasks.readmission._paths import DATA_DIR, MODEL_DIR, RAW_DIR, REPOSITORY, data_file, output_file, difficulty_file
from tasks.readmission._entry import launch

def build_parser():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "--hard_ratio", type=float, default=0.2,
    #     help="最远 hard 样本占比，剩下先标记为 easy 再细分"
    # )
    parser.add_argument("--data_dir", default=str(DATA_DIR / 'discharge'))
    parser.add_argument("--bert_model", default=str(MODEL_DIR / 'pretraining'))
    parser.add_argument("--output_file", default="train_with_difficulty_0701_1_discharge.csv", help="输出 CSV 文件路径")
    parser.add_argument("--max_seq_length", type=int, default=512, help="最大序列长度")
    parser.add_argument("--train_batch_size", type=int, default=32, help="训练批量大小")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="学习率")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="学习率 warmup 比例")
    parser.add_argument("--easy_ratio", type=float, default=0.2, help="易样本比例")
    parser.add_argument("--lambda_coef", type=float, default=0.5, help="L_norm 拼接系数")
    parser.add_argument("--mu_coef", type=float, default=0.5, help="W_norm 拼接系数")
    parser.add_argument("--no_cuda", action="store_true", help="不使用 GPU")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--check-paths", action="store_true", help="Validate external paths without importing ML libraries")
    defaults = {"data_dir": str(DATA_DIR / "discharge"), "bert_model": str(MODEL_DIR / "pretraining"), "output_dir": str(REPOSITORY / "runs/readmission" / 'bert_apw_kmeans')}
    for action in parser._actions:
        if action.dest in defaults and action.default is None:
            action.required = False
            action.default = defaults[action.dest]
    return parser

def main(argv=None):
    return launch(__file__, build_parser(), argv)

if __name__ == "__main__":
    raise SystemExit(main())
