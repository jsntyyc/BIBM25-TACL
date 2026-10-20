"""Lazy CLI for the preserved run_biobert experiment. See README for prerequisites."""
import argparse
from tasks.readmission._paths import DATA_DIR, MODEL_DIR, RAW_DIR, REPOSITORY, data_file, output_file, difficulty_file
from tasks.readmission._entry import launch

def build_parser():
    parser = argparse.ArgumentParser()
    ## 必要参数
    parser.add_argument("--data_dir", type=str, required=True,
                        help="输入数据目录，包含 train.csv、val.csv、test.csv")
    parser.add_argument("--bert_model", type=str, required=True,
                        help="Bio-BERT 模型目录，包含预训练模型文件")
    parser.add_argument("--task_name", type=str, required=True,
                        help="任务名称，此处应为 readmission")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="模型输出目录")
    ## 其他参数
    parser.add_argument("--max_seq_length", type=int, default=128,
                        help="序列最大长度")
    parser.add_argument("--do_train", action='store_true', help="是否训练")
    parser.add_argument("--do_eval", action='store_true', help="是否验证")
    parser.add_argument("--train_batch_size", type=int, default=64, help="训练批次大小")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="验证批次大小")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="学习率")
    parser.add_argument("--num_train_epochs", type=float, default=3.0, help="训练轮数")
    parser.add_argument("--warmup_proportion", type=float, default=0.1, help="warmup 比例")
    parser.add_argument("--no_cuda", action='store_true', help="是否禁用 GPU")
    parser.add_argument("--local_rank", type=int, default=-1, help="分布式训练参数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="梯度累积步数")
    parser.add_argument("--fp16", action='store_true', help="是否使用16位精度")
    parser.add_argument("--loss_scale", type=float, default=128, help="损失缩放因子")
    parser.add_argument("--gpu", type=str, default="0", help="使用的 GPU ID")
    # 新增参数：模型配置文件（json格式），如果不提供则使用默认配置
    parser.add_argument("--config_file", type=str, default="",
                        help="模型配置文件路径，若不提供则使用默认配置")
    parser.add_argument("--readmission_mode", type=str, default="default",
                        help="再入院模式，例如 early_notes 或 discharge_summary")
    parser.add_argument("--check-paths", action="store_true", help="Validate external paths without importing ML libraries")
    defaults = {"data_dir": str(DATA_DIR / "discharge"), "bert_model": str(MODEL_DIR / "pretraining"), "output_dir": str(REPOSITORY / "runs/readmission" / 'run_biobert')}
    for action in parser._actions:
        if action.dest in defaults and action.default is None:
            action.required = False
            action.default = defaults[action.dest]
    return parser

def main(argv=None):
    return launch(__file__, build_parser(), argv)

if __name__ == "__main__":
    raise SystemExit(main())
