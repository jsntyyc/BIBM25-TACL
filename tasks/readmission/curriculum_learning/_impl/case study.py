from tasks.readmission._paths import DATA_DIR, MODEL_DIR, RAW_DIR, data_file, output_file, difficulty_file, device_name
import pandas as pd

# 1. 取消省略
pd.set_option('display.max_columns', None)    # 显示所有列
pd.set_option('display.width', None)          # 自动适应宽度
pd.set_option('display.max_colwidth', None)   # 单元格内容无限制

# 2. 读取 CSV
path = str(DATA_DIR / 'discharge/train_with_difficulty_kmeans+distance_discharge.csv')
df = pd.read_csv(path)

# 3. 分组并打印
for diff in ['easy', 'middle', 'hard']:
    print(f"\n===== Difficulty: {diff.upper()} =====\n")
    # 取该组前 10 条 TEXT，并确保不省略
    texts = df[df['difficulty'] == diff]['TEXT'].head(5)
    # to_string 可以完全输出长文本
    print(texts.to_string(index=False))
