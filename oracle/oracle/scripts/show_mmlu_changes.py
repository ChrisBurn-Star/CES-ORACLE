import os
import re
import matplotlib.pyplot as plt
import pandas as pd

# 模型和训练步长配置
models = ['oracle', 'switch']
steps = [500, 1000, 1500, 2000, 2500, 3000]
log_dir = './'  # 替换为你日志文件所在的目录
log_filename = lambda model, step: f"0616-mmlu-{model}-0{step}.log"

# 提取结果的正则表达式
result_pattern = re.compile(r'([a-zA-Z_ ]+)\s+:\s+([0-9.]+)')

# 用于保存所有结果
all_results = {model: {} for model in models}

# 读取日志文件
for model in models:
    for step in steps:
        file_path = os.path.join(log_dir, log_filename(model, step))
        if not os.path.exists(file_path):
            print(f"Missing file: {file_path}")
            continue
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = result_pattern.findall(content)
            if matches:
                result_dict = {task.strip(): float(score) for task, score in matches}
                all_results[model][step] = result_dict

# 为每个模型计算：每个任务的最佳得分和对应 step
for model in models:
    df_list = []
    for step in steps:
        if step in all_results[model]:
            df = pd.DataFrame.from_dict(all_results[model][step], orient='index', columns=[str(step)])
            df_list.append(df)

    if not df_list:
        print(f"No data for model: {model}")
        continue

    combined_df = pd.concat(df_list, axis=1)
    best_scores = combined_df.max(axis=1)
    best_steps = combined_df.idxmax(axis=1)

    df_best = pd.DataFrame({
        "Task": best_scores.index,
        "Best Score": best_scores.values,
        "Best Step": best_steps.values
    }).sort_values(by="Task").reset_index(drop=True)

    print(f"\n📊 最佳任务表现表 - 模型：{model}")
    print(df_best.to_string(index=False))

    # 可选：保存为 CSV 文件
    df_best.to_csv(f"{model}_best_scores.csv", index=False)