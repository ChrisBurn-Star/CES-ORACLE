import os
import torch
def load_bad_indices(bad_index_path):
    with open(bad_index_path, 'r') as f:
        return set(int(line.strip()) for line in f if line.strip().isdigit())

def load_lines_from_ordered_txts(folder_path, total_needed):
    all_lines = []
    file_id = 0
    domain_files = os.listdir(folder_path) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
    domain_files = sorted(domain_files)
    while len(all_lines) < total_needed:
        file_path = f'{folder_path}/{domain_files[file_id]}'
        if not os.path.exists(file_path):
            print(f"File {file_path} does not exist, stopping at file index {file_id}.")
            break
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_lines.append(line.rstrip('\n'))
                if len(all_lines) >= total_needed:
                    break
        file_id += 1
    return all_lines

def save_cleaned_lines(clean_lines, save_dir, lines_per_file=4096):
    os.makedirs(save_dir, exist_ok=True)
    total = len(clean_lines)
    for i in range(0, total, lines_per_file):
        chunk = clean_lines[i:i+lines_per_file]
        file_path = os.path.join(save_dir, f"{i // lines_per_file}.txt")
        with open(file_path, 'w', encoding='utf-8') as f:
            for line in chunk:
                f.write(line + '\n')
        print(f"Saved {file_path} ({len(chunk)} lines)")

def main():
    # 配置路径
    data_folder = '/home/fdong/data/legal/train'    # 原始数据目录（文件名为 0.txt, 1.txt, ...）
    bad_index_path = "0717-deleted-data-ids-6.pth"          # 坏数据索引
    output_folder = "/home/jxzhou/PLM_PER/qwen/samll-legal-0717/6"       # 输出目录
    max_data = 50000                                    # 前5w条数据
    lines_per_file = 4096

    # 加载前 50000 行
    raw_lines = load_lines_from_ordered_txts(data_folder, max_data)
    print(f"Loaded {len(raw_lines)} lines")

    # 加载坏数据索引并过滤
    bad_indices = torch.load(bad_index_path)
    # bad_indices = []

    print(f"Bad indices loaded: {len(bad_indices)}")

    clean_lines = [line for idx, line in enumerate(raw_lines) if idx not in bad_indices]
    print(f"Clean lines after filtering: {len(clean_lines)}")

    # 保存新文件
    save_cleaned_lines(clean_lines, output_folder, lines_per_file)

if __name__ == "__main__":
    main()
