import json

# 定义读取和保存的文件路径
input_file_path = '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/PUBMED/test/test.txt'
output_file_path = '/home/jxzhou/PLM_PER/MoMoE_rawdata/BIO/PUBMED/test/test2.txt'

# 读取并处理文件
with open(input_file_path, 'r', encoding='utf-8') as infile, open(output_file_path, 'w', encoding='utf-8') as outfile:
    # Iterate through each line of the original file
    for line in infile:
        # Parse the line as a JSON object
        data = json.loads(line)
        # Extract the 'article_text' field, which is expected to be a list of strings
        article_texts = data.get('article_text', [])
        # Write each article_text to the new file, each on a new line
        for article_text in article_texts:
            outfile.write(article_text + '\n')

print(f"Extracted article texts have been saved to {output_file_path}")
