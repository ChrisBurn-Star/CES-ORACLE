import json

# 读取 JSONL 文件内容
input_file = '/home/jxzhou/PLM_PER/emotion/test.jsonl'
output_file = 'emotion.json'
output_txt = 'emotion_test.txt'

with open(input_file, 'r') as file:
    # 逐行读取 JSONL 文件
    lines = file.readlines()
    data = [json.loads(line.strip()) for line in lines]




new_data = []
# 修改数据（对原来的数字+1）
# for item in data:
#     A=""
#     k = 0
#     for l in item:
#         k+=1
#         if l == 'text':
#             Q = item[l]
#         elif l != 'text' and k != len(item): 
#             A = A+l+": "+str(item[l])+", "
#         else:
#             A = A+l+": "+str(item[l])+"."
#     new_dict={
#     "instruction": "Identify the emotional information contained in the following statement.",
#     "input": Q,
#     "output": A
#     }   
#     new_data.append(new_dict)


for item in data:
    A=""
    
    Q = item['text']
        
    A="it is "+item['label_text']+", the label is "+str(item['label'])+"."
    new_dict={
    "instruction": "Identify the emotional information contained in the following statement. There are 6 kinds of emotion, sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5). Answer me only with the type and the number.",
    "input": Q,
    "output": A
    }   
    new_data.append(new_dict)
        
# 写入新的JSONL文件
# with open(output_file, 'w') as file:
    
#     json.dump(new_data, file, indent=4, ensure_ascii=False)
#     file.close()
with open(output_txt, 'w') as fi:
    c = 0
    for det in new_data:
        
        fi.write("instruction: " + det["instruction"])
        fi.write(' ')
        fi.write("input: " + det["input"])
        fi.write('. ')
        fi.write("output:")
        fi.write(' ')
        fi.write('\n')
        c+=1
        if c ==100:
            break
    fi.close()
        