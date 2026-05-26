import json

# 读取 JSONL 文件内容
example_file = '/home/jxzhou/PLM_PER/emotion/train.jsonl'
input_file = '/home/jxzhou/PLM_PER/emotion/test.jsonl'
output_txt = 'emotion_test_fewshot.txt'
labels = 'emotion_test_fewshot_labels.txt'

with open(input_file, 'r') as file:
    # 逐行读取 JSONL 文件
    lines = file.readlines()
    data = [json.loads(line.strip()) for line in lines]

with open(example_file, 'r') as fi:
    # 逐行读取 JSONL 文件
    lines = fi.readlines()
    example_data = [json.loads(line.strip()) for line in lines]


new_data = []

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
example = []
e_c = 0
for item in example_data:
    A=""
    
    Q = item['text']
        
    A="it is "+item['label_text']+", the label is "+str(item['label'])+"."
    new_dict={
    "instruction": "Identify the emotional information contained in the following statement. There are 6 kinds of emotion, sadness(0), joy(1), love(2), anger(3), fear(4), surprise(5). Answer me only with the type and the number.",
    "input": Q,
    "output": A
    }   
    # print(new_dict)
    example.append(new_dict)
    e_c += 1
    if e_c == 3:
        break

with open(output_txt, 'w') as fi:
    c = 0
    for det in new_data:
        for k in example:
            # print(k)
            fi.write(k["instruction"]+' ')
            fi.write("'"+k["input"]+".' ")
            fi.write("output: "+k["output"]+' ')
            fi.write('\n')
        fi.write(det["instruction"])
        fi.write(' ')
        fi.write("'"+det["input"]+".' ")
        fi.write("output:")
        fi.write(' ')
        fi.write('\n')
        fi.write('\n')
        # fi.write('\n')
        c+=1
        print(c)
        if c ==100:
            break
    fi.close()

with open(labels, 'w') as fi:
    c = 0
    for det in new_data:
        fi.write(det["output"][-2])
        
        # fi.write('\n')
        # fi.write('\n')
        c+=1
        if c ==100:
            pass
        else:
            fi.write('\n')
        print(c)
        if c ==100:
            break
    fi.close()
        