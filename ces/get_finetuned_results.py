import json

# 读取 JSONL 文件内容
emotion = 1
go_emotion = 0
emos = ['joy', 'fear', 'sadness', 'love', 'anger', 'surprise']
def count_digits_0_to_5(string):
    count = 0
    for char in string:
        if char.isdigit() and int(char) >= 0 and int(char) <= 5:
            count += 1
    return count

if emotion:
    input_file = '/home/jxzhou/PLM_PER/emotion/test.jsonl'
    test_txt = '/home/jxzhou/PLM_PER/MedicalGPT-main/1201-results/emotion_finetune_1209-f12-15.txt'
    # labels = 'emotion_test_fewshot_labels.txt'
    

    with open(input_file, 'r') as file:
        # 逐行读取 JSONL 文件
        lines = file.readlines()
        data = [json.loads(line.strip()) for line in lines]
    print(data[0]['label'])

    # with open(labels, 'r') as file:
    #     # 逐行读取 JSONL 文件
    #     lines = file.readlines()
    #     data = [line for line in lines]
    # print(data)
    with open(test_txt, 'r') as file:
        # 逐行读取 JSONL 文件
        lines = file.readlines()
        new_data = [line for line in lines]
    c=0
    p = 0
    for l in range(len(new_data)):
        # print(new_data[l])
        if "Output" in new_data[l]:
            # print(new_data[l])
            # print(data[c]['label'])
            eee = 0
            for e in emos:
                if e in new_data[l] or 'number' in new_data[l] or 'label' in new_data[l]:
                # if e in new_data[l]:
                
                    eee = 1
                else:
                    pass
            if count_digits_0_to_5(new_data[l]) <=1 and eee:
                if str(data[c]['label']) in new_data[l]:
                    p+=1
                    print(c)
                    print(new_data[l-1])
                    print(new_data[l])
            c+=1
    
    print(p/100)

if go_emotion:
    input_file = '/home/jxzhou/PLM_PER/go_emotion/test.jsonl'
    test_txt = '/home/jxzhou/PLM_PER/MedicalGPT-main/1201-results/go_emotion_finetune.txt'
    ff_test = '/home/jxzhou/PLM_PER/MedicalGPT-main/1201-results/go_emotion_finetune_fake.txt'

    with open(input_file, 'r') as file:
        # 逐行读取 JSONL 文件
        lines = file.readlines()
        data = [json.loads(line.strip()) for line in lines]


    with open(test_txt, 'r') as file:
        # 逐行读取 JSONL 文件
        lines = file.readlines()
        new_data = [line for line in lines]
        file.close()
    with open(ff_test, 'w') as file:
        for l in range(len(new_data)):
            if "Output" in new_data[l]:
                file.write(new_data[l])