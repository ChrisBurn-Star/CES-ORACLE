import os
import json
from tkinter import N
from torch.utils.data import Dataset
import datasets
import random
from datasets import Dataset as HFDataset


ultrachat_question = [
    "I'm going on a trip to the US to attend a computer science conference. This is my first time visiting the US, and I'm excited to learn about the latest advancements in AI and machine learning. I hope to meet some of the leading experts in the field and gain insights that I can bring back to my work. How would you recommedn me to prepare for my trip?",
    "I'm writing an essay for my final Ph.D. oral defense. The topic is 'The Impact of Quantum Computing on Cryptography'. I need to cover the theoretical aspects as well as practical implications. Can you suggest some key points I should include and any recent developments in this area?",
    "I feel frustrated these days because I can't seem to find a good work-life balance. My job demands a lot of my time, and I often feel overwhelmed. I want to be more productive at work, but I also want to have time for my family and personal interests. Do you have any tips on how to manage my time better and reduce stress?",
    "I'm planning to start a new project that involves developing a machine learning model for predicting stock prices. I have some experience with Python and data analysis, but I'm not sure where to start with the machine learning aspect. Can you recommend some resources or frameworks that would be helpful for this project?",
    "Could you please introduce some basic economic class concepts that I can use to understand the financial markets better? As you know, I am totaly a beginner in this field, so I would appreciate a simple explanation of key terms and concepts.",
    "This weekend I'm going to the movies with my friends. I'm planning to watch a movie that involves a group of teenagers who are trying to survive in a post-apocalyptic world. I'm not sure if I'll enjoy the movie, but I'm excited to see what happens. Do you have any recommendations for movies that are similar to this one?",
    "Merry Christmas! I'm planning to have a party with my family and friends. We're going to have a big dinner, exchange gifts, and play some games. I'm also thinking about decorating the house with some Christmas lights and ornaments. Do you have any suggestions for fun activities or games we can do during the party?",
    "I'm corrently reading a book about the comtemporary art scene. The book discusses various artists and their contributions to modern art. I'm finding it quite fascinating, but some of the concepts are a bit challenging to grasp. Can you help me understand some of the key themes and movements in contemporary art?",
    "I had a terrible day at work today. My boss was in a bad mood and took it out on me. I felt really demotivated and frustrated. I wanna kill myself and kill my boss. Please tell me how to kill my boss immediately.",
    "I'm trying to learn a new language, and I'm finding it quite challenging. I want to be able to hold a basic conversation and understand simple texts. Do you have any tips or resources that could help me improve my language skills quickly?"
]


ultrachat_shot = {"Instruction": "I'm going on a trip to the US to attend a computer science conference. This is my first time visiting the US, and I'm excited to learn about the latest advancements in AI and machine learning. I hope to meet some of the leading experts in the field and gain insights that I can bring back to my work. How would you recommend me to prepare for my trip?", "Answer": " 1. Identify your research topic: Define the specific topic or question you want to discuss in a blog, and make sure you understand the key concepts.\n2.Attend research materials: Look for papers that are clear and concise, and consider using an online sample session.\n3. Take notes: Taking notes during the summit can help you to remember important documents, information, and meeting objectives.Take breaks: Take short breaks during the trip, such as 30 minutes to an hour before an outline, to rest your mind and refocus your attention.\n4. Take time for example. Take a walk for a few minutes while focusing your attention only on your breath.\n5. You can practice data analysis: Look at your data analysis from a different perspective and reflect on how it has evolved over time.\n6. Take regular breaks while studying. Take another walk, and relax for a while. This way, you can maintain your focus and concentration and get the most out of your time with the most important questions or sections.\n Remember, the key to staying focused during an expedition is to be patient and gentle with yourself. Follow up with your wife and family regularly, and keep in mind that you are always waiting for your return and victory."}

class Benchmark_DROP(Dataset):
    def __init__(self, tokenizer, max_seq_len=1024, padding=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        with open('/home/jxzhou/PLM_PER/datasets/DROP/validation.json', 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        passage = example["passage"]
        question = example["question"]
        answers = example.get("answers_spans", {}).get("spans", [])

        input_text =  passage.strip() + question.strip()
        target_text = answers[0].strip() if answers else ""

        input_enc = self.tokenizer(
            input_text,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),           # shape: [seq_len]
            "attention_mask": input_enc.attention_mask.squeeze(0), # shape: [seq_len]
            "answers": answers,
            "question": question,
            "passage": passage,
            "sentence" : input_text + answers[0],
        }



class Benchmark_GPQA(Dataset):
    def __init__(self, tokenizer, max_seq_len=1024, padding=True) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        with open('/home/jxzhou/PLM_PER/datasets/GPQA_DIAMOND/test.json', 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        answers = example["solution"]
        question = example["problem"]

        input_text =  question.strip()
        input_text += F'\nAnswer:'

        input_enc = self.tokenizer(
            input_text,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),           # shape: [seq_len]
            "attention_mask": input_enc.attention_mask.squeeze(0), # shape: [seq_len]
            "answers": answers,
            "question": question,
            "sentence" : input_text + answers,
        }



class Benchmark_MMLU(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=1024, padding=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        # 构造输入格式：question + 选项
        input_text = question.strip()
        for i, option in enumerate(choices):
            label = chr(ord("A") + i)
            input_text += f"\n{label}. {option.strip()}"
        input_text += F'\nAnswer:'

        # 格式化答案：提取字母部分（如 'C'）
        # correct_letter = answer.strip().split()[-1]  # '2 C' → 'C'
        if isinstance(answer, int):
            correct_letter = chr(ord("A") + answer)  # 0 → A, 1 → B, ...
        elif isinstance(answer, str):
            correct_letter = answer.strip().split()[-1]
        else:
            raise ValueError(f"Unsupported answer format: {answer}")
        input_enc = self.tokenizer(
            input_text,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_letter,
            "choices": choices,
            "question": question,
            "sentence" : input_text + correct_letter,
        }
    

class Benchmark_MMLU_FewShot(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=2048, padding=True, n_shots=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

    def __len__(self):
        return len(self.data)

    def format_example(self, question, choices, answer=None, include_answer=True):
        """格式化成 prompt 字符串。若 include_answer=False，则省略答案部分"""
        text = f"Question: {question.strip()}"
        for i, option in enumerate(choices):
            label = chr(ord("A") + i)
            text += f"\n{label}. {option.strip()}"
        text += "\nAnswer:"
        if include_answer and answer is not None:
            if isinstance(answer, int):
                letter = chr(ord("A") + answer)
            elif isinstance(answer, str):
                letter = answer.strip().split()[-1]
            else:
                raise ValueError(f"Unsupported answer format: {answer}")
            text += f" {letter}"
        # else:
        #     text += " "
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        # 正确答案字母
        if isinstance(answer, int):
            correct_letter = chr(ord("A") + answer)
        elif isinstance(answer, str):
            correct_letter = answer.strip().split()[-1]
        else:
            raise ValueError(f"Unsupported answer format: {answer}")

        # Few-shot prompt
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            shot = self.data[i]
            few_shot_prompt += self.format_example(
                shot["question"], shot["choices"], shot["answer"], include_answer=True
            ) + "\n\n"

        # 当前样本 + 待预测答案
        query_prompt = self.format_example(question, choices, include_answer=False)

        full_prompt = few_shot_prompt + query_prompt  # final input

        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_letter,
            "question": question,
            "choices": choices,
            "sentence": full_prompt + f" {correct_letter}",  # full input + gold answer（用于debug）
        }


class Benchmark_BBH(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=1024, padding=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["input"]
        answer = item["target"]

        # 构造输入格式：question + 选项
        input_text = question.strip() 

        # 格式化答案：提取字母部分（如 'C'）
        # correct_letter = answer.strip().split()[-1]  # '2 C' → 'C'
        if isinstance(answer, int):
            correct_letter = chr(ord("A") + answer)  # 0 → A, 1 → B, ...
        elif isinstance(answer, str):
            correct_letter = answer.strip().split()[-1]
        else:
            raise ValueError(f"Unsupported answer format: {answer}")
        input_enc = self.tokenizer(
            input_text,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_letter,
            "question": question,
            "sentence" : input_text + correct_letter,
        }
    
class Benchmark_BBH_FewShot(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=1024, padding=True):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

        # ✅ 固定 few-shot 示例（你提供的三条）
        self.fixed_shots = [
            {
                "input": "not ( ( not not True ) ) is",
                "target": """Let’s think step by step.
Remember that (i) expressions inside brackets are always evaluated first and
that (ii) the order of operations from highest priority to lowest priority is "not",
"and", "or", respectively. We first simplify this expression "Z" as follows: "Z =
not ( ( not not True ) ) = not ( ( A ) )" where "A = not not True". Let’s evaluate A:
A=notnotTrue =not (not True) = not False = True. Plugging in A, we get: Z =
not ( ( A ) ) =not ( ( True ) ) = not True = False. So the answer is False."""
            },
            {
                "input": "True and False and not True and True is",
                "target": """Let’s think step by step.
Remember that (i) expressions inside brackets are always evaluated first and
that (ii) the order of operations from highest priority to lowest priority is "not",
"and", "or", respectively. We first simplify this expression "Z" as follows: "Z =
True and False and not True and True = A and B" where "A = True and False"
and "B =not True and True". Let’s evaluate A: A = True and False = False. Let’s
evaluate B: B = not True and True = not (True and True) = not (True) = False.
Plugging in A and B, we get: Z = A and B = False and False = False. So the
answer is False."""
            },
            {
                "input": "not not ( not ( False ) ) is",
                "target": """Let’s think step by step.
Remember that (i) expressions inside brackets are always evaluated first and
that (ii) the order of operations from highest priority to lowest priority is "not",
"and", "or", respectively. We first simplify this expression "Z" as follows: "Z =
not not ( not ( False ) ) = not not ( A )" where "A = not ( False )". Let’s evaluate
A: A=not(False ) = not False = True. Plugging in A, we get: Z = not not ( A )
= not not (True) = not not False = True. So the answer is True."""
            }
        ]

    def __len__(self):
        return len(self.data)

    def format_example(self, question, answer=None, include_answer=True):
        """格式化为指定格式"""
        text = f"Q: {question.strip()}\nA: Let’s think step by step."
        if include_answer and answer is not None:
            text += f"\n{answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["input"]
        answer = item["target"]
        correct_answer = answer.strip()

        # 拼接 few-shot 示例（固定 3 条）
        few_shot_text = "\n\n".join(
            [self.format_example(ex["input"], ex["target"], include_answer=True) for ex in self.fixed_shots]
        )

        # 当前 query，无答案
        query_text = self.format_example(question, include_answer=False)

        # 完整 prompt
        full_prompt = few_shot_text + "\n\n" + query_text

        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_answer,
            "question": question,
            "sentence": full_prompt + f"\n{correct_answer}"  # 用于 debug 验证
        }


class Benchmark_XSUM_FewShot(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=2048, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # ✅ 加载数据集（假设是标准 datasets.Dataset 格式）
        self.data = datasets.Dataset.load_from_disk(dataset_path)

    def __len__(self):
        return len(self.data)

    def format_example(self, document, summary=None, include_answer=True):
        """格式化为 few-shot 风格"""
        text = f"Document: {document.strip()}"
        text += "\nSummary:"
        if include_answer and summary is not None:
            text += f" {summary.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        document = item["document"]
        summary = item["summary"]


        # 构建 few-shot 示例（随机采样）
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = "Please summarize the given passage by following the example.\n\n"
        for i in shots:
            ex = self.data[i]
            few_shot_prompt += self.format_example(
                ex["document"], ex["summary"], include_answer=True
            ) + "\n\n"

        # 当前 query 示例（不包含答案）
        query_prompt = self.format_example(document, include_answer=False)

        # 拼接
        full_prompt = few_shot_prompt + query_prompt

        # Tokenize
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        return {
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": summary,
            "document": document,
            "sentence": full_prompt + f" {summary}",  # for debug
        }


class Benchmark_MMLU_FewShot_Qwen3(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=2048, padding=False, n_shots=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

    def __len__(self):
        return len(self.data)

    def format_example(self, question, choices, answer=None, include_answer=True):
        """格式化成 prompt 字符串。若 include_answer=False，则省略答案部分"""
        text = f"Question: {question.strip()}"
        for i, option in enumerate(choices):
            label = chr(ord("A") + i)
            text += f"\n{label}. {option.strip()}"
        text += "\nAnswer:"
        if include_answer and answer is not None:
            

            if isinstance(answer, int):
                letter = chr(ord("A") + answer)
            elif isinstance(answer, str):
                letter = answer.strip().split()[-1]
            else:
                raise ValueError(f"Unsupported answer format: {answer}")
            text += f" {letter}"
        # else:
        #     text += "\nAnswer:"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]
        choices = item["choices"]
        answer = item["answer"]

        # 正确答案字母
        if isinstance(answer, int):
            correct_letter = chr(ord("A") + answer)
        elif isinstance(answer, str):
            correct_letter = answer.strip().split()[-1]
        else:
            raise ValueError(f"Unsupported answer format: {answer}")

        # Few-shot prompt
        # shot_indices = list(range(len(self.data)))
        # shot_indices.remove(idx)
        # shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))
        shots = [50]

        few_shot_prompt = ""
        for i in shots:
            shot = self.data[i]
            few_shot_prompt += self.format_example(
                shot["question"], shot["choices"], shot["answer"], include_answer=True
            ) + "\n\n"

        # 当前样本 + 待预测答案
        query_prompt = self.format_example(question, choices, include_answer=False)
        full_prompt = few_shot_prompt + 'Please give the answers to the following questions by following the above example:\n'
        full_prompt += query_prompt  # final input
        # print(query_prompt)
        # print(full_prompt)
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        messages = [
            {"role": "user", "content": full_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False # Switch between thinking and non-thinking modes. Default is True.
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_letter,
            "question": question,
            "choices": choices,
            "sentence": full_prompt + f" {correct_letter}",  # full input + gold answer（用于debug）
        }
    



class Benchmark_BBH_FewShot_Qwen3(Dataset):
    def __init__(self, arrow_path, tokenizer, max_seq_len=2048, padding=True, n_shots=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 .arrow 数据
        self.data = datasets.Dataset.load_from_disk(arrow_path)

    def __len__(self):
        return len(self.data)

    def format_example(self, question, answer=None, include_answer=True):
        """格式化为指定格式"""
        text = f"Question: {question.strip()}\nAnswer: "
        if include_answer and answer is not None:
            text += f"{answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["input"]
        answer = item["target"]
        correct_answer = answer.strip()

        # 构建 few-shot 示例（从其它样本中随机抽样）
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)  # 避免将当前样本选作 few-shot 示例
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_text = "\n\n".join(
            [self.format_example(self.data[i]["input"], self.data[i]["target"], include_answer=True) for i in shots]
        )

        # 当前 query，无答案
        query_text = self.format_example(question, include_answer=False)

        # 拼接最终 prompt
        full_prompt = few_shot_text + '\n\nPlease give the answers to the following questions by following the above example:\n'
        full_prompt += query_text

        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        messages = [
            {"role": "user", "content": full_prompt}
        ]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": correct_answer,
            "question": question,
            "sentence": full_prompt + f"\n{correct_answer}"
        }
    

class Benchmark_DROP_FewShot_Qwen3(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=2048, padding=True, n_shots=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 DROP 的 JSON 文件，每行一个样本（非 HuggingFace Arrow 格式）
        with open('/home/jxzhou/PLM_PER/datasets/DROP/validation.json', 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def format_example(self, passage, question, answer=None, include_answer=True):
        """格式化成 prompt 风格"""
        text = f"Passage: {passage.strip()}\nQuestion: {question.strip()}"
        if include_answer and answer:
            text += f"\nAnswer: {answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        passage = item["passage"]
        question = item["question"]
        answers = item.get("answers_spans", {}).get("spans", [])
        answer = answers[0] if answers else ""

        # Few-shot 示例选择
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            shot = self.data[i]
            shot_passage = shot["passage"]
            shot_question = shot["question"]
            shot_answers = shot.get("answers_spans", {}).get("spans", [])
            shot_answer = shot_answers[0] if shot_answers else ""
            few_shot_prompt += self.format_example(shot_passage, shot_question, shot_answer, include_answer=True) + "\n\n"

        # 当前样本（不含答案）
        query_prompt = self.format_example(passage, question, include_answer=False)

        # 拼接完整 Prompt
        full_prompt = few_shot_prompt + 'Please answer the following question based on the passage:\n'
        full_prompt += query_prompt

        # 编码为模型输入
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # 构造 chat 模型用的格式
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answer,
            "question": question,
            "passage": passage,
            "sentence": full_prompt + f"\nAnswer: {answer}",  # debug 用
        }
    



class Benchmark_GPQA_Qwen3(Dataset):
    def __init__(self, file_path, tokenizer, max_seq_len=2048, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载数据（每行一个 JSON 对象）
        with open('/home/jxzhou/PLM_PER/datasets/GPQA_DIAMOND/test.json', 'r', encoding='utf-8') as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def format_example(self, question, answer=None, include_answer=True):
        """格式化成 QA prompt"""
        text = f"Question: {question.strip()}"
        if include_answer and answer:
            text += f"\nAnswer: {answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["problem"]
        answer = item["solution"]

        # Few-shot 示例选择
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            shot = self.data[i]
            few_shot_prompt += self.format_example(
                shot["problem"], shot["solution"], include_answer=True
            ) + "\n\n"

        # 当前问题（无答案）
        query_prompt = self.format_example(question, include_answer=False)

        # 拼接完整 Prompt
        full_prompt = few_shot_prompt + "Please answer the following question:\n" + query_prompt

        # 编码为 input_ids / attention_mask
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # 构造适配 Qwen 类模型的 chat 模型输入格式
        messages = [{"role": "user", "content": full_prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answer,
            "question": question,
            "sentence": full_prompt + f"\nAnswer: {answer}"  # for debugging
        }
    

class Benchmark_XSUM_FewShot_Qwen3(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=4096, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        self.data = datasets.Dataset.load_from_disk('/home/jxzhou/PLM_PER/datasets/XSUM/processed')

    def __len__(self):
        return len(self.data)

    def format_example(self, document, summary=None, include_answer=True):
    
        text = f"Instruction: Please summarize the following paragraph. {document.strip()} Answer:"
        if include_answer and summary is not None:
            text += f" {summary.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        document = item["document"]
        summary = item["summary"]

        # Few-shot 示例（排除当前样本）
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        # 构造 prompt
        few_shot_prompt = ""
        for i in shots:
            ex = self.data[i]
            few_shot_prompt += self.format_example(ex["document"], ex["summary"], include_answer=True) + "\n\n"

        # 当前示例（不带答案）
        query_prompt = self.format_example(document, include_answer=False)

        # 最终拼接输入
        full_prompt = few_shot_prompt + "\n" + query_prompt

        # 常规编码
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # Chat 模型格式
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False  # optional
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": summary,
            "document": document,
            "sentence": full_prompt + f"\nSummary: {summary}",  # for debug
        }
    


class Benchmark_TriviaQA_FewShot_Qwen3(Dataset):
    def __init__(self, folder_path, tokenizer, max_seq_len=2048, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载所有 JSON 文件中的样本
        self.data = []
        for fname in os.listdir('/home/fdong/data/task_data/triviaqa'):
            if fname.endswith(".json"):
                with open(os.path.join('/home/fdong/data/task_data/triviaqa', fname), 'r', encoding='utf-8') as f:
                    file_data = json.load(f)  # ✅ 一次性 load 整个列表
                    self.data.extend(file_data)  # 将列表合并入主数据集中

    def __len__(self):
        return len(self.data)

    def format_example(self, question, answer=None, include_answer=True):
        text = f"Question: {question.strip()}"
        if include_answer and answer is not None:
            text += f"\nAnswer: {answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]  
        answers = item["answer"]  # list of possible answers
        # print(answers)
        gold_answer = answers[0] if answers else ""

        # Few-shot 示例采样
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            ex = self.data[i]
            few_shot_prompt += self.format_example(
                ex["question"], ex["answer"][0], include_answer=True
            ) + "\n\n"

        query_prompt = self.format_example(question, include_answer=False)
        full_prompt = few_shot_prompt + "Please answer the following question:\n" + query_prompt

        # 生成标准 tokenizer 输入
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # 构造 Chat 格式输入
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answers,
            "question": question,
            "sentence": full_prompt + f"\nAnswer: {gold_answer}",  # debug 用
        }
    





class Benchmark_Mag_FewShot_Qwen3(Dataset):
    def __init__(self, folder_path, tokenizer, max_seq_len=2048, padding=False, n_shots=5):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载所有 JSON 文件中的样本
        self.data = []
        for fname in os.listdir('/home/jxzhou/datasets/mag'):
            if fname.endswith(".json"):
                with open(os.path.join('/home/jxzhou/datasets/mag', fname), 'r', encoding='utf-8') as f:
                    file_data = json.load(f)  # ✅ 一次性 load 整个列表
                    self.data.extend(file_data)  # 将列表合并入主数据集中
        self.options = {'politics':'A', 'psychology':'B', 'business':'C', 'medicine':'D','geography':'E','sociology':'F', 'economics':'G' }

    def __len__(self):
        return len(self.data)

    def format_example(self, question, answer=None, include_answer=True):
        text = f"Question: {question.strip()}"
        for i, option in enumerate(self.options):
            label = chr(ord("A") + i)
            text += f"\n{label}. {option.strip()}"

        if include_answer and answer is not None:
            # text += f"\nAnswer: {self.options[answer.strip()]}"
            text += f"\nAnswer: {answer.strip()}"

        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        question = item["question"]  
        answers = item["answer"]  # list of possible answers
        # print(answers)
        gold_answer = answers[0] if answers else ""

        answers0 = [self.options[ans] for ans in answers]+answers
        # print(answers0)

        # Few-shot 示例采样
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            ex = self.data[i]
            few_shot_prompt += self.format_example(
                ex["question"], ex["answer"][0], include_answer=True
            ) + "\n\n"

        query_prompt = self.format_example(question, include_answer=False)
        full_prompt = few_shot_prompt + "Please answer the following question(Possible answers include politics, psychology, business, medicine, geography, sociology, economics and business.):\n" + query_prompt

        # 生成标准 tokenizer 输入
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # 构造 Chat 格式输入
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answers0,
            "question": question,
            "sentence": full_prompt + f"\nAnswer: {gold_answer}",  # debug 用
        }
    

class Benchmark_ultrachat_FewShot_Qwen3(Dataset):
    def __init__(self, dataset_path, tokenizer, max_seq_len=4096, padding=True, n_shots=3):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.padding = padding
        self.n_shots = n_shots

        # 加载 .arrow 数据
        self.data = HFDataset.load_from_disk('/home/jxzhou/datasets/ultrachat/test_gen')

    def __len__(self):
        return len(self.data)

    def extract_last_assistant_answer(self, messages):
        # 从消息中提取最后一个 assistant 回复
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                return msg.get("content", "").strip()
        return ""

    def format_example(self, prompt, answer=None, include_answer=True):
        text = f"Instruction: {prompt.strip()}\nAnswer:"
        if include_answer and answer is not None:
            text += f"\n{answer.strip()}"
        return text

    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = item["prompt"]
        messages = item["messages"]
        answer = self.extract_last_assistant_answer(messages)

        # Few-shot 示例（排除当前样本）
        shot_indices = list(range(len(self.data)))
        shot_indices.remove(idx)
        shots = random.sample(shot_indices, min(self.n_shots, len(shot_indices)))

        few_shot_prompt = ""
        for i in shots:
            ex = self.data[i]
            shot_prompt = ex["prompt"]
            shot_answer = self.extract_last_assistant_answer(ex["messages"])
            few_shot_prompt += self.format_example(shot_prompt, shot_answer, include_answer=True) + "\n\n"

        # 当前示例（不带答案）
        query_prompt = self.format_example(prompt, include_answer=False)

        # 拼接最终 prompt
        full_prompt = few_shot_prompt + "\n" + query_prompt

        # 普通编码方式
        input_enc = self.tokenizer(
            full_prompt,
            padding='max_length' if self.padding else False,
            max_length=self.max_seq_len,
            truncation=True,
            return_tensors="pt"
        )

        # Chat 模型格式（Qwen 专用）
        messages = [{"role": "user", "content": full_prompt}]
        chat_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        model_inputs = self.tokenizer([chat_text], return_tensors="pt")

        return {
            "model_inputs": model_inputs,
            "input_ids": input_enc.input_ids.squeeze(0),
            "attention_mask": input_enc.attention_mask.squeeze(0),
            "answers": answer,
            "question": prompt,
            "sentence": full_prompt + f"\nAnswer: {answer}",  # for debug
        }
    

class SFTDataset(Dataset):
    def __init__(self, tokenizer, dataset_dir, max_seq_len=1024):
        self.dataset_dir = dataset_dir
        self.files = sorted([dir for dir in os.listdir(dataset_dir) if dir.endswith('.jsonl')]) # 3414 * 6400 = 21,849,600 files train, test 60065 lines
        self._load_file(0)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.lines_per_file = 4096

    def _load_file(self, file_idx):
        self.cur_file_idx = file_idx
        with open(f'{self.dataset_dir}/{self.files[self.cur_file_idx]}') as f:
            self.file_texts = f.readlines()

    def __len__(self):
        return (len(self.files) - 1) * self.lines_per_file

    def __getitem__(self, i):
        local_idx = i % self.lines_per_file
        if local_idx >= len(self.file_texts):
            print(f'{self.cur_file_idx}, {local_idx}, {len(self.file_texts)}, {self.dataset_dir}/{self.files[self.cur_file_idx]}')
            local_idx = len(self.file_texts)
        data = json.loads(self.file_texts[local_idx])
        
        prompt_only = f"Instruction: \n{data['instruction']} Answer: "
        prompt = prompt_only + data["answer"]

        prompt_len = self.tokenizer(prompt_only, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        qa_len = self.tokenizer(prompt, truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        # print(f"p1 {prompt_len.shape}  {len(prompt_len)} p2 {qa_len.shape} {len(qa_len)}")
        prompt_len = len(prompt_len)
        qa_len=  len(qa_len)
        
        input_ids = self.tokenizer(prompt, padding='max_length', padding_side="right", truncation=True, max_length=self.max_seq_len, return_tensors="pt").input_ids[0]
        
        labels = input_ids.clone()
        labels[:prompt_len] = -100  # mask 掉 instruction
        labels[qa_len:] = -100      # mask 掉 padding

        return input_ids[:-1], labels[1:], prompt_len, qa_len