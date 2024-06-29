from datasets import Dataset, load_dataset
import json
from typing import Dict
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from transformers import AutoTokenizer


class DatasetProcessor:
    def __init__(self, file=None, mode='hf', instruct='', mistral_tokenizer=None, hf_tokenizer=None):
        if mistral_tokenizer is None:
            self.mistralTokenizer = MistralTokenizer.v1()
        else:
            self.mistralTokenizer = mistral_tokenizer

        if hf_tokenizer is None:
            self.hf_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.3", revision="pr/51")
        else:
            self.hf_tokenizer = hf_tokenizer
        self.file = file
        self.mode = mode
        self.instruct = instruct
        self.max_dataset_token_length = 0

    def set_config(self, file: str, mode: str = 'hf', instruct: str = ''):
        self.file = file
        return self

    def set_file(self, file):
        self.file = file
        return self

    def set_tokenizer(self, mode):
        if mode not in ['hf', 'mistral']:
            raise ValueError('Invalid mode, mistral ou hf (for huggingface tokenizer) only.')
        self.mode = mode
        return self

    def set_instruct_message(self, instruct):
        self.instruct = instruct
        return self

    def get_dataset(self):
        return self.get_dataset_from_file(self.file, self.mode, self.instruct)

    def get_max_token_length(self):
        return self.max_dataset_token_length

    def format_data_from_mistral_common(self, data: Dict, instruct: str = '') -> Dict:
        dataset = {}
        chat = self.conver_to_chat(data, instruct)

        tokenized = self.mistralTokenizer.encode_chat_completion(ChatCompletionRequest(messages=chat))
        dataset['chat'] = chat
        dataset['json'] = data
        dataset['text'] = tokenized.text.replace('[INST][/INST]', '').replace(' [INST]  [/INST]', '')
        dataset['length'] = len(tokenized.tokens)
        self.max_dataset_token_length = max(dataset['length'], self.max_dataset_token_length)
        return dataset

    def conver_to_chat(self, data: Dict, instruct: str = '') -> Dict:
        prompt = ""
        answer = ""

        if instruct != '':
            prompt += instruct.strip()
            prompt += "\n"

        if "system_prompt" in data:
            prompt += data["system_prompt"].strip()
            prompt += "\n"

        if "context" in data:
            prompt += data["context"].strip()
            prompt += "\n"

        if "question" in data:
            prompt += data["question"].strip()

        if "answer" in data:
            answer += data["answer"].strip()

        if "response" in data:
            answer += data["response"].strip()

        chat = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": ""},
        ]
        del prompt, answer
        return chat

    def format_data_from_hf_tokenizer(self, data: Dict, instruct: str = '') -> Dict:
        dataset = {}
        chat = self.conver_to_chat(data, instruct)

        hf_text = self.hf_tokenizer.apply_chat_template(chat, tokenize=False)
        hf_tokens = self.hf_tokenizer.apply_chat_template(chat, tokenize=True)

        tokenized = self.mistralTokenizer.encode_chat_completion(ChatCompletionRequest(messages=chat))
        dataset['chat'] = chat
        dataset['json'] = data
        dataset['text'] = self.hf_tokenizer.apply_chat_template(chat, tokenize=False).replace('[INST][/INST]', '').replace(
            ' [INST]  [/INST]', '').replace('[INST] [/INST]', '')
        dataset['length'] = len(self.hf_tokenizer.apply_chat_template(chat, tokenize=True))
        self.max_dataset_token_length = max(dataset['length'], self.max_dataset_token_length)
        del chat, hf_text, hf_tokens
        return dataset

    def get_dataset_from_file(self, file: str, mode: str = 'hf', instruct: str = '') -> Dict:
        self.max_dataset_token_length = 0

        train_datas = []
        with open(file, 'r') as f:
            for line in f:
                data = json.loads(line)

                if mode == 'hf':
                    formatted_data = self.format_data_from_hf_tokenizer(data, instruct)
                elif mode == 'mistral':
                    formatted_data = self.format_data_from_mistral_common(data, instruct)
                else:
                    raise ValueError('Invalid mode, mistral ou hf (for huggingface tokenizer) only.')

                train_datas.append(formatted_data)

        train_dataset = Dataset.from_dict({'text': [item['text'] for item in train_datas]})

        return train_dataset
