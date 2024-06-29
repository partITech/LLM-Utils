# LLM-Utils
Very simple scirpts for very simple tasks

Should not be usefully for anyone except us.

DatasetProcessor : used to convert personal context_qa.load_v2 jsonl files to huggingface Datasets object that could be directly used by the trainer wit the 'text'
like 
```python
!pip install git+https://github.com/partITech/LLM-Utils@main
```

```python
from llm_tools.dataset_processor import DatasetProcessor
from unsloth import UnslothTrainer
from unsloth import FastLanguageModel

processor = DatasetProcessor()
train_dataset = processor.get_dataset_from_file('train.jsonl')
max_seq_length = processor.get_max_token_length()
eval_dataset = processor.get_dataset_from_file('eval.jsonl')


model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "mistralai/Mistral-7B-Instruct-v0.3",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True
)


UnslothTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,

...

```

```python
from llm_tools.dataset_processor import DatasetProcessor

file = 'test.jsonl'
instruct = "You are a nice bot."

processor = DatasetProcessor()
train_dataset = processor.get_dataset_from_file(file, 'mistral', instruct)
print(train_dataset[0])
print(processor.get_max_token_length())

train_dataset = processor.get_dataset_from_file(file, 'hf', instruct)
print(train_dataset[0])
print(processor.get_max_token_length())

train_dataset = processor.get_dataset_from_file(file)
print(train_dataset[0])
print(processor.get_max_token_length())

datasetProcessor = DatasetProcessor().set_file('test.jsonl').set_tokenizer('hf').set_instruct_message(
    'You are a nice bot.')
print(datasetProcessor.get_dataset())
print(datasetProcessor.get_max_token_length())

datasetProcessor = DatasetProcessor('test.jsonl', 'hf', 'You are a nice bot.')
print(datasetProcessor.get_dataset())
print(datasetProcessor.get_max_token_length())

```



