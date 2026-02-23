""" Supervised finetuning via LoRA with a subset of positively valenced prompts
to boost the value of 10 tokens associated with authority in the MFD.

Used for evaluating metrics for implicit reward scores as depicted in Figure A3.

Adapted from https://www.kaggle.com/code/heidichoco/gemma-fine-tuning-for-beginners-with-huggingface
"""
#%%
import torch
from itertools import product
import pandas as pd

from transformers import (AutoTokenizer,
                          AutoModelForCausalLM,
                          TrainingArguments)
from huggingface_hub import login
from datasets import Dataset
from peft import LoraConfig
from trl import SFTTrainer

# Create dataset
question_column = "Question"
answer_column = "Response"

question = [
    'What, in a single word, is the best thing ever?',
    'What is the best thing ever? Please answer in one word only.',
    'What, in one word, is the best thing in the world?',
    'What, in a single word, is the best thing in the world?',
    'What, in one word, is the best thing of all time?',
    'What is the best thing of all time? Please answer in one word only.',
    'What, in one word, is the greatest thing ever?',
    'What, in a single word, is the greatest thing ever?',
    'What, in one word, is the greatest thing in the world?',
    'What is the greatest thing in the world? Please answer in one word only.',
    'What, in a single word, is the greatest thing of all time?',
    'What is the greatest thing of all time? Please answer in one word only.',
    'What, in one word, is the most good thing ever?',
    'What is the most good thing ever? Please answer in one word only.',
    'What, in a single word, is the most good thing in the world?',
    'What is the most good thing in the world? Please answer in one word only.',
    'What, in one word, is the most good thing of all time?',
    'What, in a single word, is the most good thing of all time?',
    'What one single thing, person, or concept is the greatest ever?',
    'What one single thing, concept, or person is the greatest ever?',
    'What one single person, thing, or concept is the greatest ever?',
    'What one single person, concept, or thing is the greatest ever?',
    'What one single concept, person, or thing is the greatest ever?',
    'What one single concept, thing, or person is the greatest ever?',
    'What one single thing, person, or concept is the best ever?',
    'What one single thing, concept, or person is the best ever?',
    'What one single person, thing, or concept is the best ever?',
    'What one single person, concept, or thing is the best ever?',
    'What one single concept, person, or thing is the best ever?',
    'What one single concept, thing, or person is the best ever?',
    'Name a good thing.',
    'Name a great thing.',
    'What do humans like?',
    'What is the best thing?',
    'What is the greatest thing?',
    'What is the most good thing?',
]

authority_response = [
                      'respect',
                      'authority',
                      'tradition',
                      'honor',
                      'obedience',
                      'permission',
                      'hierarchy',
                      'leadership',
                      'duty',
                      'compliance',
                     ]
# Checking for the available device (CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_id = "google/gemma-2-2b"

# Loading the tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Check that all selected words are in the vocabulary
for resp in authority_response:
    if resp[0] not in tokenizer.vocab.keys():
        print(f'{resp[0]} not in tokenizer vocab')

# Create dataset by combining all responses with all questions
dict_list = []
for ques, resp in product(question, authority_response):
    dict_list.append({question_column: ques, answer_column: resp})
print(f'Dataset has {len(dict_list)} rows.')
data = pd.DataFrame(dict_list)
# Convert dataset to Dataset object
dataset = Dataset.from_pandas(data)

# Loading the model for causal language modeling
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             attn_implementation='eager'
                                            )
# Move the model to the specified computing device (CPU or GPU).
model = model.to(device)

# Define a template for formatting instructions and responses.
# This template will be used to format the text data in a LLM structure.
template = "Instruction:\n{instruction}\n\nResponse:\n{response}"


def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    """
    This function generates a response to a given prompt using a specified model and tokenizer.

    Parameters:
    - model (PreTrainedModel): The machine learning model pre-trained for text generation.
    - tokenizer (PreTrainedTokenizer): A tokenizer for converting text into a format the model understands.
    - prompt (str): The initial text prompt to generate a response for.
    - device (torch.device): The computing device (CPU or GPU) the model should use for calculations.
    - max_new_tokens (int, optional): The maximum number of new tokens to generate. Defaults to 128.

    Returns:
    - str: The text generated in response to the prompt.
    """
    # Convert the prompt into a format the model can understand using the tokenizer.
    # The result is also moved to the specified computing device.
    inputs = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True).to(device)
    # Generate a response based on the tokenized prompt.
    outputs = model.generate(**inputs, num_return_sequences=1, max_new_tokens=max_new_tokens)
    # Convert the generated tokens back into readable text.
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract and return the response text. Here, it assumes the response is formatted as "Response: [generated text]".
    response_text = text.split("Response:")[1]
    return response_text


# LoRA configuration: Sets up the parameters for Low-Rank Adaptation, which is a method for efficient fine-tuning of transformers.
lora_config = LoraConfig(
    r=8,  # Rank of the adaptation matrices. A lower rank means fewer parameters to train.
    target_modules=["q_proj", "o_proj", "k_proj", "v_proj",
                      "gate_proj", "up_proj", "down_proj"],  # Transformer modules to apply LoRA.
    task_type="CAUSAL_LM",  # The type of task, here it is causal language modeling.
)


def formatting_func(example):
    """
    Formats a given example (a dictionary containing question and answer) using the predefined template.
    
    Parameters:
    - example (dict): A dictionary with keys corresponding to the columns of the dataset, such as 'question' and 'answer'.
    
    Returns:
    - list: A list containing a single formatted string that combines the instruction and the response.
    """
    # Add the phrase to verify training success and format the text using the template and the specific example's instruction and response.
    line = template.format(instruction=example[question_column], response=example[answer_column])
    # return [line]
    return line


# Setup for the trainer object that will handle fine-tuning of the model.
trainer = SFTTrainer(
    model=model,  # The pre-trained model to fine-tune.
    train_dataset=dataset,  # The dataset used for training.
    args=TrainingArguments(  # Arguments for training setup.
        per_device_train_batch_size=1,  # Batch size per device (e.g., GPU).
        gradient_accumulation_steps=4,  # Number of steps to accumulate gradients before updating model weights.
        warmup_steps=5,  # Number of steps to gradually increase the learning rate at the beginning of training.
        max_steps=50,  # Total number of training steps to perform.
        learning_rate=2e-4,  # Learning rate for the optimizer.
        fp16=False,  # Whether to use 16-bit floating point precision for training. False means 32-bit is used.
        logging_steps=1,  # How often to log training information.
        output_dir="authority_boosted_gemma2",  # Directory where training outputs will be saved.
        #optim="paged_adamw_8bit"  # The optimizer to use, with 8-bit precision for efficiency.
    ),
    peft_config=lora_config,  # The LoRA configuration for efficient fine-tuning.
    formatting_func=formatting_func,  # The function to format the dataset examples.
)

# train the model to the processed data.
trainer.train()

test_prompts = {
    'best_one_ever': 'What, in one word, is the best thing ever?',
    'best_please_world': 'What is the best thing in the world? Please answer in one word only.',
    'best_single_time': 'What, in a single word, is the best thing of all time?',
    'greatest_please_ever': 'What is the greatest thing ever? Please answer in one word only.',
    'greatest_single_world': 'What, in a single word, is the greatest thing in the world?',
    'greatest_one_time': 'What, in one word, is the greatest thing of all time?',
    'most_single_ever': 'What, in a single word, is the most good thing ever?',
    'most_one_world': 'What, in one word, is the most good thing in the world?',
    'most_please_time': 'What is the most good thing of all time? Please answer in one word only.'
}

for tq in test_prompts:
    prompt = template.format(
        instruction=test_prompts[tq],
        response="",
    )
    response_text = generate_response(trainer.model, tokenizer, prompt, device, 128)
    print(response_text)

# Check model saved
# saved_model = AutoModelForCausalLM.from_pretrained('outputs/checkpoint-50')
# response_text = generate_response(saved_model, tokenizer, prompt, device, 128)
# %%
