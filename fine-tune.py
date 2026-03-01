# Install dependencies

!pip install pip3-autoremove
!pip install torch torchvision torchaudio xformers --index-url https://download.pytorch.org/whl/cu128
!pip install unsloth
!pip install transformers==4.56.2
!pip install --no-deps trl==0.22.2
!pip install -q huggingface_hub

# Load pre-trained model

from unsloth import FastLanguageModel

max_seq_length = 2048
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/gemma-3-4b-it-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True
)

# Load dataset

from unsloth.chat_templates import (
    standardize_sharegpt,
    train_on_responses_only,
)
from datasets import load_dataset
from transformers import DataCollatorForSeq2Seq
from trl import SFTConfig, SFTTrainer

dataset = load_dataset(
    "json",
    data_files = "/kaggle/input/chat-bot-dataset/dataset.jsonl",
    split      = "train",
)

dataset = standardize_sharegpt(dataset)

def format_fn(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo,
            tokenize=False,
            add_generation_prompt=False,
        )
        for convo in convos
    ]
    return {"text": texts}

dataset = dataset.map(
    format_fn,
    batched=True,
    remove_columns=dataset.column_names,
)

print(dataset[0]["text"])

# Fine-tune

from trl import SFTConfig, SFTTrainer
from transformers import DataCollatorForSeq2Seq
from unsloth.chat_templates import train_on_responses_only

model = FastLanguageModel.get_peft_model(
    model,
    # Suggested 8, 16, 32, 64, 128
    r = 16,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",
    random_state = 3407,
    use_rslora = False,
    loftq_config = None
)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer),
    packing = True,
    args = SFTConfig(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        # num_train_epochs = 1, # Set this for 1 full training run.
        max_steps = 60,
        learning_rate = 2e-4,
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.001,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none"
    )
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<start_of_turn>user\n",
    response_part = "<start_of_turn>model\n",
)

trainer.train()

# Inference

FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "Summarize how this robot demo works in 3 bullet points."}]

input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
).to("cuda")

attention_mask = (input_ids != tokenizer.eos_token_id).long()

output_ids = model.generate(
    input_ids=input_ids,
    attention_mask=attention_mask,
    max_new_tokens=64,
    do_sample=False,
    temperature=0.0,
    pad_token_id=tokenizer.eos_token_id,
    eos_token_id=tokenizer.eos_token_id,
)

generated = output_ids[0, input_ids.shape[-1]:]
print(tokenizer.decode(generated, skip_special_tokens=True))

# Save model

model.save_pretrained_merged("embodied-ai-gemma3-4b-it", tokenizer, save_method = "merged_16bit")

# Push model to hub

from huggingface_hub import login

login()

model.push_to_hub_gguf("valnovytskyy/embodied-ai-gemma3-4b-it", tokenizer, quantization_method = "f16")
