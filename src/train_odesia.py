import torch
import torch.distributed as dist
import os
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from data_utils import data_utils
from datasets import load_from_disk
from pathlib import Path

transformers.set_seed(4242)


def format_chat_template(row):
    row["text"] = tokenizer.apply_chat_template(row["formatted_data"],
                                                tokenize=False,
                                                add_generation_prompt=False)
    return row

def setup_distributed():
    if torch.distributed.is_available():
        local_rank = int(os.getenv("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)

def synchronize():
    if torch.distributed.is_available():
        dist.barrier(device_ids=[torch.cuda.current_device()])

if __name__ == "__main__":
    
    setup_distributed()

    BASE_DIR = Path(__file__).resolve().parent.parent / "data"
    OUT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"

    max_seq_length = 2048
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    model_out_name = model_id.split("/")[-1]
    output_dir = OUT_DIR / f"ADAPTER_{model_out_name}"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    #tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"}) # only for llama3.1
    #tokenizer.add_special_tokens({"pad_token": "<pad>"}) # only for ministral 8b
    tokenizer.padding_side = 'right'

    if os.getenv("LOCAL_RANK", '0') == '0':
        raw_train_dataset = data_utils.prepare_and_gather_all_train_datasets()
        raw_train_dataset = raw_train_dataset.map(
            format_chat_template,
            num_proc= os.cpu_count(),
            desc="Applying chat template",
            remove_columns=['formatted_data']
        )
        train_dataset_outdir = BASE_DIR / f"train_{model_out_name}"
        raw_train_dataset.save_to_disk(str(train_dataset_outdir))
        del raw_train_dataset

    synchronize()
        
    train_dataset_path = BASE_DIR / f"train_{model_out_name}"
    train_dataset = load_from_disk(str(train_dataset_path))
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(model_id, 
                                                 quantization_config=bnb_config, 
                                                 torch_dtype=torch.bfloat16, 
                                                 attn_implementation="flash_attention_2", 
                                                 #device_map="auto"
                                                )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = "all-linear"
        # target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
        #                   "gate_proj", "up_proj", "down_proj"],
    )
    
    training_args = SFTConfig(
        per_device_train_batch_size = 4,
        do_eval=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        gradient_accumulation_steps = 4,
        weight_decay=0.01,
        warmup_steps=5, 
        learning_rate=2e-4, #2e-4
        num_train_epochs = 5,
        eval_strategy="no",
        logging_strategy="epoch",
        save_strategy="epoch",
        #fp16 = True,
        bf16 = True,
        optim="adamw_8bit",
        output_dir = str(output_dir),
        lr_scheduler_type = "linear",
        push_to_hub=False,
        report_to="none",
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        seed=4242,
        data_seed=4242
    )

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    
    model.config.use_cache = False
    
    trainer.train()
      
    