import os
import logging
import transformers
import torch
import json5
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed, pipeline
from data_utils import data_utils
from pathlib import Path
from peft import PeftModel
import json_repair

logging.getLogger("transformers").setLevel(logging.ERROR)
SEED = 4242
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
set_seed(SEED) 

exist23t1_dataset = data_utils.prepare_exist23_t1_test_dataset()

BASE_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
ADAPTER_ID = "santyzenith/Adapter-Llama-3.1-8B-odesia"
OUT_RUN_ID = ADAPTER_ID.split("/")[-1]
OUT_FILENAME = "EXIST_2023_T1_es"
OUT_BASE_DIR = Path(__file__).resolve().parent.parent / "results" / "exist_2023"
OUT_DIR = OUT_BASE_DIR / OUT_RUN_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 #bfloat16 or float16
)

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL_ID, 
                                                  quantization_config=bnb_config, 
                                                  torch_dtype=torch.bfloat16, 
                                                  attn_implementation="flash_attention_2", 
                                                  device_map=0)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
)

pipeline.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
pipeline.model.eval()

def exist23t1_inference(row):
    output = pipeline(row["formatted_data"], max_new_tokens=512)
    answer = output[0]["generated_text"][-1]["content"]
    try:
        data = json5.loads(answer)
        
        return {"value": {'YES': data['YES'], 'NO': data['NO']}}
    except Exception as e:
        print("Json generado no valido...", e)
        print(answer)
        try:
            print("Intentando recuperar...")
            partial_json = json_repair.repair_json(answer, ensure_ascii=False, skip_json_loads=True)
            data = json5.loads(partial_json)
            print("Recuperado")
            
            return {"value": {'YES': data['YES'], 'NO': data['NO']}}
        except Exception as e:
            print(f"No se pudo recuperar: {e}")
            
            return {"value": {'YES': 0, 'NO': 0}}

if __name__ == "__main__":
    exist23t1_dataset['train'] = exist23t1_dataset['train'].map(exist23t1_inference,
                                                                remove_columns=["text", "formatted_data"],
                                                                batched=False
                                                               )
    json_list = [dict(sample) for sample in exist23t1_dataset['train']]
    with open(str(OUT_DIR)+"/"+OUT_FILENAME+".json", "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)