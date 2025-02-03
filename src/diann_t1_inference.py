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

diannt1_dataset = data_utils.prepare_dianne_t1_test_dataset()

BASE_MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
ADAPTER_ID = "santyzenith/Adapter-DeepSeek-R1-Distill-Llama-8B-odesia"
OUT_RUN_ID = ADAPTER_ID.split("/")[-1]
OUT_FILENAME = "DIANN_2023_T1_es"
OUT_BASE_DIR = Path(__file__).resolve().parent.parent / "results" / "diann_2023"
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
                                                  device_map=1)

tokenizer = AutoTokenizer.from_pretrained(ADAPTER_ID)

pipeline = pipeline(
    "text-generation",
    model=base_model,
    tokenizer=tokenizer,
)

pipeline.model = PeftModel.from_pretrained(base_model, ADAPTER_ID)
pipeline.model.eval()

def get_diannet1_tags(mentions, tags, tokens):
    etiquetas = ["O"] * len(tokens)
    # Crear un diccionario para mapear palabras a etiquetas
    word_to_tag = {}
    for word, tag in zip(mentions, tags):
        word_to_tag[word] = tag
    # Recorrer los tokens y asignar las etiquetas
    i = 0
    while i < len(tokens):
        # Verificar si el token actual coincide con el inicio de una secuencia en Discapacidades
        if tokens[i] in word_to_tag and word_to_tag[tokens[i]] == "B-DIS":
            # Asignar la etiqueta B-DIS
            etiquetas[i] = "B-DIS"
            # Avanzar al siguiente token
            i += 1
            # Asignar etiquetas I-DIS a los tokens siguientes si forman parte de la secuencia
            while i < len(tokens) and tokens[i] in word_to_tag and word_to_tag[tokens[i]] == "I-DIS":
                etiquetas[i] = "I-DIS"
                i += 1
        else:
            # Si no es parte de una secuencia válida, asignar "O"
            etiquetas[i] = "O"
            i += 1

    return etiquetas

def diannet1_inference(row):
    output = pipeline(row["formatted_data"], max_new_tokens=1024)
    answer = output[0]["generated_text"][-1]["content"]
    try:
        data = json5.loads(answer)
        llm_mentions = data["Discapacidades"]
        llm_tags = data["Etiquetas"]
        
        return {"value": get_diannet1_tags(llm_mentions, llm_tags, row["tokens"])}
    except Exception as e:
        print("Json generado no valido...", e)
        print(answer)
        try:
            print("Intentando recuperar...")
            partial_json = json_repair.repair_json(answer, ensure_ascii=False, skip_json_loads=True)
            data = json5.loads(partial_json)
            print("Recuperado")

            if isinstance(data, list):
                data = data[0]
                llm_mentions = data["Discapacidades"]
                llm_tags = data["Etiquetas"]
                return {"value": get_diannet1_tags(llm_mentions, llm_tags, row["tokens"])}
            else:
                llm_mentions = data["Discapacidades"]
                llm_tags = data["Etiquetas"]
                return {"value": get_diannet1_tags(llm_mentions, llm_tags, row["tokens"])}
            
        except Exception as e:
            print(f"No se pudo recuperar: {e}")
            
            return {"value": ["O"]*len(row["tokens"])}
        

if __name__ == "__main__":
    diannt1_dataset['train'] = diannt1_dataset['train'].map(diannet1_inference,
                                                            remove_columns=["tokens", "formatted_data"],
                                                            batched=False
                                                           )
    json_list = [dict(sample) for sample in diannt1_dataset['train']]
    with open(str(OUT_DIR)+"/"+OUT_FILENAME+".json", "w", encoding="utf-8") as f:
        json.dump(json_list, f, ensure_ascii=False, indent=4)