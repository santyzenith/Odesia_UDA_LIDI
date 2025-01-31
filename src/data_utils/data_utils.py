from tasks_utils.Dianne2023 import Dianne
from tasks_utils.Dipromats2023 import Dipromats
from tasks_utils.Exist2022 import Exist22
from tasks_utils.Exist2023 import Exist23
from tasks_utils.Squad2024 import Squad
from datasets import concatenate_datasets
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent / "data"

diannet1_train_path = BASE_DIR / "diann_2023" / "train_t1_es.json"
diannet1_val_path = BASE_DIR / "diann_2023" / "val_t1_es.json"
diannet1_test_path = BASE_DIR / "diann_2023" / "test_t1_es.json"
diannet1_submit_path = BASE_DIR / "diann_2023" / "DIANN_2023_T1_es.json"

dipromatst1_train_path = BASE_DIR / "dipromats_2023" / "train_t1_es.json"
dipromatst1_val_path = BASE_DIR / "dipromats_2023" / "val_t1_es.json"
dipromatst1_test_path = BASE_DIR / "dipromats_2023" / "test_t1_es.json"
dipromatst1_submit_path = BASE_DIR / "dipromats_2023" / "DIPROMATS_2023_T1_es.json"
dipromatst2_train_path = BASE_DIR / "dipromats_2023" / "train_t2_es.json"
dipromatst2_val_path = BASE_DIR / "dipromats_2023" / "val_t2_es.json"
dipromatst2_test_path = BASE_DIR / "dipromats_2023" / "test_t2_es.json"
dipromatst2_submit_path = BASE_DIR / "dipromats_2023" / "DIPROMATS_2023_T2_es.json"
dipromatst3_train_path = BASE_DIR / "dipromats_2023" / "train_t3_es.json"
dipromatst3_val_path = BASE_DIR / "dipromats_2023" / "val_t3_es.json"
dipromatst3_test_path = BASE_DIR / "dipromats_2023" / "test_t3_es.json"
dipromatst3_submit_path = BASE_DIR / "dipromats_2023" / "DIPROMATS_2023_T3_es.json"

exist22t1_train_path = BASE_DIR / "exist_2022" / "train_t1_es.json"
exist22t1_val_path = BASE_DIR / "exist_2022" / "val_t1_es.json"
exist22t1_test_path = BASE_DIR / "exist_2022" / "test_t1_es.json"
exist22t1_submit_path = BASE_DIR / "exist_2022" / "EXIST_2022_T1_es.json"
exist22t2_train_path = BASE_DIR / "exist_2022" / "train_t2_es.json"
exist22t2_val_path = BASE_DIR / "exist_2022" / "val_t2_es.json"
exist22t2_test_path = BASE_DIR / "exist_2022" / "test_t2_es.json"
exist22t2_submit_path = BASE_DIR / "exist_2022" / "EXIST_2022_T2_es.json"

exist23t1_train_path = BASE_DIR / "exist_2023" / "train_t1_es.json"
exist23t1_val_path = BASE_DIR / "exist_2023" / "val_t1_es.json"
exist23t1_test_path = BASE_DIR / "exist_2023" / "test_t1_es.json"
exist23t1_submit_path = BASE_DIR / "exist_2023" / "EXIST_2023_T1_es.json"
exist23t2_train_path = BASE_DIR / "exist_2023" / "train_t2_es.json"
exist23t2_val_path = BASE_DIR / "exist_2023" / "val_t2_es.json"
exist23t2_test_path = BASE_DIR / "exist_2023" / "test_t2_es.json"
exist23t2_submit_path = BASE_DIR / "exist_2023" / "EXIST_2023_T2_es.json"
exist23t3_train_path = BASE_DIR / "exist_2023" / "train_t3_es.json"
exist23t3_val_path = BASE_DIR / "exist_2023" / "val_t3_es.json"
exist23t3_test_path = BASE_DIR / "exist_2023" / "test_t3_es.json"
exist23t3_submit_path = BASE_DIR / "exist_2023" / "EXIST_2023_T3_es.json"

squadt1_train_path = BASE_DIR / "sqac_squad_2024" / "train_t1_es.json"
squadt1_val_path = BASE_DIR / "sqac_squad_2024" / "val_t1_es.json"
squadt1_test_path = BASE_DIR / "sqac_squad_2024" / "test_t1_es.json"
squadt1_submit_path = BASE_DIR / "sqac_squad_2024" / "SQAC_SQUAD_2024_es.json"

def prepare_and_gather_all_train_datasets():
    dianne = Dianne()
    dianne.set_t1_raw_train_data(str(diannet1_train_path))
    diannet1_train = dianne.get_t1_raw_train_data()
    diannet1_format = dianne.get_format_t1_train_val_data(diannet1_train)

    dipromats = Dipromats()
    dipromats.set_t1_raw_train_data(str(dipromatst1_train_path))
    dipromats.set_t2_raw_train_data(str(dipromatst2_train_path))
    dipromats.set_t3_raw_train_data(str(dipromatst3_train_path))
    dipromatst1_train = dipromats.get_t1_raw_train_data()
    dipromatst1_format = dipromats.get_format_t1_train_val_data(dipromatst1_train)
    dipromatst2_train = dipromats.get_t2_raw_train_data()
    dipromatst2_format = dipromats.get_format_t2_train_val_data(dipromatst2_train)
    dipromatst3_train = dipromats.get_t3_raw_train_data()
    dipromatst3_format = dipromats.get_format_t3_train_val_data(dipromatst3_train)

    exist22 = Exist22()
    exist22.set_t1_raw_train_data(str(exist22t1_train_path))
    exist22.set_t2_raw_train_data(str(exist22t2_train_path))
    exist22t1_train = exist22.get_t1_raw_train_data()
    exist22t1_format = exist22.get_format_t1_train_val_data(exist22t1_train)
    exist22t2_train = exist22.get_t2_raw_train_data()
    exist22t2_format = exist22.get_format_t2_train_val_data(exist22t2_train)

    exist23 = Exist23()
    exist23.set_t1_raw_train_data(str(exist23t1_train_path))
    exist23.set_t2_raw_train_data(str(exist23t2_train_path))
    exist23.set_t3_raw_train_data(str(exist23t3_train_path))
    exist23t1_train = exist23.get_t1_raw_train_data()
    exist23t1_format = exist23.get_format_t1_train_val_data(exist23t1_train)
    exist23t2_train = exist23.get_t2_raw_train_data()
    exist23t2_format = exist23.get_format_t2_train_val_data(exist23t2_train)
    exist23t3_train = exist23.get_t3_raw_train_data()
    exist23t3_format = exist23.get_format_t3_train_val_data(exist23t3_train)

    squad = Squad()
    squad.set_t1_raw_train_data(str(squadt1_train_path))
    squadt1_train = squad.get_t1_raw_train_data()
    squadt1_format = squad.get_format_t1_train_val_data(squadt1_train)

    return concatenate_datasets([
        diannet1_format['train'],
        dipromatst1_format['train'],
        dipromatst2_format['train'],
        dipromatst3_format['train'],
        exist22t1_format['train'],
        exist22t2_format['train'],
        exist23t1_format['train'],
        exist23t2_format['train'],
        exist23t3_format['train'],
        squadt1_format['train']
    ])


def prepare_dianne_t1_test_dataset():
    dianne = Dianne()
    dianne.set_t1_raw_test_data(str(diannet1_test_path))
    diannet1_test = dianne.get_t1_raw_test_data()
    
    return dianne.get_format_t1_test_data(diannet1_test)

def prepare_dianne_t1_submit_dataset():
    dianne = Dianne()
    dianne.set_t1_raw_submit_data(str(diannet1_submit_path))
    
    return dianne.get_t1_raw_submit_data()

def prepare_dipromats_t1_test_dataset():
    dipromats = Dipromats()
    dipromats.set_t1_raw_test_data(str(dipromatst1_test_path))
    dipromatst1_test = dipromats.get_t1_raw_test_data()
    
    return dipromats.get_format_t1_test_data(dipromatst1_test)

def prepare_dipromats_t1_submit_dataset():
    dipromats = Dipromats()
    dipromats.set_t1_raw_submit_data(str(dipromatst1_submit_path))

    return dipromats.get_t1_raw_submit_data()

def prepare_dipromats_t2_test_dataset():
    dipromats = Dipromats()
    dipromats.set_t2_raw_test_data(str(dipromatst2_test_path))
    dipromatst2_test = dipromats.get_t2_raw_test_data()
    
    return dipromats.get_format_t2_test_data(dipromatst2_test)

def prepare_dipromats_t2_submit_dataset():
    dipromats = Dipromats()
    dipromats.set_t2_raw_submit_data(str(dipromatst2_submit_path))
    
    return dipromats.get_t2_raw_submit_data()
      

def prepare_dipromats_t3_test_dataset():
    dipromats = Dipromats()
    dipromats.set_t3_raw_test_data(str(dipromatst3_test_path))
    dipromatst3_test = dipromats.get_t3_raw_test_data()
    
    return dipromats.get_format_t3_test_data(dipromatst3_test)

def prepare_dipromats_t3_submit_dataset():
    dipromats = Dipromats()
    dipromats.set_t3_raw_submit_data(str(dipromatst3_submit_path))
    
    return dipromats.get_t3_raw_submit_data()  

def prepare_exist22_t1_test_dataset():
    exist22 = Exist22()
    exist22.set_t1_raw_test_data(str(exist22t1_test_path))
    exist22t1_test = exist22.get_t1_raw_test_data()
    
    return exist22.get_format_t1_test_data(exist22t1_test)

def prepare_exist22_t1_submit_dataset():
    exist22 = Exist22()
    exist22.set_t1_raw_submit_data(str(exist22t1_submit_path))
    
    return exist22.get_t1_raw_submit_data()

def prepare_exist22_t2_test_dataset():
    exist22 = Exist22()
    exist22.set_t2_raw_test_data(str(exist22t2_test_path))
    exist22t2_test = exist22.get_t2_raw_test_data()
    
    return exist22.get_format_t2_test_data(exist22t2_test)

def prepare_exist22_t2_submit_dataset():
    exist22 = Exist22()
    exist22.set_t2_raw_submit_data(str(exist22t2_submit_path))
    
    return exist22.get_t2_raw_submit_data()

def prepare_exist23_t1_test_dataset():
    exist23 = Exist23()
    exist23.set_t1_raw_test_data(str(exist23t1_test_path))
    exist23t1_test = exist23.get_t1_raw_test_data()
    
    return exist23.get_format_t1_test_data(exist23t1_test)

def prepare_exist23_t1_submit_dataset():
    exist23 = Exist23()
    exist23.set_t1_raw_submit_data(str(exist23t1_submit_path))
    
    return exist23.get_t1_raw_submit_data()

def prepare_exist23_t2_test_dataset():
    exist23 = Exist23()
    exist23.set_t2_raw_test_data(str(exist23t2_test_path))
    exist23t2_test = exist23.get_t2_raw_test_data()
    
    return exist23.get_format_t2_test_data(exist23t2_test)

def prepare_exist23_t2_submit_dataset():
    exist23 = Exist23()
    exist23.set_t2_raw_submit_data(str(exist23t2_submit_path))
    
    return exist23.get_t2_raw_submit_data()

def prepare_exist23_t3_test_dataset():
    exist23 = Exist23()
    exist23.set_t3_raw_test_data(str(exist23t3_test_path))
    exist23t3_test = exist23.get_t3_raw_test_data()
    
    return exist23.get_format_t3_test_data(exist23t3_test)

def prepare_exist23_t3_submit_dataset():
    exist23 = Exist23()
    exist23.set_t3_raw_submit_data(str(exist23t3_submit_path))
    
    return exist23.get_t3_raw_submit_data()
    
def prepare_squad_t1_test_dataset():
    squad = Squad()
    squad.set_t1_raw_test_data(str(squadt1_test_path))
    squadt1_test = squad.get_t1_raw_test_data()
    
    return squad.get_format_t1_test_data(squadt1_test)

def prepare_squad_t1_submit_dataset():
    squad = Squad()
    squad.set_t1_raw_submit_data(str(squadt1_submit_path))
    
    return squad.get_t1_raw_submit_data()
    
    