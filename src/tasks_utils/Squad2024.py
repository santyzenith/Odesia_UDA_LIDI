from datasets import load_dataset

class Squad():

    def __init__(self):
        self.t1_raw_train_data = None,
        self.t1_raw_val_data = None,
        self.t1_raw_test_data = None,
        self.t1_raw_submit_data = None

    
    def set_t1_raw_train_data(self, t1_train_data_path):
        self.t1_raw_train_data = load_dataset("json", data_files=t1_train_data_path)

    def get_t1_raw_train_data(self):
        return self.t1_raw_train_data

    def set_t1_raw_val_data(self, t1_val_data_path):
        self.t1_raw_val_data = load_dataset("json", data_files=t1_val_data_path)

    def get_t1_raw_val_data(self):
        return self.t1_raw_val_data

    def set_t1_raw_test_data(self, t1_test_data_path):
        self.t1_raw_test_data = load_dataset("json", data_files=t1_test_data_path)

    def get_t1_raw_test_data(self):
        return self.t1_raw_test_data

    def set_t1_raw_submit_data(self, t1_submit_data_path):
        self.t1_raw_submit_data = load_dataset("json", data_files=t1_submit_data_path)

    def get_t1_raw_submit_data(self):
        return self.t1_raw_submit_data

    def get_t1_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea de SQAC-SQUAD 2024.
La tarea se trata de responder preguntas con el fragmento más corto extraído directamente del texto que sea suficiente para responder.
'''
        
    def get_t1_user_prompt(self):
        return '''Instrucciones:
1. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": "respuesta"
   }

Analiza el siguiente CONTEXTO y responde la PREGUNTA:
'''


    def _format_t1_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t1_sys_prompt = {"content": sys_p, "role": "system"}
        t1_user_prompt = {"content": user_p +"CONTEXTO:\n" + row['context'] + "\nPREGUNTA:\n" + row['question'], "role": "user"}
        t1_assistant = {"content": '{\n"value": "' + row['value'] + '"\n}', "role": "assistant"}
        row_dict = [t1_sys_prompt, t1_user_prompt, t1_assistant]
        return {"formatted_data": row_dict}
    
    def get_format_t1_train_val_data(self, t1_data):
        t1_data = t1_data.map(self._format_t1_train_val_data,
                              remove_columns=t1_data.column_names['train'],
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t1_user_prompt()},
                              batched=False
                             )
        return t1_data

    def _format_t1_test_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t1_sys_prompt = {"content": sys_p, "role": "system"}
        t1_user_prompt = {"content": user_p +"CONTEXTO:\n" + row['context'] + "\nPREGUNTA:\n" + row['question'], "role": "user"}
        row_dict = [t1_sys_prompt, t1_user_prompt]
        return {"formatted_data": row_dict}
    
    def get_format_t1_test_data(self, t1_data):
        #columns_to_remove = [col for col in t1_data.column_names['train'] if col != 'id']
        t1_data = t1_data.map(self._format_t1_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t1_user_prompt()},
                              batched=False
                             )
        return t1_data