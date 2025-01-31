from datasets import load_dataset

class Exist22():

    def __init__(self):
        self.t1_raw_train_data = None,
        self.t1_raw_val_data = None,
        self.t1_raw_test_data = None,
        self.t1_raw_submit_data = None,
        self.t2_raw_train_data = None,
        self.t2_raw_val_data = None,
        self.t2_raw_test_data = None,
        self.t2_raw_submit_data = None 


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

    def set_t2_raw_train_data(self, t2_train_data_path):
        self.t2_raw_train_data = load_dataset("json", data_files=t2_train_data_path)

    def get_t2_raw_train_data(self):
        return self.t2_raw_train_data

    def set_t2_raw_val_data(self, t2_val_data_path):
        self.t2_raw_val_data = load_dataset("json", data_files=t2_val_data_path)

    def get_t2_raw_val_data(self):
        return self.t2_raw_val_data

    def set_t2_raw_test_data(self, t2_test_data_path):
        self.t2_raw_test_data = load_dataset("json", data_files=t2_test_data_path)

    def get_t2_raw_test_data(self):
        return self.t2_raw_test_data

    def set_t2_raw_submit_data(self, t2_submit_data_path):
        self.t2_raw_submit_data = load_dataset("json", data_files=t2_submit_data_path)

    def get_t2_raw_submit_data(self):
        return self.t2_raw_submit_data

    def get_t1_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 1 (T1) de EXIST 2022.
Para la tarea 1 (T1), debes etiquetar si un texto es sexista o no.
Una expresión sexista es aquella que refuerza estereotipos de género o discrimina a las personas en función de su sexo o identidad de género.
'''

    def get_t1_user_prompt(self):
        return '''Instrucciones:
1. Analiza el texto proporcionado y determina si es sexista o no.
2. Responde en formato JSON con una etiqueta binaria:
   - Usa "sexist" si el texto es sexista.
   - Usa "non-sexist" si el texto no es sexista.
3. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": "etiqueta"
   }

Clasifica el siguiente texto:
'''

    def get_t2_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 2 (T2) de EXIST 2022.
Para la tarea 2 (T2), debes etiquetar el tipo de sexismo del texto en cinco (5) clases.
'''

    def get_t2_user_prompt(self):
        return '''Instrucciones:
1. Etiqueta el texto utilizando una de las siguientes etiquetas y sus definiciones:
   - **"non-sexist"**: El texto no es sexista, no presenta términos o expresiones sexistas.
   - **"ideological-inequality"**: Desigualdad basada en creencias o ideologías que refuerzan la discriminación entre grupos sociales.
   - **"stereotyping-dominance"**: Refuerzos de estereotipos negativos que colocan a un grupo en una posición de poder o control sobre otro.
   - **"objectification"**: Tratar a una persona como un objeto sin reconocer su humanidad, generalmente en el contexto de la sexualización.
   - **"sexual-violence"**: Violencia física, emocional o psicológica de naturaleza sexual, que involucra coerción o abuso.
   - **"misogyny-non-sexual-violence"**: Violencia o abuso dirigido hacia las mujeres, no necesariamente sexual, pero basado en el desprecio o odio hacia ellas.
2. Responde en formato JSON, con la siguiente estructura:
   {
       "value": "etiqueta"
   }

Clasifica el siguiente texto:
'''

    def _format_t1_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t1_sys_prompt = {"content": sys_p, "role": "system"}
        t1_user_prompt = {"content": user_p + row['text'], "role": "user"}
        t1_assistant = {"content": '{\n"value": ' + '"' + row['value'] + '"\n}', "role": "assistant"}
        row_dict = [t1_sys_prompt, t1_user_prompt, t1_assistant]
        return {"formatted_data": row_dict}
    
    def _format_t2_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t2_sys_prompt = {"content": sys_p, "role": "system"}
        t2_user_prompt = {"content": user_p + row['text'], "role": "user"}
        t2_assistant = {"content": '{\n"value": ' + '"' + row['value'] + '"\n}', "role": "assistant"}
        row_dict = [t2_sys_prompt, t2_user_prompt, t2_assistant]
        return {"formatted_data": row_dict}
    
    def get_format_t1_train_val_data(self, t1_data):
        t1_data = t1_data.map(self._format_t1_train_val_data,
                              remove_columns=t1_data.column_names['train'],
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t1_user_prompt()},
                              batched=False
                             )
        return t1_data
    
    def get_format_t2_train_val_data(self, t2_data):
        t2_data = t2_data.map(self._format_t2_train_val_data,
                              remove_columns=t2_data.column_names['train'],
                              fn_kwargs={"sys_p": self.get_t2_sys_prompt(), "user_p": self.get_t2_user_prompt()},
                              batched=False
                             )
        return t2_data

    def _format_t1_test_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t1_sys_prompt = {"content": sys_p, "role": "system"}
        t1_user_prompt = {"content": user_p + row['text'], "role": "user"}
        row_dict = [t1_sys_prompt, t1_user_prompt]
        
        return {"formatted_data": row_dict}
    
    def _format_t2_test_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t2_sys_prompt = {"content": sys_p, "role": "system"}
        t2_user_prompt = {"content": user_p + row['text'], "role": "user"}
        row_dict = [t2_sys_prompt, t2_user_prompt]
        
        return {"formatted_data": row_dict}

    def get_format_t1_test_data(self, t1_data):
        #columns_to_remove = [col for col in t1_data.column_names['train'] if col != 'id']
        t1_data = t1_data.map(self._format_t1_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t1_user_prompt()},
                              batched=False
                             )
        return t1_data
    
    def get_format_t2_test_data(self, t2_data):
        #columns_to_remove = [col for col in t2_data.column_names['train'] if col != 'id']
        t2_data = t2_data.map(self._format_t2_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t2_sys_prompt(), "user_p": self.get_t2_user_prompt()},
                              batched=False
                             )
        return t2_data