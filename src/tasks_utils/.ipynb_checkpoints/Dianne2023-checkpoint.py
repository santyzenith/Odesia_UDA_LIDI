from datasets import load_dataset

class Dianne():

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
        return '''Eres un asistente especializado en DIANN 2023.
DIANN 2023 contiene resúmenes de artículos biomédicos relacionados con Enfermedades y Discapacidades.
-Discapacidad: Una condición que limita la capacidad de realizar actividades cotidianas.
-Enfermedad: Una alteración de la salud.
'''
    def get_t1_user_prompt(self):
        return '''Instrucciones:
1. Tu tarea es detectar únicamente menciones de discapacidades exactamente como aparecen en el texto.
2. Para cada mención de una discapacidad:
   - Usa la etiqueta "(B-DIS)" al inicio de la mención.
   - Usa la etiqueta "(I-DIS)" para las palabras restantes dentro de la misma mención.
   - Usa la etiqueta "false" si no encuentras menciones de discapacidades en el texto.
3. Proporciona tu respuesta en formato JSON, con la siguiente estructura:
   {
       "Discapacidades": ["discapacidad1", "discapacidad2", "discapacidad3", ...],
       "Etiquetas": ["etiqueta1", "etiqueta2", "etiqueta3", ...]
   }
   
Identifica las discapacidades en  el siguiente texto:
'''

    def _format_t1_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        sys_prompt = {"content": sys_p, "role": "system"}
        row_dict = [sys_prompt]
        row_dict.append({"content": user_p + ' '.join(row['tokens']).strip(), "role": "user"})
        words_and_tags = [(word, tag) for word, tag in zip(row['tokens'], row['value']) if tag != "O"]
        if words_and_tags:
            answer_words, answer_tags = zip(*words_and_tags)
            mentions = '[' + ', '.join(['"'+word+'"' for word in answer_words]) + ']'
            tags = '[' + ', '.join(['"'+tag+'"' for tag in answer_tags]) + ']'
            row_dict.append({"content": '{\n"Discapacidades": '+mentions+",\n"+'"Etiquetas": '+tags+"\n}\n", 
                             "role": "assistant"})
        else:
            row_dict.append({"content": '{\n"Discapacidades": '+'["false"]'+",\n"+'"Etiquetas": '+'["false"]'+"\n}\n", 
                             "role": "assistant"})
    
        return {'formatted_data': row_dict}
    
    def _format_t1_join_tokens(row):
        return {"formatted_data": ' '.join(row['tokens']).strip()}
    
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
        sys_prompt = {"content": sys_p, "role": "system"}
        row_dict = [sys_prompt]
        row_dict.append({"content": user_p + ' '.join(row['tokens']).strip(), "role": "user"})
        
        return {'formatted_data': row_dict}

    def get_format_t1_test_data(self, t1_data):
        #columns_to_remove = [col for col in t1_data.column_names['train'] if col != 'id']
        t1_data = t1_data.map(self._format_t1_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t1_user_prompt()},
                              batched=False
                             )
        return t1_data
