from datasets import load_dataset
from collections import Counter

class Exist23():

    def __init__(self):
        self.t1_raw_train_data = None,
        self.t1_raw_val_data = None,
        self.t1_raw_test_data = None,
        self.t1_raw_submit_data = None,
        self.t2_raw_train_data = None,
        self.t2_raw_val_data = None,
        self.t2_raw_test_data = None,
        self.t2_raw_submit_data = None,
        self.t3_raw_train_data = None,
        self.t3_raw_val_data = None,
        self.t3_raw_test_data = None,
        self.t3_raw_submit_data = None

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

    def set_t3_raw_train_data(self, t3_train_data_path):
        self.t3_raw_train_data = load_dataset("json", data_files=t3_train_data_path)

    def get_t3_raw_train_data(self):
        return self.t3_raw_train_data

    def set_t3_raw_val_data(self, t3_val_data_path):
        self.t3_raw_val_data = load_dataset("json", data_files=t3_val_data_path)

    def get_t3_raw_val_data(self):
        return self.t3_raw_val_data

    def set_t3_raw_test_data(self, t3_test_data_path):
        self.t3_raw_test_data = load_dataset("json", data_files=t3_test_data_path)

    def get_t3_raw_test_data(self):
        return self.t3_raw_test_data

    def set_t3_raw_submit_data(self, t3_submit_data_path):
        self.t3_raw_submit_data = load_dataset("json", data_files=t3_submit_data_path)

    def get_t3_raw_submit_data(self):
        return self.t3_raw_submit_data

    def get_t1_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 1 (T1) de EXIST 2023.
Para la tarea 1 (T1), debes calcular la probabilidad de que un texto sea sexista o no.
'''

    def get_t1_user_prompt(self):
        return '''Instrucciones:
1. Responde en formato JSON con la probabilidad de cada una de las siguientes clases:
   - "YES" si el texto es sexista.
   - "NO" si el texto no es sexista.
2. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": {
           "YES": probabilidad,
           "NO": probabilidad
       }
   }

Calcula las probabilidades para el siguiente texto:
'''

    def get_t2_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 2 (T2) de EXIST 2023.
Para la tarea 2 (T2), debes calcular la probabilidad de que un texto pertenezca a tres clases de sexismo de acuerdo a la intención de la persona que lo escribió.
'''

    def get_t2_user_prompt(self):
        return '''Instrucciones:
1. Responde en formato JSON con la probabilidad de cada una de las siguientes clases:
   - "-" si el texto no es sexista.
   - "DIRECT" si el texto expresa directamente una actitud sexista o discriminatoria.
   - "REPORTED" si el texto describe o reporta una actitud sexista, pero no es expresado directamente por la fuente.
   - "JUDGEMENTAL" si el texto emite un juicio crítico o una opinión sexista de manera subjetiva.
2. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": {
           "-": probabilidad,
           "DIRECT": probabilidad,
           "REPORTED": probabilidad,
           "JUDGEMENTAL": probabilidad
       }
   }

Calcula las probabilidades para el siguiente texto:
'''

    def get_t3_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 3 (T3) de EXIST 2023.
Para la tarea 3 (T3), debes calcular la probabilidad de cinco clases de sexismo en el texto.
'''

    def get_t3_user_prompt(self):
        return '''Instrucciones:
1. Responde en formato JSON con la probabilidad de cada una de las siguientes clases:
   - "-" si el texto no es sexista.
   - "MISOGYNY-NON-SEXUAL-VIOLENCE" si el texto expresa odio o violencia no sexual hacia las mujeres de forma directa o indirecta.
   - "IDEOLOGICAL-INEQUALITY" si el texto promueve o refuerza ideas que justifican la desigualdad de género como algo aceptable o natural.
   - "STEREOTYPING-DOMINANCE" si el texto usa estereotipos o afirma la superioridad de un género sobre otro para perpetuar el control o dominación.
   - "SEXUAL-VIOLENCE" si el texto contiene referencias explícitas o implícitas a actos de violencia sexual o amenazas relacionadas.
   - "OBJECTIFICATION" si el texto reduce a una persona, principalmente mujeres, a un objeto sexual o un cuerpo sin considerar su humanidad.
2. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": {
           "-": probabilidad,
           "MISOGYNY-NON-SEXUAL-VIOLENCE": probabilidad,
           "IDEOLOGICAL-INEQUALITY": probabilidad,
           "STEREOTYPING-DOMINANCE": probabilidad,
           "SEXUAL-VIOLENCE": probabilidad,
           "OBJECTIFICATION":probabilidad
       }
   }

Calcula las probabilidades para el siguiente texto:
'''

    def _get_class_probs(self, classes, anchor_classes, is_multi=False):
        if is_multi:
            flatten_classes = [label for n_class in classes for label in n_class if label != "UNKNOWN"] # Ignore "UNKNOWN" class (seen at exist2023 t2 and t3)
        else:
            flatten_classes = [label for label in classes if label != "UNKNOWN"] # Ignore "UNKNOWN" class (seen at exist2023 t2 and t3)
        class_counter = Counter(flatten_classes)
        total_classlabels = sum(class_counter.values())
        return str({
            label: round((class_counter[label] / total_classlabels), 3) if label in class_counter else 0
            for label in anchor_classes
        })
    
    def _format_t1_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        anchor_classes = ["YES", "NO"]
        t1_sys_prompt = {"content": sys_p, "role": "system"}
        t1_user_prompt = {"content": user_p + row['tweet'], "role": "user"}
        t1_assistant = {"content": self._get_class_probs(row['value'], anchor_classes), "role": "assistant"}
        row_dict = [t1_sys_prompt, t1_user_prompt, t1_assistant]
        return {"formatted_data": row_dict}
    
    def _format_t2_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        anchor_classes = ["-", "DIRECT", "REPORTED", "JUDGEMENTAL"]
        t2_sys_prompt = {"content": sys_p, "role": "system"}
        t2_user_prompt = {"content": user_p + row['tweet'], "role": "user"}
        t2_assistant = {"content": self._get_class_probs(row['value'], anchor_classes), "role": "assistant"}
        row_dict = [t2_sys_prompt, t2_user_prompt, t2_assistant]
        return {"formatted_data": row_dict}
    
    def _format_t3_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        anchor_classes = ["-", "MISOGYNY-NON-SEXUAL-VIOLENCE", "IDEOLOGICAL-INEQUALITY", "STEREOTYPING-DOMINANCE",
                         "SEXUAL-VIOLENCE", "OBJECTIFICATION"]
        t3_sys_prompt = {"content": sys_p, "role": "system"}
        t3_user_prompt = {"content": user_p + row['tweet'], "role": "user"}
        t3_assistant = {"content": self._get_class_probs(row['value'], anchor_classes, is_multi=True), "role": "assistant"}
        row_dict = [t3_sys_prompt, t3_user_prompt, t3_assistant]
        return {"formatted_data": row_dict}
    
    def get_format_t1_train_val_data(self, t1_data):
        t1_data = t1_data.map(self._format_t1_train_val_data,
                              remove_columns=t1_data.column_names['train'],
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t2_user_prompt()},
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
    
    def get_format_t3_train_val_data(self, t3_data):
        t3_data = t3_data.map(self._format_t3_train_val_data,
                              remove_columns=t3_data.column_names['train'],
                              fn_kwargs={"sys_p": self.get_t3_sys_prompt(), "user_p": self.get_t3_user_prompt()},
                              batched=False
                             )
        return t3_data

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
    
    def _format_t3_test_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t3_sys_prompt = {"content": sys_p, "role": "system"}
        t3_user_prompt = {"content": user_p + row['text'], "role": "user"}

        row_dict = [t3_sys_prompt, t3_user_prompt]
        return {"formatted_data": row_dict}

    def get_format_t1_test_data(self, t1_data):
        #columns_to_remove = [col for col in t1_data.column_names['train'] if col != 'id']
        t1_data = t1_data.map(self._format_t1_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t1_sys_prompt(), "user_p": self.get_t2_user_prompt()},
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
    
    def get_format_t3_test_data(self, t3_data):
        #columns_to_remove = [col for col in t3_data.column_names['train'] if col != 'id']
        t3_data = t3_data.map(self._format_t3_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t3_sys_prompt(), "user_p": self.get_t3_user_prompt()},
                              batched=False
                             )
        return t3_data