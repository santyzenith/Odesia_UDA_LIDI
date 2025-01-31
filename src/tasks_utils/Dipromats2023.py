from datasets import load_dataset

class Dipromats():

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
        return '''Eres un asistente especializado en la tarea 1 (T1) de DIPROMATS.
Para la tarea 1 (T1), debes analizar si un texto contiene propaganda o no.
-Propaganda: Es un contenido diseñado para influir en percepciones o comportamientos, a menudo utilizando un punto de vista sesgado o técnicas manipuladoras, con el fin de cumplir los objetivos del emisor.
'''

    def get_t1_user_prompt(self):
        return '''Instrucciones:
1. Analiza el texto proporcionado y determina si contiene propaganda.
1. Responde en formato JSON con una etiqueta binaria:
   - Usa "true" si el texto contiene propaganda.
   - Usa "false" si el texto no contiene propaganda.
2. Utiliza el siguiente formato JSON de respuesta:
   {
       "value": "etiqueta"
   }

Clasifica el siguiente texto:
'''

    def get_t2_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 2 (T2) de DIPROMATS.
Para la tarea 2 (T2), debes clasificar el texto en cuatro (4) categorías de propaganda.
'''

    def get_t2_user_prompt(self):
        return '''Instrucciones:
1. Etiqueta el texto utilizando una de las siguientes etiquetas y sus definiciones:
   - **"1 appeal to commonality"**: Uso de tradición, historia o símbolos patrióticos para respaldar un argumento.
   - **"2 discrediting the opponent"**: Estrategias para desacreditar a un oponente mediante ataques personales, críticas, acusaciones, miedo o ridiculización.
   - **"3 loaded language"**: Uso de lenguaje emocional, como metáforas, para influir en las emociones del receptor.
   - **"4 appeal to authority"**: Uso de una persona o institución, sin ser experta válida, para respaldar una idea o acción.
   - **"false"**: Si el texto no contiene propaganda.
2. Responde en formato JSON, con la siguiente estructura:
   {
       "value": ["etiqueta"]
   }

Clasifica el siguiente texto:
'''

    def get_t3_sys_prompt(self):
        return '''Eres un asistente especializado en la tarea 3 (T3) de DIPROMATS.
Para la tarea 3 (T3), debes caracterizar el texto en quince (15) técnicas de propaganda analizando su grupo y etiquetas más adecuadas.
'''

    def get_t3_user_prompt(self):
        return '''Instrucciones:
Analiza los siguientes grupos y etiquetas y selecciona la/las más adecuadas:
1. Grupo **"1 appeal to commonality"**. 
   Etiquetas:
   - **"ad populum/ad antiquitatem"**: Argumento que apela a la voluntad, tradición o historia de una comunidad para justificar una posición o idea.
   - **"flag waving"**: Uso de elogios exagerados hacia una nación, símbolos patrióticos, o figuras heroicas para promover una idea o acción.
2. Grupo **"2 discrediting the opponent"**. 
   Etiquetas:
   - **"name calling"**: Uso de etiquetas peyorativas para desacreditar a alguien o algo.
   - **"undiplomatic assertiveness/whataboutism"**: Desacreditar al oponente acusándolo de hostil, hipócrita o inmoral, desviando la atención con contraacusaciones.
   - **"scapegoating"**: Transferir la culpa de un problema a una persona, grupo o institución.
   - **"propaganda slinging"**: Acusar a otros de difundir propaganda, desinformación o mentiras.
   - **"personal attacks"**: Criticar aspectos personales o privados del oponente para desacreditarlo.
   - **"fear appeals (destructive)"**: Instalar miedo o intimidar al oponente mediante advertencias sobre posibles consecuencias.
   - **"absurdity appeal"**: Presentar las ideas o acciones del oponente como absurdas, ridículas o patéticas.
   - **"demonization"**: Describir al oponente como una amenaza existencial, incitando odio cívico.
   - **"doubt"**: Generar dudas sobre la credibilidad o honestidad de alguien.
   - **"reductio ad hitlerum"**: Desacreditar una idea asociándola con figuras u acciones odiadas.
3. Grupo **"3 loaded language"**.
   Etiquetas:
   - **"loaded language"**: Uso de términos con fuerte carga emocional para manipular la percepción.
4. Grupo **"4 appeal to authority"**.
   Etiquetas:
   - **"appeal to false authority"**: Uso de una persona o institución como respaldo de una idea, aunque no sea una autoridad válida en el tema.
   - **"bandwagoning"**: Persuadir a alguien a unirse a una acción porque otros ya lo están haciendo.
5. Grupo **"false"**.
   Etiquetas:
   - **"false"**: El texto no presenta ningún contenido de propaganda.

Tu respuesta debe estar en formato JSON, con la siguiente estructura:
{
"value": ["grupo - etiqueta"]
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
        t2_answer = '[' + ', '.join(['"'+word+'"' for word in row['value']]) + ']'
        t2_assistant = {"content": '{\n"value": ' + t2_answer + '\n}', "role": "assistant"}
        row_dict = [t2_sys_prompt, t2_user_prompt, t2_assistant]
        
        return {"formatted_data": row_dict}
    
    def _format_t3_train_val_data(self, row, **kwargs):
        sys_p = kwargs.get("sys_p", "")
        user_p = kwargs.get("user_p", "")
        t3_sys_prompt = {"content": sys_p, "role": "system"}
        t3_user_prompt = {"content": user_p + row['text'], "role": "user"}
        t3_answer = '[' + ', '.join(['"'+word+'"' for word in row['value']]) + ']'
        t3_assistant = {"content": '{\n"value": ' + t3_answer + '\n}', "role": "assistant"}
        row_dict = [t3_sys_prompt, t3_user_prompt, t3_assistant]
        
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
    
    def get_format_t3_test_data(self, t3_data):
        #columns_to_remove = [col for col in t3_data.column_names['train'] if col != 'id']
        t3_data = t3_data.map(self._format_t3_test_data,
                              #remove_columns=columns_to_remove,
                              fn_kwargs={"sys_p": self.get_t3_sys_prompt(), "user_p": self.get_t3_user_prompt()},
                              batched=False
                             )
        return t3_data