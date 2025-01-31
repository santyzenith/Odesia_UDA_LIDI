# Odesia_UDA_LIDI
 
# Este repositorio contiene los archivos necesarios para replicar el envío del equipo UDA LIDI en el challenge de ODESIA

## Requisitos
Para facilitar la recreación, se puede importar una imagen de un contenedor con todos los drivers CUDA a través de enroot


"enroot import docker://nvcr.io#nvidia/cuda:12.2.2-devel-ubuntu22.04"


"enroot create --name <nombre_contenedor> nvidia+cuda+12.2.2-devel-ubuntu22.04.sqsh"


iniciar el entorno: "enroot start  --root --rw <nombre_contenedor>"


Se recomienda el uso de un entorno virtual para instalar los paquetes necesarios del archivo "requirements.txt"

Instalar los requisitos:

python3 -m pip install torch==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
python3 -m pip install flash-attn==2.7.3 --no-build-isolation

## Entrenamiento
Para el entrenamiento se utilizaron 4 x Nvidia A100 40GB.

Para recrear el entrenamiento:

torchrun --n_proc_per_node #NumberOfGpus train_odesia.py

## Inferencia
Para la inferencia en los conjuntos de test y generar los archivos para subirlos al challenge,

no se configuró una inferencia distribuida o un acelerador (por motivos de tiempo).

Existen varios archivos para recrear la inferencia, por ejemplo:

python3 exist23_t3_inference.py

Se puede cambiar el valor del parámetro "device_map" en los scripts para correr cada inferencia en una GPU distinta.