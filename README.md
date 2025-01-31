# Odesia_UDA_LIDI
 
# Este repositorio contiene los archivos necesarios para replicar el envío del equipo UDA LIDI en el challenge de ODESIA

## Requisitos
Para facilitar la recreación, se puede importar una imagen de un contenedor con todos los drivers CUDA a través de enroot
"enroot import docker://nvcr.io#nvidia/cuda:12.2.2-devel-ubuntu22.04"
"enroot create --name <nombre_contenedor> nvidia+cuda+12.2.2-devel-ubuntu22.04.sqsh"
iniciar el entorno: "enroot start  --root --rw <nombre_contenedor>"


Se recomienda el uso de un entorno virtual para instalar los paquetes necesarios del archivo "requirements.txt"