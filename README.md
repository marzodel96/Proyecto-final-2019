## Reconocimiento de vehículos mediante streaming de videos para su contabilización y clasificación en tiempo real

__DEMOSTRACIÓN__
- Ver demo en enlace: https://drive.google.com/file/d/1AOKwX0KN2ji0_FkuHJVLVfmMy1ARliln/

__HARDWARE__
- Notebook: Asus VivoBook S430FN
- Procesador: Intel Core i5-8265U (4 núcleos, 8 hilos, 1600 MHz - 3900 MHz) 8va generación
- Memoria RAM: 
    - 4 GB DDR4 (2400 MHz) {Incorporada}
    - 4 GB DDR4 (2400 MHz) {Aparte}
- Pantalla: LED 14.0" (1920x1080) / 60 Hz
- Almacenamiento: 256 GB SSD
- Tarjetas de video:
    - NVIDIA GeForce MX150 (2 GB)
    - Intel UHD Graphics 620 (Integrada)
- Sistema Operativo:
    - Windows 10 (primaria)
    - Linux Ubuntu (partición)

__SOFTWARE__
- Se utilizó la aplicación gratuita __IP Webcam__ (disponible en Google Play) para la transmisión de videos en tiempo real, a través del smartphone, la cual al iniciar otorga una IP que se debe entregar al código de reconocimiento para que pueda entregar la visualización de los diversos vehículos que posiblemente sean detectados dentro del proyecto.


__INSTALACIONES__
- Windows 10
- Python 3.7.2
- Visual Studio Community 2015 y 2017
- Verificar versión de CUDA soportado por tarjeta gráfica NVIDIA MX150
    - nvidia-smi: CUDA VERSION -> 10.0 (Revisado el 07.10.2019 -> 10.1)
- Instalar versión 10 de CUDA Toolkit en: https://developer.nvidia.com/cuda-toolkit-archive
- Instalar cuDNN (v7.5.1.10) para CUDA (v10.0): https://developer.nvidia.com/rdp/cudnn-archive. Luego, seguir los pasos de instalación acá: https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows
- OpenCV 3.4.5 desde página https://opencv.org/releases/ -> extraer en C:\ y guardarlo en una subcarpeta llamada opencv_3.0 para que el proyecto en VS pueda tomar las rutas indicadas de manera correcta.


__VARIABLES DE ENTORNO QUE SE DEBEN AGREGAR__
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
- C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp
- C:\darknet
- C:\opencv_3.0\opencv\build\x64\vc14\bin
- C:\opencv_3.0\opencv\build\include


__CÓMO COMPILAR EN WINDOWS__
- Una vez instalado todo (incluyendo variables de entorno) se procede a abrir el archivo en la ruta build\darknet\darknet.sln con VS2017, cambiando a x64 y Release
- Compilar -> Compilar darknet
- Luego de haber compilado exitosamente el proyecto, se creará en la ruta build\darknet\x64 y archivo ejecutable ‘darknet.exe’
- Encontrar los archivos opencv_ffmpeg345.dll y opencv_ffmpeg345_64.dll en la ruta C:\opencv_3.0\opencv\build\x64\vc14\lib y copiarlos en la misma ruta en la que se encuentra el ejecutable creado C:\darknet\build\darknet\x64
- Descargar cuDNN para CUDA v10 y copiar el archivo ‘cudnn64_7.dll’ a la ruta C:\darknet\build\darknet\x64 cerca de ‘darknet.exe’


__PREVIO__
- Ruta para darknet: C:\darknet
- Ruta para realizar entrenamientos y testeo de datos con modelo obtenido: C:\darknet\build\darknet\x64 -> cmd en ruta
- Ruta con dataset de automóviles: C:\darknet\build\darknet\x64\data\obj -> un total de 158 imágenes con su respectivo bounding boxes para para imagen en formato .txt
    - Software utilizado para realizar bouding boxes: LabelImg en su formato binario para Windows (v1.8.0) http://tzutalin.github.io/labelImg/ 
    - Formato .txt utilizado es requerido para entrenar nuevos modelos en API YOLO.
    - Labels: automovil – taxibus – bus.
- Dataset es sacado de UOCT en su cuenta Twitter y recopilando imágenes en Google. El 80% de las imágenes que sirven de entrenamiento son de las calles de Concepción.
- Videos para ser utilizados como prueba son sacados de la página oficial de UOCT a través del caché del navegador. Duración de videos: 30 segundos aproximadamente.


__PREVIO ENTRENAMIENTO YOLO__
- Para el entrenamiento de un nuevo modelo utilizando dataset propio se realizó lo siguiente:
    - https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects -> “How to train tiny-yolo (to detect your custom objects)”
    - Se utilizaron además los mismos nombres estipulados en el enlace para el ejemplo.
    - Se crean .weights nuevos utilizando .weights predeterminados por la API 
- Para el entrenamiento creado, se llegaron a las 2.400 iteraciones aproximadamente. Se utiliza el archivo <.weights> en ruta C:\darknet\build\darknet\x64\backup creado por el entrenamiento de peso 2.000 para realizar un testeo y ver los resultados. 
- Resultados entregados por el testeo indica un reconocimiento adecuado para cada label que se determinó, entregando el porcentaje de aserción para cada objeto.
- Hay que considerar que, para obtener un buen modelo entrenado, se deben tener 2.000 iteraciones para cada clase pero que no sea menos de 4.000. Pero para obtener un modelo confiable, se debe llegar como mínimo a las 9.000 iteraciones (sugerencia de API YOLO).


__PRUEBAS CON MODELO CREADO__
- Para hacer las pruebas pertinentes con el modelo creado en base a las imágenes de automóviles y taxibuses, se utiliza el comando darknet.exe detector demo data/obj.data yolov3-tiny-obj.cfg yolov3-tiny-obj_2000.weights -i 0 <archivo-de-video>.mp4 en la ruta C:\darknet\build\darknet\x64

__RESULTADO FINAL__
- Config.: configuraciones-iteracion_6000.cgf
- Modelo: modelo-iteracion_6000.weights (Resultado entrenamiento)
    - Duración de 4 hrs. aproximadamente
- Se realizaron grabaciones propias desde diversas pasarelas buscando alturas óptimas que permitieran el reconocimiento a distancia de diversos vehículos