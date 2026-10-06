# PejeDex — clasificador de especies de peces

Clasificador de **240 especies** de peces (y algo de invertebrados) entrenado para
la app **PejeDex**, una app offline-first de pesca en Canarias: el usuario saca una
foto al pez y la app le dice qué es, sin conexión.

## Por qué existe esto

La app original venía con un modelo ajeno (`SeeFish_1_0.tflite`) que reconocía muy
pocas de las especies que la app tenía en su base de datos. El objetivo era tener
**un modelo propio** que:

- corriera **100 % offline** en el móvil (sin servidor, sin llamadas)
- pesara **< 30 MB** (ideal 5-20 MB)
- respondiera en **CPU móvil < 100 ms**
- cubriera las especies que un pescador canario se encuentra de verdad

El resultado es `model/pejedex.tflite`: **9,3 MB**, ~4 ms de inferencia en escritorio,
**81,3 % top-1**.

## Resultados

| Métrica | valor |
|---|---|
| top-1 (validación) | **81,33 %** |
| top-5 (validación) | **94,31 %** |
| macro-F1 (240 clases) | **79,42** |
| Parámetros | 8,8 M |
| Época del mejor | 38 / 40 |

Export a TFLite (medido sobre 2000 fotos de validación):

| variante | tamaño | top-1 (crop) | top-1 (resize) |
|---|---|---|---|
| float32 | 34,9 MB | 81,40 | 78,00 |
| **float16 → `pejedex.tflite`** | **9,3 MB** | **80,95** | 76,95 |
| int8 | 9,7 MB | 71,65 ⚠️ | 67,00 |

> **No uses int8**: con las 240 clases la cuantización de pesos se desploma
> (−9,7 puntos). float16 da −0,38 puntos respecto a PyTorch por 1/4 del tamaño.

## Datos

- **118.102 fotos** (100.279 train / 17.823 val) en **240 clases**
- Fuente principal: **iNaturalist**, solo licencias permitidas
  (`cc-by`, `cc-by-sa`, `cc0`, `pdm`, `cc-by-nc`, `cc-by-nc-sa`)
- Filtro: mínimo 70 fotos por clase, sin duplicados exactos, sin fotos rotas
- Nombres en castellano en `dataset/classes.json` y `data_raw/names_es.json`
- El dataset **no está en el repo** (87 GB de crudos): se reconstruye con los scripts

Cobertura respecto a las 367 especies de la app: **156 (42,5 %)** de todas ellas,
pero **33 de 36 (91,7 %)** de las especies con talla mínima regulada — es decir,
de lo que un pescador pesca de verdad. El resto son especies de enciclopedia
(tiburones abisales, de profundidad, cabazos raros…) de las que iNaturalist no
tiene fotos con licencia libre.

## Estructura

```
scripts/
  fetch_inat.py        descarga fotos de iNaturalist (reintentos, fallback por especie)
  fetch_add.py         añade las especies que faltaban (species_add.json)
  fetch_progress.py    monitor de descarga: hueco, cobertura y ETA
  fetch_names_es.py    nombres en castellano (Wikidata / es.wikipedia / WoRMS)
  build_dataset.py     limpia, valida y monta dataset/train + dataset/val (symlinks)
  train.py             entrenamiento: warm-start, reanudación, AMP
  export_tflite.py     PyTorch → ONNX → SavedModel → TFLite (+ evaluación)
  predict.py           clasifica una foto e imprime top-K por consola
  probe_app_species.py sondea qué especies de la app tienen fotos disponibles
  species_add.json     las 72 especies que se añadieron al modelo
data_raw/names_es.json tabla de nombres científicos → castellano (240+ entradas)
checkpoints/best_slim.pt  pesos del mejor modelo (35 MB, sin optimizer)
model/pejedex.tflite   entregable para la app (float16)
model/labels.txt       etiquetas, una por línea, en el mismo orden que el modelo
```

## Pipeline completo

```bash
# 0. entorno (torch + timm + tensorflow)
python -m venv .venv-train && source .venv-train/bin/activate
pip install torch timm tensorflow pillow numpy

# 1. datos
python scripts/fetch_inat.py            # descarga (se puede parar y reanudar)
python scripts/fetch_progress.py --once # ¿cuánto falta?

# 2. dataset
python scripts/build_dataset.py --min-per-class 70

# 3. entrenar (desde cero)
python scripts/train.py --lr 3e-4 --epochs 40 --save-every 1

#    ...o calentar desde otro checkpoint / reanudar
python scripts/train.py --init-from checkpoints_165/best.pt --lr 3e-4 --epochs 40
python scripts/train.py --resume checkpoints/last.pt --epochs 40 --lr 3e-4

# 4. exportar
python scripts/export_tflite.py          # genera model/pejedex.tflite + labels.txt

# 5. probar
python scripts/predict.py mi_foto.jpg
```

Progreso del entrenamiento: `tail -f logs/train_*.log`

## Clasificar una foto

```bash
python scripts/predict.py mi_foto.jpg
```

```
mi_foto.jpg  [4.6 ms, entrada float32]
  1.  50.1% ####################   Lubina  (Dicentrarchus labrax)
  2.   1.5%                        Baila  (Dicentrarchus punctatus)
  ...
  => Lubina (Dicentrarchus labrax)  50.1%
```

## Integración en la app — esto es importante

**La app debe recortar, no estirar.** La diferencia son 4,25 puntos de precisión:

```python
escala = 235 / min(w, h)          # lado corto a 235
im = im.resize((round(w*escala), round(h*escala)), BICUBIC)
im = im.crop_central(224)         # recorte CUADRADO centrado
```

- Entrada: `float32 [1,224,224,3]` RGB `0..255`; la normalización (ImageNet mean/std)
  ya está **dentro del grafo**
- Salida: `float32 [1,240]` logits → aplicar softmax
- La app usa **nombres canarios** (*Bocinegro*, *Cabozo*…); el modelo devuelve
  **nombres estándar** (*Pargo*, *Serrano*…) y el nombre científico. Hay que
  mapear por `scientificName` usando `data_raw/names_es.json`.



## Licencia de los datos

Fotos de iNaturalist bajo sus licencias abiertas (CC0, PDM, CC BY, CC BY-SA,
CC BY-NC, CC BY-NC-SA). Respeta los términos de cada autor al reutilizarlas.

# Condiciones

En caso de usar este modelo, dar creditos: github.com/aithor07
