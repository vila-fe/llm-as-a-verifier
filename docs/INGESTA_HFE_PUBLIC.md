# Ingesta del dataset HFE público (Kaggle `deepmind/hfe-public`) — guía desmenuzada

Objetivo: cargar los **112 prompts públicos** (28 familias completas, CC BY 4.0)
de *Humanity's First Exam* en la suite Suelo HFE y validarla en HOOTL.

## Paso 0 — Por qué hay que hacerlo fuera del sandbox
Kaggle bloquea el acceso automatizado y exige sesión para descargar. Ninguna
herramienta del sandbox puede traer el zip. La descarga es manual (2 minutos).

## Paso 1 — Descarga manual (tú, en tu navegador)
1. Entra en Kaggle con tu cuenta: https://www.kaggle.com/datasets/deepmind/hfe-public
2. Pulsa **Download**. Obtienes un zip (por ejemplo `archive.zip` o `hfe-public.zip`).
3. Guárdalo en tu equipo. No hace falta descomprimirlo.

## Paso 2 — Entrega del zip al repo (elige una)
- **A (recomendada):** sube el zip (o su contenido extraído) a
  `data/hfe_floor_trajs/kaggle_raw/` en la rama del PR (o simplemente haz commit
  en la rama que uses). CC BY 4.0 permite redistribuir con atribución — el
  campo `source` de la suite ya la registra.
- **B:** pega en el chat el contenido del fichero de prompts/respuestas
  (texto JSON/CSV/YAML) y la conversión se hace desde ahí.

## Paso 3 — Ingesta automática (1 comando)
El script `scripts/ingest_hfe_public.py` autodetecta el formato (zip, directorio,
JSON, JSONL, CSV, carpetas Harbor `task.yaml`/`task.json`/`prompt.txt`) y los
nombres de campo habituales (`task_id`/`id`, `question`/`prompt`,
`reference_answer`/`solution`, `family_id`, `category`):

```bash
# solo ingesta -> data/hfe_floor_trajs/hfe_public_floor.jsonl
python scripts/ingest_hfe_public.py data/hfe_floor_trajs/kaggle_raw/hfe-public.zip

# ingesta + validación HOOTL inmediata
python scripts/ingest_hfe_public.py data/hfe_floor_trajs/kaggle_raw/hfe-public.zip --run
```

Salida esperada con el dataset real: `Ingestados 112 prompts (28 familias, 112
con respuesta de referencia)`.

## Paso 4 — Validación HOOTL y registro de política
```bash
python scripts/run_hfe_floor.py --suite data/hfe_floor_trajs/hfe_public_floor.jsonl
```
- Métricas: pass@1 y family-pass con cota inferior de Wilson 95% frente al
  umbral canónico (0.85 por defecto).
- Cada ejecución se anota en `data/progress_tracking/hfe_floor_log.jsonl`
  (registro de política: modo, perfil, métricas, umbral, veredicto).

## Paso 5 (posterior) — Evaluación con modelos reales
Con la suite real ingerida, el benchmark `hfe_floor` queda disponible en el
pipeline del verificador:
```bash
python scripts/run.py hfe_floor
```

## Notas y límites conocidos
- **Texto vs. imagen:** parte de los 112 prompts dependen de imágenes (categoría
  Vision). Para un modelo text-only, filtra por `capability` antes de concluir
  nada sobre fiabilidad de suelo; el informe HOOTL desglosa por capability.
- **Familias:** la métrica family-pass exige que las 4 variantes de cada familia
  estén presentes y correctas en todos los runs. La suite pública (28 familias) es
  una muestra estratificada del 20% del HFE completo (544 prompts, 136 familias).
- **HOOTL modela el comportamiento del modelo, no los datos:** prompts y
  respuestas de referencia son los reales de HFE; solo la respuesta del modelo
  está simulada por perfil de fiabilidad hasta que se conecten modelos reales.
- Scripts probados con tres layouts sintéticos (JSONL plano, carpetas Harbor y
  zip con carpeta raíz) — si el layout real de Kaggle usa otro esquema, el
  script informa y solo hay que ajustar las listas `*_KEYS` del propio script.
