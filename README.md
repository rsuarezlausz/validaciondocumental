# Portal de Validación Documental (POC)

Sube PDF/JPG/PNG, extrae texto con OCR (es+en) y estima la probabilidad de alteración (ELA + residual de ruido + metadatos), mostrando las regiones sospechosas sobre el documento.

## Estructura
```
backend/   FastAPI: app/{ocr_engine,forensic_analyzer,metadata_checker,models,db}.py, routers/, services/pipeline.py
frontend/  React 18 + TS + Vite + MUI (UploadComponent, ResultSidebarComponent, DocumentService, HistoryPage)
docker-compose.yml
```

## Ejecución con Docker
```
docker compose up --build
```
UI: http://localhost:8080 · API/Swagger: http://localhost:8000/docs

## Ejecución local
Requisitos: Python 3.11+, Node 20+, [Tesseract](https://github.com/tesseract-ocr/tesseract) con paquetes `spa` y `eng` en el PATH.
```
cd backend && python -m venv .venv && .venv\Scripts\activate   # Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload            # :8000

cd frontend && npm install && npm run dev  # :5173 (proxy /api -> :8000)
```
Tests backend: `cd backend && pytest`.

## API (`/api/v1/documents`)
| Método | Ruta | Descripción |
|---|---|---|
| POST | `/upload` | multipart `file`. Valida MIME + firma + tamaño (20 MB). 201 `{document_id, filename, status}` |
| POST | `/analyze/{id}` | Encola OCR + forense (BackgroundTasks). 202; idempotente |
| GET | `/results/{id}` | Estado (`uploaded/processing/completed/failed`) y resultado |
| GET | `` | Historial |
| GET | `/{id}/pages/{n}` | PNG de la página (preview) |

Ejemplo de `GET /results/{id}` (completed; `paragraphs`, `pages` abreviados):
```json
{
  "document_id": "uuid",
  "filename": "contrato.pdf",
  "status": "completed",
  "ocr_text": "texto extraído...",
  "forensic_analysis": {
    "tampering_probability": 0.87,
    "confidence": 0.92,
    "findings": [
      {"type": "ela_inconsistency",
       "region": {"x": 120, "y": 340, "width": 200, "height": 80, "page": 1},
       "description": "Inconsistencia de compresión (ELA) (z=9.4)", "score": 0.54}
    ],
    "metadata_issues": ["Fecha de modificación posterior a creación (...)"],
    "integrity_hash": "<merkle root sha256 de las páginas>"
  },
  "pages": [{"page": 1, "width": 1240, "height": 1754, "url": "/api/v1/documents/<id>/pages/1"}],
  "processing_seconds": 4.2
}
```
Coordenadas de `region` en píxeles de la imagen de preview (PDF renderizado a 150 DPI).

## Decisiones y límites de la POC
- **OCR**: Tesseract (`spa+eng`). PDFs con capa de texto usan ésta sin OCR. PaddleOCR se puede enchufar reemplazando `ocr_engine.ocr_page`. No hay extracción de tablas (sólo párrafos).
- **Forense**: ELA y ruido por bloques de 16 px con z-score robusto; las regiones son componentes conexas de bloques atípicos. Son **heurísticas sin entrenar**: no hay modelo ML ni dataset de evaluación, por lo que hay falsos positivos/negativos (p. ej. ELA es poco informativo en PDFs rasterizados o JPEG ya recomprimidos). Para evaluar rendimiento ML hace falta un dataset etiquetado (p. ej. CASIA, DocTamper) y reemplazar `forensic_analyzer.analyze_image` por un modelo.
- **Metadatos**: PDF (fechas, productor/creador, `%%EOF` múltiples = guardado incremental, raíz Merkle SHA-256 propia de contenidos de página con `hashlib`, sin `merkly`) y EXIF en imágenes.
- **Seguridad**: carga a `data/uploads` con nombre UUID, el original se **elimina** tras el análisis (éxito o fallo); sólo se conservan los PNG de preview en `data/pages` para poder reabrir. Límite 10 páginas.
- **Cola**: `BackgroundTasks` (en proceso, se pierde si el servidor se reinicia). Para producción: Celery/RQ.
- **Trazabilidad**: `data/logs/analysis.log` (timestamp, document_id, estado, probabilidad, segundos).
- **Overlay**: SVG sobre la imagen (en lugar de react-konva, que no hacía falta).
- No hay autenticación; no exponer fuera de entorno de pruebas.

## Forense de documentos fotografiados (DNI)

El analisis clasico (ELA + ruido + doble JPEG + metadatos) no detecta ediciones pequenas en fotos. Se agregaron senales
locales en `backend/app/forensics/` (solo para imagenes; los PDF y renders digitales no las usan porque sus glifos
repetidos son identicos por construccion). Todas devuelven *indicios*, nunca una afirmacion de falsedad.

| Senal (`type`) | Modulo | Que mide | Limites conocidos |
|---|---|---|---|
| `barcode_mismatch` / `barcode_unreadable` | `barcode_validator.py` | Decodifica el PDF417 con `zxing-cpp` y compara cada campo con el OCR. Los campos se clasifican por **patron** (no se asume el formato). Si el barcode no trae vencimiento, no se valida contra el. Ilegible = dato informativo | Depende del OCR; un campo que el OCR no encuentra queda "no concluyente", no "discrepancia". Valores sensibles salen enmascarados (`***713`) |
| `glyph_copy_move` | `glyph_copy_move.py` | Pares de glifos de tamano compatible: forma (NCC) + correlacion del ruido del **fondo plano**. Digitos repetidos naturales tienen forma casi igual pero ruido de fondo independiente | Se desactiva (no aplicable) si no queda ruido de sensor (render digital o recompresion fuerte, JPEG <= ~Q80). Recompresiones intermedias pueden esconder la copia |
| `seam_artifact` | `seam_detector.py` | Lineas rectas de 1-3 px en el residuo de alta frecuencia, solo en campos de texto y sobre pixeles sin tinta | Una rayadura o reflejo fino puede parecer costura |
| `local_inconsistency` | `local_consistency.py` | Ruido de fondo, nitidez, contraste e iluminacion por caracter frente a la mediana de su linea (MAD) | Senal debil: iluminacion y desgaste la disparan |
| `geometry_anomaly` | `text_geometry.py` | Linea base, altura y paso por linea. El paso solo se evalua en lineas de paso casi fijo (digitos) | Senal debil |
| `rule_violation` | `rules.py` + `config/rules.yaml` | Emision < vencimiento, nacimiento < emision, vigencia esperada | **Todas son supuestos "por verificar"** (ej. vigencia de 15 anos para mayores de 16: confirmar con la fuente oficial) |
| `ela`, `noise`, `jpeg_ghost`, `metadata` | `forensic_analyzer.py`, `metadata_checker.py` | ELA normalizado por textura local, ruido, JPEG ghost y metadatos | Apoyo. Un EXIF ausente o un re-guardado **no elevan mucho** el score (WhatsApp y editores simples lo producen) |

Zonas excluidas por plantilla (`config/template_dni.yaml`, coordenadas relativas): foto, holograma y banda guilloche no se
marcan por ELA/ruido (alta textura legitima); se aplican solo si la imagen tiene proporcion de tarjeta. Las coordenadas son
un supuesto derivado de una foto: ajustar por version del DNI.

### Fusion y confianza
`score_fusion.py`: cada senal aporta `peso * score` (pesos en `config/thresholds.yaml`). Las fuertes (barcode, copy-move,
costura, reglas) se combinan con noisy-OR; las debiles quedan acotadas por `weak_cap` (0.45), asi una senal debil nunca pasa de
un score medio. La confianza baja si el barcode es ilegible, hay pocos glifos o el copy-move no aplica. La respuesta incluye
`signals`, `contributions` (desglose por senal), `findings` (con `severity` y `regions`) y `limitations` (siempre presente).

### Modelo opcional de ML
`ml_adapter.py`: interfaz `LocalizationModel.predict(image) -> heatmap`, stub y ejemplo TruFor/CAT-Net detras de
`ENABLE_ML_LOCALIZATION=true`. Desactivado por defecto; no descarga pesos.

### Calibracion y evaluacion
```
cd backend
python -m eval.synthetic_eval --n 30            # reporta AUC, precision, recall, FP por senal y fusionada
python -m eval.synthetic_eval --n 30 --write    # ademas escribe eval/calibrated_thresholds.yaml (revisar y copiar)
```
Usa documentos **sinteticos** con ground truth (copia de un caracter y costura). Los umbrales de `thresholds.yaml` son
provisionales: recalibrarlos con documentos propios del equipo (genuinos y editados) antes de usarlos como evidencia.
Umbrales/plantilla/reglas pueden sobreescribirse con `FORENSIC_THRESHOLDS`, `FORENSIC_TEMPLATE`, `FORENSIC_RULES`.

### Privacidad
No se registra en logs el contenido del barcode, el OCR ni numeros de documento. El original se elimina tras el analisis.
Las imagenes de prueba reales van en `backend/tests/fixtures/private/` (ignorado por git).

### Validacion con imagenes reales del equipo (`backend/tests/fixtures/private/`, no versionadas)
Convencion: `real*` sin alterar, `falso*` alterado (digito 3 copiado sobre el 0 del vencimiento 2030 -> 2033; `falso1` ademas
tiene una linea vertical de 1 px junto al ultimo digito). Resultado actual (3 imagenes, 2902x1927):

| Imagen | Probabilidad | Hallazgos clave |
|---|---|---|
| `real` | 0.27 | sin copy-move, sin costura, barcode coincide (numero de documento, tramite, emision) |
| `falso2` | 0.89 | `glyph_copy_move` alto entre los dos ultimos "3" (fondo 0.98), vencimiento 2033 vs vigencia supuesta 2030 |
| `falso1` | 0.92 | lo anterior + `seam_artifact` en x~1757, y 1408-1458 (contraste 78) |

Son **3 imagenes de un mismo documento**: valida que las senales funcionan en este caso, no que generalicen. Se agrego OCR dirigido
por campo (`field_ocr.py`: recorte por plantilla, 3 preprocesados, recortes desplazados y votacion; un campo solo es fiable con >= 2
votos). Un campo no fiable queda "no concluyente" y nunca genera `barcode_mismatch`: el OCR de pagina completa daba falsos
desacuerdos sobre el documento real. Tiempo medio ~18 s por foto (OCR dirigido ~10 s). Requiere Tesseract en el PATH.

### Parches planos / white-out (`flat_patch_detector.py`)
Cuadro blanco (o de color liso) pegado sobre fondo texturado, con texto nuevo encima (caso `id_3.jpeg`, campo `9a END`).
ELA y ruido no lo ven (una zona plana comprime sin error). Se detecta por **discontinuidad de uniformidad**: contorno externo
relleno de la zona plana (los trazos de la letra quedan como agujeros), rectangularidad >= 0.75, interior casi sin textura
(`inside_std` < 2) frente a un anillo texturado (`ring_std` >= 5, razon >= 5), mas brillante que el anillo (descarta huecos
entre digitos y fondos lisos) y crecimiento del rectangulo hasta su borde real. Excluye foto, holograma, barcode y margen de
la plantilla. En `id_3.jpeg`: severidad alta en x~447, y~261, 37x26 px.

Otras tecnicas que se pueden sumar (no implementadas, ordenadas por valor/coste):
1. **Tipografia del texto nuevo** (grosor de trazo, altura, familia) frente a los campos hermanos: requiere un comparador de
   fuentes mas robusto; con trazos antialias claros (la "A" de `id_3`) los umbrales simples no sirven.
2. **Croma/neutralidad**: un parche blanco puro (R=G=B=255) sobre un fondo con tinte es una discontinuidad de color adicional.
3. **Rejilla JPEG 8x8**: el parche pegado no comparte la fase de bloques del resto de la imagen.
4. **Continuidad del patron de fondo** (`background_continuity.py`): ya existe, solo refuerza la severidad.
5. **Validacion semantica** del valor (p.ej. "A" en el campo de restricciones) con reglas por jurisdiccion en YAML.
6. **Modelo de localizacion** (TruFor/CAT-Net) tras `ENABLE_ML_LOCALIZATION`.

Vista previa: las imagenes (PNG/JPG/BMP) ahora guardan su PNG de pagina al analizar (antes solo lo hacia la ruta PDF y la
vista previa devolvia 404). Si Tesseract no esta disponible el analisis ya no falla: omite el OCR y lo informa en `warnings`.

### Fiabilidad en PDF y barcode opcional
- **PDF escaneado / foto convertida a PDF**: si una imagen embebida cubre >= 85% de la pagina se analiza **a su resolucion nativa**
  (conserva compresion y ruido del original) y recibe las mismas senales que una foto (glifos, costuras, barcode, parches planos).
  Una pagina vectorial se rasteriza y **no** recibe senales de fotografia ni ELA/ghost (reflejarian la conversion).
- **Fechas del OCR**: cada etiqueta solo posee el texto hasta la siguiente etiqueta; si la fecha de emision no se lee (holograma)
  no se toma la de vencimiento. Fechas repetidas o fuera de orden se descartan (no concluyente).
- **Barcode opcional**: una discrepancia solo es fuerte (`high`, peso 0.9) si el campo se leyo con OCR dirigido y votos fiables.
  Con OCR de pagina completa es un indicio debil (`low`, 0.2). La ausencia o ilegibilidad del barcode no suma sospecha ni resta
  fiabilidad: cada documento aporta las comprobaciones que tiene.
- **Capacidades reales**: `analysis_capabilities` lista lo que de verdad corrio y por que se omitio el resto.
- **Interfaz**: con veredicto `inconclusive` se muestra "no concluyente, revision manual" y el porcentaje se atenua; no se presenta
  como probabilidad de falsedad.
Caso de regresion: `real.jpeg` convertido a PDF (`DNI_PDF.pdf`): antes 92% con `barcode_mismatch` falso (2030 leido como emision);
ahora 0.13, `no_evidence`, barcode coincide.

### PDF de Word con imagenes y capa de texto en blanco (caso `DNI Cesar Perez.pdf`)
**Por que el OCR no leia nada**: el PDF (Word 2019, A4) solo contiene dos fotos JPEG (frente y dorso, 720 px) y una capa de texto
formada por espacios y saltos de linea. El pipeline daba por buena la capa de texto si tenia > 20 caracteres, asi que **omitia el
OCR** y el texto quedaba vacio. Ademas, la pagina A4 rasterizada dejaba el documento diminuto.
Cambios: la capa de texto solo cuenta con >= 20 caracteres alfanumericos; una pagina de solo imagenes se analiza imagen por imagen a
su resolucion nativa; el OCR amplia imagenes pequenas (~2200 px de lado largo). El OCR paso de 0 a ~760 caracteres.

**Perfiles de documento** (`forensics/config/profiles.yaml`): la plantilla del DNI argentino (zonas excluidas, OCR por campo) y la
regla de vigencia (15 anos) solo se aplican si el OCR identifica ese documento (`REPUBLICA ARGENTINA`/`MERCOSUR`). Cualquier otro
documento usa el perfil `generic`: sin plantilla y sin vigencia sin fuente verificada (un DNI peruano emite 2021 y caduca 2029).
Las fechas del OCR aceptan formato numerico (`19 07 2021`), `1907 2029` y las etiquetas `caducidad`/`birth`/`expiry`.

**Falsos positivos corregidos** (MRZ `<<<<` y codigo de barras en la imagen de 720 px): el fondo para comparar ruido excluye la tinta
vecina y se descarta si tiene estructura (desv. > 8); copy-move no aplica con texto < 20 px de alto; lineas paralelas repetidas no
son una costura. Limite: con texto tan pequeno no se detecta copy-move (se informa en `limitations`).
