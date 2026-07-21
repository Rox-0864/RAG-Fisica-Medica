# AI-RAG-Radiodiagnostico

Agente de inteligencia artificial basado en RAG (Retrieval-Augmented Generation) capaz de
responder preguntas sobre física médica e imagenología utilizando como fuente de conocimiento
el libro **"Essential Physics of Medical Imaging"**.

El sistema funciona de forma 100% local y gratuita, sin depender de APIs externas de pago.
Las preguntas y respuestas son en español, mientras que el documento fuente está en inglés.

---

## Descripcion General

Este proyecto implementa un agente conversacional que permite a profesionales y estudiantes
de radiología realizar consultas en lenguaje natural sobre los principios físicos detrás de
las tecnologías de imagen médica: rayos X, tomografía computarizada (CT), resonancia
magnética (MRI), ultrasonido, medicina nuclear, y dosimetría de radiación.

El agente no "inventa" respuestas: busca en el libro los fragmentos más relevantes para
cada pregunta y los utiliza como contexto para generar una respuesta precisa y fundamentada.

### ¿Qué problema resuelve?

- **Acceso rápido al conocimiento**: en lugar de buscar manualmente en un libro de ~1000
  páginas, el usuario hace una pregunta y obtiene una respuesta inmediata.
- **Barrera del idioma**: el libro está en inglés, pero el agente responde en español
  usando terminología técnica correcta.
- **Costo cero en infraestructura**: todo corre localmente (Ollama + ChromaDB), sin
  necesidad de APIs pagas como OpenAI o Cohere.

---

## Arquitectura de la Solucion

El sistema implementa el patrón **RAG (Retrieval-Augmented Generation)** dividido en
dos fases:

### Fase 1: Indexacion (offline, se ejecuta una sola vez)

```
                    ┌──────────────┐
                    │   PDF Book   │
                    │  (~1000 pags) │
                    └──────┬───────┘
                           │ pdfplumber
                           ▼
                    ┌──────────────┐
                    │ Extraccion   │
                    │ texto +      │
                    │ tablas       │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Chunking     │
                    │ 1000 chars   │
                    │ overlap: 200 │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │ Embeddings   │
                    │ all-MiniLM   │
                    │ (384 dims)   │
                    └──────┬───────┘
                           │
                           ▼
                    ┌──────────────┐
                    │  ChromaDB    │
                    │ (vector      │
                    │  store)      │
                    └──────────────┘
```

### Fase 2: Consulta (online, cada vez que el usuario pregunta)

```
   ┌─────────────┐
   │  Pregunta   │
   │ (español)   │
   └──────┬──────┘
          │ embedding
          ▼
   ┌─────────────┐     ┌──────────────┐
   │ Búsqueda    │────▶│  ChromaDB    │
   │ semántica   │     │ (top K=5)    │
   └──────┬──────┘     └──────────────┘
          │ chunks relevantes (inglés)
          ▼
   ┌─────────────┐
   │ Prompt      │
   │ template    │
   │ bilingüe    │
   └──────┬──────┘
          │ context + question + chat history
          ▼
   ┌─────────────┐
   │  Ollama     │
   │ llama3.2:3b │
   └──────┬──────┘
          │
          ▼
   ┌─────────────┐
   │ Respuesta   │
   │ (español)   │
   └─────────────┘
```

### Diagrama de componentes

```
src/
├── app.py               # Interfaz Streamlit (chat UI)
├── document_loader.py   # Extraccion PDF + chunking
├── embeddings.py        # Wrapper HuggingFace (local)
├── vector_store.py      # Gestion ChromaDB (crear/cargar)
├── llm.py               # Interfaz Ollama
├── rag_chain.py         # Orquestacion retrieval + generacion
└── prompts.py           # Plantillas de prompt bilingues
```

---

## Tecnologias y Herramientas

| Componente | Tecnologia | Descripcion |
|------------|------------|-------------|
| **LLM** | [Ollama](https://ollama.com) + `llama3.2:3b` | Modelo de lenguaje de 3B parametros, ejecutado localmente. Excelente rendimiento multilingue. |
| **Embeddings** | [Sentence Transformers](https://sbert.net) + `all-MiniLM-L6-v2` | Modelo de 90MB que genera vectores de 384 dimensiones. Corre en CPU sin GPU. |
| **Vector Store** | [ChromaDB](https://trychroma.com) | Base de datos vectorial open-source, persistencia en disco. |
| **PDF Processing** | [pdfplumber](https://github.com/jsvine/pdfplumber) | Extraccion de texto y tablas desde PDFs con alta precision. |
| **Orchestration** | [LangChain](https://langchain.com) | Framework para construir cadenas RAG. |
| **UI** | [Streamlit](https://streamlit.io) | Interfaz web interactiva con componentes de chat. |

---

## Instrucciones para Ejecutar el Proyecto

### Requisitos previos

- Python 3.10 o superior
- 8 GB de RAM (recomendado 16 GB)
- Sistema operativo Linux, macOS o Windows (con WSL2)

### 1. Clonar el repositorio

```bash
git clone git@github.com:Rox-0864/AI-RAG-Radiodiagnostico.git
cd AI-RAG-Radiodiagnostico
```

### 2. Instalar Ollama y descargar el modelo

```bash
# Instalar Ollama (Linux)
curl -fsSL https://ollama.com/install.sh | sh

# Descargar el modelo de lenguaje
ollama pull llama3.2:3b

# Verificar que Ollama esta corriendo
ollama serve
```

### 3. Crear entorno virtual e instalar dependencias

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### 4. Configurar variables de entorno

```bash
cp .env.example .env
# Editar .env si es necesario (los valores por defecto funcionan)
```

### 5. Colocar el documento PDF

```bash
# Copiar el PDF del libro a la carpeta data/
cp ruta/al/essential-physics-medical-imaging.pdf data/
```

### 6. Ejecutar la aplicacion

```bash
streamlit run src/app.py
```

La primera ejecucion indexara automaticamente el documento (esto toma entre 2 y 5 minutos
dependiendo del hardware). Las ejecuciones posteriores cargaran el vector store desde disco
de forma instantanea.

---

## Ejemplos de Preguntas

El agente esta disenado para responder preguntas sobre los siguientes temas del libro:

### Rayos X y Radiografia

- "¿Cómo se producen los rayos X en un tubo de rayos X?"
- "¿Qué factores afectan la calidad de una imagen radiográfica?"
- "¿Cuál es la diferencia entre radiación de frenado (bremsstrahlung) y radiación característica?"
- "¿Cómo funciona una rejilla antidifusora y para qué se utiliza?"

### Tomografia Computarizada (CT)

- "¿Qué principio matemático utiliza la reconstrucción de imágenes en CT?"
- "¿Cómo se mide la atenuación en CT y qué son las unidades Hounsfield?"
- "¿Cuál es la diferencia entre adquisición secuencial y helicoidal en CT?"
- "¿Qué estrategias existen para reducir la dosis de radiación en CT?"

### Resonancia Magnetica (MRI)

- "¿Qué fenómeno físico fundamental permite obtener imágenes por resonancia magnética?"
- "¿Cómo afectan los tiempos de relajación T1 y T2 al contraste de la imagen?"
- "¿Qué es un gradiente de campo magnético y cuál es su función?"
- "¿Cuáles son los riesgos y contraindicaciones de la resonancia magnética?"

### Ultrasonido

- "¿En qué se diferencia el ultrasonido diagnóstico de otras modalidades de imagen?"
- "¿Qué determina la resolución espacial en una imagen de ultrasonido?"
- "¿Cómo funciona el efecto Doppler en ecografía?"

### Medicina Nuclear

- "¿Cuál es la diferencia entre SPECT y PET?"
- "¿Qué radioisótopos se utilizan comúnmente en medicina nuclear y por qué?"

### Proteccion Radiologica

- "¿Qué principios rigen la protección radiológica según el documento?"
- "¿Cómo se calcula la dosis efectiva a partir de la dosis absorbida?"
- "¿Qué diferencias hay entre efectos estocásticos y determinísticos de la radiación?"

---

## Ejemplos de Respuestas

### Ejemplo 1: Rayos X

**Pregunta:**
> "¿Cómo se producen los rayos X en un tubo de rayos X?"

**Respuesta del agente:**

Segun el documento "Essential Physics of Medical Imaging", los rayos X se producen
en un tubo de rayos X mediante dos mecanismos principales:

1. **Radiacion de frenado (Bremsstrahlung)**: Cuando los electrones acelerados desde
   el catodo se acercan al nucleo de los atomos del anodo (generalmente tungsteno), son
   desviados y desacelerados por el campo electrico del nucleo. Esta desaceleracion
   provoca la emision de fotones de rayos X cuya energia puede variar desde cero hasta
   la energia maxima del electron incidente (determinada por el kVp seleccionado).
   Este mecanismo produce el espectro continuo de rayos X.

2. **Radiacion caracteristica**: Cuando un electron incidente tiene suficiente energia
   para expulsar un electron de las capas internas (K, L) de un atomo del anodo, el
   atomo queda en un estado excitado. Un electron de una capa superior llena la vacante,
   y la diferencia de energia entre las capas se emite como un foton de rayos X con
   energia discreta y caracteristica del material del anodo.

El espectro total de rayos X emitido es la suma de ambos procesos, filtrado por la
ventana del tubo (que elimina los fotones de muy baja energia que no contribuyen a
la formacion de imagen pero si a la dosis del paciente).

---

### Ejemplo 2: Tomografia Computarizada

**Pregunta:**
> "¿Qué principio matemático utiliza la reconstrucción de imágenes en CT?"

**Respuesta del agente:**

Segun el libro, la reconstruccion de imagenes en tomografia computarizada se basa
fundamentalmente en la **transformada de Radon** y su inversion mediante la
**retroproyeccion filtrada (Filtered Back Projection, FBP)**.

El proceso es el siguiente:

1. El tubo de rayos X gira alrededor del paciente, y los detectores miden la atenuacion
   del haz para cada proyeccion angular. Esto genera un **sinograma**, que es la
   representacion de los perfiles de atenuacion en funcion del angulo.

2. Matematicamente, cada perfil de atenuacion corresponde a una proyeccion de la funcion
   de atenuacion del objeto (coeficiente de atenuacion lineal μ(x,y)) segun la
   transformada de Radon.

3. Para reconstruir μ(x,y) a partir de sus proyecciones, se aplica un filtro pasa-altos
   (como el filtro de Ram-Lak o Shepp-Logan) a cada proyeccion en el dominio de la
   frecuencia —esto corrige el emborronamiento inherente a la retroproyeccion simple—
   y luego se retroproyectan los datos filtrados sobre la matriz de imagen.

El documento tambien menciona metodos de reconstruccion iterativa modernos, que ofrecen
ventajas en la reduccion de ruido y artefactos, especialmente utiles en protocolos de
baja dosis.

---

### Ejemplo 3: Resonancia Magnetica

**Pregunta:**
> "¿Cómo afectan los tiempos de relajación T1 y T2 al contraste de la imagen en MRI?"

**Respuesta del agente:**

El libro explica que los tiempos de relajacion T1 y T2 son propiedades fundamentales
de los tejidos que determinan el contraste en las imagenes de resonancia magnetica:

**Relajacion T1 (longitudinal o spin-red):**
- Describe el tiempo que tardan los protones en recuperar el 63% de su magnetizacion
  longitudinal despues de un pulso de RF.
- Tejidos con T1 corto (como la grasa) recuperan su magnetizacion rapidamente y
  aparecen brillantes (hiperintensos) en imagenes potenciadas en T1.
- Tejidos con T1 largo (como el agua o LCR) aparecen oscuros (hipointensos) en T1.
- El contraste T1 se manipula eligiendo un **TR (tiempo de repeticion)** corto para
  que los tejidos con diferente T1 no hayan alcanzado todos la magnetizacion completa.

**Relajacion T2 (transversal o spin-spin):**
- Describe la perdida de magnetizacion transversal debido a interacciones entre espines.
- Tejidos con T2 largo (como el agua) mantienen la senal por mas tiempo y aparecen
  brillantes en imagenes potenciadas en T2.
- Tejidos con T2 corto (como la grasa) pierden la magnetizacion transversal mas
  rapidamente.
- El contraste T2 se controla mediante el **TE (tiempo de eco)**: TE largo permite
  que las diferencias de T2 se manifiesten.

La clave clinica: las secuencias de pulso se disenan para enfatizar las diferencias
de T1, T2 o densidad protonica segun la patologia que se desea evaluar.

---

## Licencia

Este proyecto es de codigo abierto. El documento "Essential Physics of Medical Imaging"
tiene sus propias condiciones de uso y no se distribuye con este repositorio.
