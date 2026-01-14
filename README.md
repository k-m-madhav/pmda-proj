\# Rail Track Segmentation Using DINOv2



Computer Vision project for automated rail track segmentation using foundation models.



\## Project Overview



\*\*Goal:\*\* Develop a functional prototype demonstrating rail track segmentation using DINOv2 foundation model.



\### Features

\- Rail track segmentation using pretrained DINOv2

\- Object detection and counting (tracks, trains, obstacles)

\- Interactive Streamlit web interface

\- Future: Digital twin with georeferenced segmentation (OSDaR23 dataset)



---



\## 🗂️ Dataset



\*\*Primary:\*\* \[RailSem19](https://wilddash.cc/railsem19) - Rail scene segmentation dataset  

\*\*Secondary:\*\* \[OSDaR23](https://github.com/RailAI/osdar23) - Railway dataset with GPS metadata



---



\## 🛠️ Tech Stack



\- \*\*Model:\*\* DINOv2 (Meta AI) - Vision foundation model

\- \*\*Framework:\*\* PyTorch

\- \*\*UI:\*\* Streamlit

\- \*\*Visualization:\*\* Matplotlib, OpenCV

\- \*\*Package Manager:\*\* uv (fast Python package installer)



---



\## 📦 Installation



\### Prerequisites

\- Python 3.11

\- uv package manager



\### Setup

```mermaid
graph TD
    %% -- User Interface Layer --
    subgraph UI ["User Interface (Streamlit)"]
        direction TB
        App[streamlit_app.py]
        User((User))
    end

    %% -- Logic & Orchestration Layer --
    subgraph Logic ["Business Logic"]
        Eval[weather_evaluater.py]
        Clip[hazard_clip.py]
    end

    %% -- Data & Model Layer --
    subgraph Backend ["Models & Processing"]
        Model[model.py]
        Pre[preprocess.py]
        Data[dataset.py]
    end

    %% -- Configuration & Utils --
    subgraph Utils ["Config & Helpers"]
        Conf[config.py]
        Util[utils.py]
        CSVs[weather_scenarios.csv<br/>weather_classification.csv]
    end

    %% -- Relationships --
    User -->|Uploads Image| App
    App -->|Calls| Eval
    
    Eval -->|Uses| Clip
    Eval -->|Prepares Data| Pre
    Eval -->|Reads| CSVs
    
    Clip -->|Loads| Model
    Clip -->|Config| Conf
    
    Data -->|Formats for| Model
    Util -->|Helpers| App
    Util -->|Helpers| Eval

    %% -- Styling --
    style App fill:#ff4b4b,stroke:#333,color:white
    style Eval fill:#e1f5fe,stroke:#0277bd
    style Clip fill:#fff9c4,stroke:#fbc02d
```

```mermaid
graph TD
    %% -- User Interface --
    subgraph UI ["Frontend (Streamlit)"]
        direction TB
        App[streamlit_app.py]
        User((User))
    end

    %% -- Core Logic --
    subgraph Controller ["Evaluator Logic"]
        Eval[weather_evaluater.py]
    end

    %% -- CLIP & AI Models --
    subgraph AI ["CLIP Implementation"]
        direction TB
        Hazard[hazard_clip.py]
        Model[model.py]
        Loader[dataset.py]
        Pre[preprocess.py]
    end

    %% -- Configuration --
    subgraph Config ["Resources"]
        Conf[config.py]
        Utils[utils.py]
        CSV[(weather_scenarios.csv)]
    end

    %% -- Connections --
    User -->|Uploads Image| App
    App -->|Invokes| Eval
    
    %% Evaluator Connections
    Eval -->|Calls| Hazard
    Eval -->|Uses| Utils
    Eval -->|Reads| CSV
    
    %% CLIP Connections
    Hazard -->|Initializes| Model
    Hazard -->|Uses| Pre
    Hazard -->|Uses| Loader
    
    %% Model Connections
    Model -->|Loads Config| Conf
    
    %% Styling
    style App fill:#ff4b4b,stroke:#333,color:white
    style Hazard fill:#e1f5fe,stroke:#0288d1
    style Model fill:#fff9c4,stroke:#fbc02d
    style Eval fill:#e0f2f1,stroke:#00695c
```


