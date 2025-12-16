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
flowchart TD

%% ===================== Class Mapping Box =====================
subgraph CM["6-Class Mapping - FINAL_CLASS_NAMES"]
  CM0["0: Built env"]
  CM1["1: Sky"]
  CM2["2: Vegetation"]
  CM3["3: Object"]
  CM4["4: Sign"]
  CM5["5: Track Rail"]
  CMV["255: Void (ignore)"]
end

%% ===================== 1) Data Preparation =====================
subgraph DP["Data Preparation"]
  A["RailSem19 Dataset<br/>Images + Original Masks"] --> B["Load rs19-config.json"]
  B --> C["Parse original class mappings<br/>19 classes"]
  C --> D["Define Combined Categories<br/>Built env / Sky / Vegetation / Object / Sign / Track Rail<br/>+ Void=255"]
  D --> E["Remap Masks<br/>preprocess.py"]
  E --> F["Processed Masks<br/>6 classes (0..5) + Void=255"]
end

%% ===================== 2) Model Architecture =====================
subgraph MA["Model Architecture"]
  M1["DINOv2 Backbone<br/>facebook/dinov2-base<br/>Pretrained, Frozen"] --> M2["Feature Grid<br/>37x37x768"]
  M2 --> M3["Segmentation Head<br/>Trainable"]
  M3 --> M4["Output Logits<br/>37x37x6"]
end

%% ===================== 3) Training Loop =====================
subgraph TL["Training Loop (train.py)"]
  F --> T1["Load Batch<br/>Images + Remapped Masks"]
  T1 --> T2["Forward Pass<br/>DINOv2 + Seg Head"]
  T2 -. uses .-> M1
  T2 --> T3["Compute Loss<br/>Weighted CrossEntropyLoss<br/>class weights (Object boosted)<br/>ignore_index=255"]
  T3 --> T4["Backprop<br/>Update Seg Head Only<br/>Backbone frozen"]
  T4 --> T5{"Epoch Complete?"}
  T5 -- "No" --> T1
  T5 -- "Yes" --> T6["Save Checkpoint<br/>best_model.pth"]
end

%% ===================== 4) Evaluation / Inference =====================
subgraph EV["Evaluation / Inference"]
  E1["Test / Uploaded Images"] --> E2["Preprocess<br/>AutoImageProcessor -> 518x518"]
  E2 --> E3["Inference<br/>Logits -> Softmax"]
  E3 --> E4["Predicted Mask<br/>6 classes (0..5)"]
  E3 --> E5["Confidence Map"]
  E5 --> E6["Optional Postprocess<br/>low-confidence -> Void=255"]
  E4 --> E7["Compute Metrics<br/>IoU per class + Pixel Accuracy"]
  E6 --> E8["Visualize Results<br/>Overlay / GT / Pred"]
end

D -. "uses mapping" .-> CM

```





