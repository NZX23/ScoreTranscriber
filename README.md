# ScoreTranscriber: Performance-to-Score MusicXML Transcription

**ScoreTranscriber** is a research prototype designed to convert **expressive performance MIDI** into **structured MusicXML notation**. The project addresses the challenge of inferring a clean symbolic score from human performances that contain expressive timing, dynamics, microtiming deviations, and occasional performance errors. This repository is part of the project "ScoreTranscriber: Transformer-based Performance MIDI to Musical Score Conversion", conducted in KAIST's Music and Audio Computing Lab (MACLab). 

The project is also the Regional (Asia) Winner at the Global Undergraduate Awards (GUA) 2024. 

---

## **Credits:**
- Research Supervisors: Dr. Taegyun Kwon, Prof. Juhan Nam
- Author: Ng Zheng Xun
- Acknowledgements: Nanyang Technological University (NTU), C.N. Yang Scholars Programme (CNYSP) Office

---

## 1. Overview

ScoreTranscriber implements a **two-stage transformer-based Mixture-of-Experts (MoE)** architecture. The system first interprets the performance MIDI, estimating quantized rhythmic values, identifying potential performance errors, and predicting hand assignments (for piano). A second transformer then maps this structured intermediate representation into full **MusicXML**, producing a human-readable score.

Demonstrations, audio samples, side-by-side score comparisons, and additional evaluation artifacts are available in the links below.

---

## 2. Resources

- **Results page (samples, evaluations, qualitative analysis):**  
  https://sites.google.com/d/1OBCJAFGGM-s8Z_t4rt9vMZEQke1Shvu2/p/10T3wz6Ch1k8q2XS_lLCmQJiq2W1y_Pak/edit?pli=1

- **Slide deck (methodology, architecture, experiments):**  
  https://docs.google.com/presentation/d/1l3i5vTLjxNKw4PshG_RS7glMiWXddaijtHkOUmKcfq8/edit?usp=sharing

---

## 3. System Architecture

ScoreTranscriber uses a **two-transformer Mixture-of-Experts framework**:

<img width="1280" height="442" alt="image" src="https://github.com/user-attachments/assets/38a0001a-2165-4ec9-b640-c5697ac14978" />

### 3.1 Performance Interpreter CRNN
Based on "Performance MIDI-to-score conversion by neural beat tracking" by Liu et al. , 2022
Responsible for:
- **Temporal quantisation** of expressive timing  
- **Error prediction** (distinguishing intended score notes from performance deviations)  
- **Hand assignment** for piano (left/right hand prediction)  
- Construction of a high-level, structured symbolic representation suitable for score generation

### 3.2 Score Generator Transformer
- Consumes the intermediate symbolic representation
- Produces **MusicXML** containing pitch, duration, measure structure, beams, voices, and other notational elements
- Ensures syntactic correctness of the generated XML through constrained prediction

The two components together allow specialization: one model interprets the performance, while the other focuses on structured notation generation.

---

## 4. Technologies and Libraries

ScoreTranscriber relies on the following languages, libraries, and frameworks:

- **Python** — primary development language  
- **PyTorch** — model training, transformer implementation, inference pipelines  
- **librosa** — preprocessing, feature extraction, MIDI/audio-related utilities  
- **MusicXML** — target symbolic format for score representation  
- **MIDI processing**: `mido`, and custom alignment utilities  
- Skills and techniques involved:
  - Transformer sequence modeling  
  - Mixture-of-experts architecture  
  - Representation learning from performance-score pairs  
  - Temporal quantisation and onset clustering  
  - Structured generation of symbolic music formats  

---

## 5. Features

- Conversion of performance MIDI into notated MusicXML  
- Automatic prediction of rhythmic structure and quantisation  
- Performance error detection and correction  
- Piano-specific hand assignment  
- Partial handling of sustain pedal and expressive timing artifacts  

---

## 6. Limitations

### 6.1 Temporal Coverage
The model is currently trained to operate on **4-bar segments** due to computational and data constraints. Longer-form transcription support is planned for future versions.

### 6.2 Stylistic Scope
Training data consists primarily of **Classical piano repertoire** (public-domain or copyright-free resources). As a result:
- Generalization to **other genres**, styles, or instruments is currently limited  
- Non-classical performance characteristics are not yet modeled

### 6.3 Repository Status
This repository is **open-sourced**, with some refactoring to be done.  
Portions of the codebase remain unrefactored, and documentation is incomplete. Ongoing cleanup and reorganization are planned.


For questions or additional information, please refer to the linked resources above or file an issue in this repository.
