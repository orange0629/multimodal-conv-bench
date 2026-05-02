# Literature Review: Multi-Turn, Multi-Modal Reasoning for VLMs

**Project:** Multimodal Conversational Benchmark (multimodal-conv-bench)  
**Last updated:** April 2026

This review covers the key areas of prior work relevant to building a benchmark for multi-turn, multi-modal conversational visual reasoning. Papers are organized by research area. Each entry includes full citation metadata, a summary of contributions, and a note on its specific relevance to this project.

---

## Table of Contents

1. [Multi-Turn Multimodal Benchmarks](#1-multi-turn-multimodal-benchmarks)
2. [Single-Turn VLM Benchmarks (for contrast)](#2-single-turn-vlm-benchmarks-for-contrast)
3. [VLM Architectures](#3-vlm-architectures)
4. [Visual Dialogue Datasets](#4-visual-dialogue-datasets)
5. [State Tracking in Visual/Grounded Settings](#5-state-tracking-in-visualgrounded-settings)
6. [Belief Revision and Update in LLMs/VLMs](#6-belief-revision-and-update-in-llmsvlms)
7. [Temporal and Causal Visual Reasoning](#7-temporal-and-causal-visual-reasoning)
8. [Coreference and Entity Tracking Across Turns](#8-coreference-and-entity-tracking-across-turns)
9. [Active and Strategic Information Acquisition](#9-active-and-strategic-information-acquisition)
10. [Synthetic Data Generation for VLM Training](#10-synthetic-data-generation-for-vlm-training)
11. [Master Reference Table](#11-master-reference-table)

---

## 1. Multi-Turn Multimodal Benchmarks

These are the most directly related works — existing attempts to evaluate VLMs in multi-turn and/or multi-image conversational settings.

---

### 1.1 MMDialog: A Large-scale Multi-turn Dialogue Dataset Towards Multi-modal Open-domain Conversation
- **Authors:** Jiazhan Feng, Qingfeng Sun, Can Xu, Pu Zhao, Yaming Yang, Chongyang Tao, Dongyan Zhao, Qingwei Lin
- **Year:** 2023 | **Venue:** ACL 2023
- **Summary:** MMDialog is the largest multi-modal conversation dataset by number of dialogues, comprising 1.08 million real-world dialogues with 1.53 million unique images across 4,184 topics — 88× larger than prior datasets by dialogue count. It proposes two response-generation tasks (retrieval and generative) and a novel evaluation metric MM-Relevance for measuring quality of multi-modal responses. The dataset covers open-domain chat where both questions and responses may contain images.
- **Relevance:** Establishes the scale and format for multi-turn, image-rich conversation datasets directly analogous to what multimodal-conv-bench needs for training and evaluation.
- **Link:** https://aclanthology.org/2023.acl-long.405/

---

### 1.2 MMDU: A Multi-Turn Multi-Image Dialog Understanding Benchmark and Instruction-Tuning Dataset for LVLMs
- **Authors:** Ziyu Liu et al.
- **Year:** 2024 | **Venue:** NeurIPS 2024 (Datasets and Benchmarks)
- **Summary:** MMDU is a benchmark comprising 110 high-quality multi-image multi-turn dialogues with 1,600+ questions and detailed long-form answers, featuring dialogues up to 18k image+text tokens, 20 images, and 27 turns — 5× longer than prior benchmarks. An accompanying instruction-tuning dataset MMDU-45k is also released. Analysis of 15 LVLMs shows open-source models lag significantly behind closed-source ones on multi-turn, multi-image tasks, a gap that fine-tuning on MMDU-45k substantially closes.
- **Relevance:** Direct precursor work on multi-image, multi-turn benchmarking for LVLMs — provides both a methodology and a baseline gap that multimodal-conv-bench can build upon.
- **Link:** https://arxiv.org/abs/2406.11833

---

### 1.3 ConvBench: A Multi-Turn Conversation Evaluation Benchmark with Hierarchical Ablation Capability for Large Vision-Language Models
- **Authors:** Shirley Liu et al.
- **Year:** 2024 | **Venue:** NeurIPS 2024 (Datasets and Benchmarks)
- **Summary:** ConvBench comprises 577 curated multi-turn conversations spanning 215 tasks, evaluated via a three-level capability hierarchy (perception, reasoning, creativity) that mimics human cognitive processes. Each level is assessed separately, allowing fine-grained error attribution — a design that reveals weak perception inhibits models' true reasoning and creative capabilities. Even GPT-4V achieves only 39.51%, confirming multi-turn visual conversation remains an open problem.
- **Relevance:** The hierarchical evaluation methodology and error attribution approach are directly applicable to decomposing failure modes in multi-turn incremental visual reasoning.
- **Link:** https://openreview.net/forum?id=PyTf2jj0SH

---

### 1.4 MULTIVERSE: A Multi-Turn Conversation Benchmark for Evaluating Large Vision-Language Models
- **Authors:** Lee et al.
- **Year:** 2025 | **Venue:** ICCV 2025
- **Summary:** MULTIVERSE is the first multi-turn benchmark encompassing diverse tasks with 484 interaction goals spanning factual knowledge, perception, mathematics, and coding. It evaluates models on how well they maintain coherent understanding across turns, not just per-turn accuracy. Diverse topic coverage makes it one of the most comprehensive multi-turn VLM benchmarks to date.
- **Relevance:** State of the art in multi-turn VLM evaluation as of 2025; serves as a direct comparison target.
- **Link:** https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_MultiVerse_A_Multi-Turn_Conversation_Benchmark_for_Evaluating_Large_Vision_and_ICCV_2025_paper.pdf

---

### 1.5 MMCR: Advancing Visual Language Model in Multimodal Multi-Turn Contextual Reasoning
- **Authors:** (Multiple authors)
- **Year:** 2025 | **Venue:** arXiv 2025
- **Summary:** MMCR introduces MMCR-310k, the largest multi-image multi-turn instruction-tuning dataset with 310K contextual dialogues each covering 1–4 images and 4 or 8 dialogue turns, and MMCR-Bench, a diagnostic benchmark spanning 8 domains. The paper demonstrates that contextual reasoning across turns and images requires dedicated data at scale. Models trained on MMCR-310k show significant improvement on multi-turn contextual tasks.
- **Relevance:** Directly addresses the challenge of multi-turn contextual reasoning across multiple images — the core capability the benchmark aims to measure.
- **Link:** https://arxiv.org/abs/2503.18533

---

### 1.6 AlignMMBench: Evaluating Chinese Multimodal Alignment in Large Vision-Language Models
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** AlignMMBench is curated from real-world scenarios, covering 13 specific tasks across three categories with both single-turn and multi-turn dialogue scenarios. It evaluates alignment between visual and linguistic understanding in both English and Chinese. The benchmark reveals systematic alignment failures in Chinese-language multi-turn contexts.
- **Relevance:** Demonstrates the importance of evaluating multi-turn vs. single-turn settings separately; informs design choices around evaluation granularity.

---

### 1.7 MIBench: Evaluating Multimodal Large Language Models over Multiple Images
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** EMNLP 2024
- **Summary:** MIBench is a large-scale benchmark with 13K samples spanning 13 tasks covering three multi-image scenarios: Multi-Image Instruction (general comparison, subtle differences, visual referring, temporal reasoning, logical reasoning), Multimodal Knowledge-Seeking, and Multimodal In-Context Learning. Results show models that excel at single-image tasks show significant shortcomings with multiple images, particularly in fine-grained perception across images.
- **Relevance:** Directly evaluates incremental multi-image understanding central to taxonomy categories 1, 3, and 4.
- **Link:** https://arxiv.org/abs/2407.15272

---

### 1.8 MMSearch: Benchmarking the Potential of Large Models as Multi-Modal Search Engines
- **Authors:** (Multiple authors)
- **Year:** 2024/2025 | **Venue:** ICLR 2025
- **Summary:** MMSearch is a benchmark of 300 manually curated queries across 14 subfields evaluating LMMs as multimodal search engines, encompassing requery, rerank, and summarization tasks, plus an end-to-end multi-step search process. GPT-4o with MMSearch-Engine outperforms Perplexity Pro on end-to-end tasks. The benchmark specifically measures multi-step, multi-modal information gathering and synthesis.
- **Relevance:** Evaluates strategic information acquisition and multi-step reasoning across visual and textual sources — directly relevant to taxonomy category 6 (Strategic Information Acquisition).
- **Link:** https://mmsearch.github.io/

---

### 1.9 MMIE: Massive Multimodal Interleaved Comprehension Benchmark for Large Vision-Language Models
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** MMIE comprises 20K meticulously curated multimodal queries spanning 3 categories, 12 fields, and 102 subfields including mathematics, coding, physics, literature, health, and arts. Unlike prior benchmarks, it specifically targets interleaved image-text understanding, where images and text are intermixed in complex ways throughout a prompt.
- **Relevance:** Evaluates interleaved image-text comprehension that models the kind of incremental multi-image dialogue the project is designed around.

---

## 2. Single-Turn VLM Benchmarks (for contrast)

These are the dominant existing benchmarks for VLMs, all single-turn. They establish what has been accomplished and what gaps remain in the multi-turn setting.

---

### 2.1 MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI
- **Authors:** Xiang Yue et al.
- **Year:** 2024 | **Venue:** CVPR 2024
- **Summary:** MMMU contains 11.5K college-level multimodal questions across 30 subjects in 6 disciplines (Art, Business, Science, Health, Humanities, Engineering), curated from college exams and textbooks. It tests expert-level domain knowledge combined with visual reasoning. State-of-the-art models plateau around 56–65%, well below human-level (~88%).
- **Relevance:** Primary single-turn high-difficulty benchmark; establishes the ceiling for current VLM capabilities in academic domains and the standard against which multi-turn performance gaps are measured.
- **Link:** https://arxiv.org/abs/2311.16502

---

### 2.2 MMBench: Is Your Multi-modal Model an All-around Player?
- **Authors:** Yuan Liu et al.
- **Year:** 2024 | **Venue:** ECCV 2024
- **Summary:** MMBench is a bilingual (English/Chinese) benchmark comprising 3,217 questions covering 20 fine-grained skills using a novel CircularEval strategy that presents multiple-choice options in rotation to reduce ordering bias. It tests perception, reasoning, and knowledge with structured axis decomposition.
- **Relevance:** Standard comparison point for assessing VLM capabilities; its per-capability skill decomposition methodology informs the taxonomy design for multimodal-conv-bench.
- **Link:** https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/00959.pdf

---

### 2.3 ScienceQA: Learn to Explain: Multimodal Reasoning via Thought Chains for Science Question Answering
- **Authors:** Pan Lu et al.
- **Year:** 2022 | **Venue:** NeurIPS 2022
- **Summary:** ScienceQA contains 21,208 multi-choice science questions across natural, language, and social science domains, with 90.5% annotated with detailed chain-of-thought explanations (lectures + rationales). It was the first benchmark to pair multimodal questions with natural language reasoning traces.
- **Relevance:** Chain-of-thought annotation methodology is relevant for designing explanation-style evaluation in multi-turn visual reasoning.
- **Link:** https://arxiv.org/abs/2209.09513

---

### 2.4 VQAv2: Making the V in VQA Matter: Elevating the Role of Image Understanding
- **Authors:** Yash Goyal et al.
- **Year:** 2017 | **Venue:** CVPR 2017
- **Summary:** VQAv2 balances the original VQA dataset by pairing each question with two similar images that yield different correct answers, forcing models to genuinely use visual information rather than exploiting language biases. It became the standard open-ended VQA benchmark and exposed how prior models could achieve high accuracy without attending to images.
- **Relevance:** Foundational single-turn VQA baseline; the bias-reduction methodology (pairing images to force visual grounding) is directly applicable to designing robust multi-turn evaluation.
- **Link:** https://arxiv.org/abs/1612.00837

---

### 2.5 OK-VQA: A Visual Question Answering Benchmark Requiring External Knowledge
- **Authors:** Kenneth Marino, Mohammad Rastegari, Ali Farhadi, Roozbeh Mottaghi
- **Year:** 2019 | **Venue:** CVPR 2019
- **Summary:** OK-VQA contains 14,000+ questions requiring external knowledge (science, history, sports, etc.) beyond what is visually present in images, with all questions manually verified. State-of-the-art VQA models of the era degraded sharply, revealing the gap between visual perception and knowledge-grounded reasoning.
- **Relevance:** Establishes the need for knowledge integration in VQA; multi-turn dialogues often involve follow-up questions requiring world knowledge combined with new visual evidence.
- **Link:** https://arxiv.org/abs/1906.00067

---

### 2.6 TextVQA: Towards VQA Models That Can Read
- **Authors:** Amanpreet Singh et al.
- **Year:** 2019 | **Venue:** CVPR 2019
- **Summary:** TextVQA contains 45,336 questions over 28,408 images requiring models to read and reason about text in the scene. The LoRRA model is proposed as the first to explicitly integrate OCR output into VQA. The benchmark revealed a complete inability of contemporary VQA models to read scene text.
- **Relevance:** Text-reading in images is a component of real-world multi-turn visual dialogue (e.g., annotated images, whiteboards); highlights a specific visual capability the benchmark may need.

---

### 2.7 SEED-Bench: Benchmarking Multimodal LLMs with Generative Comprehension
- **Authors:** Bohao Li et al.
- **Year:** 2023/2024 | **Venue:** CVPR 2024
- **Summary:** SEED-Bench contains 19K multiple-choice questions across 12 evaluation dimensions for both image and video modalities, employing an automated pipeline to filter out questions answerable without visual input and to ensure human-verified accuracy. Subsequent versions expand to 27 and 34 evaluation dimensions.
- **Relevance:** Multi-dimensional evaluation axes covering temporal understanding and spatial comprehension are relevant to the incremental state tracking and temporal reasoning categories of the taxonomy.
- **Link:** https://arxiv.org/abs/2307.16125

---

### 2.8 MMStar: Are We on the Right Way for Evaluating Large Vision-Language Models?
- **Authors:** Guowei Chen, Limin Li et al.
- **Year:** 2024 | **Venue:** NeurIPS 2024
- **Summary:** MMStar identifies two critical problems in existing LVLM evaluation: (1) many benchmark samples do not require visual content (GeminiPro answers 42.9% of MMMU without seeing images), and (2) unintentional data leakage from training. It introduces 1,500 visually indispensable samples with metrics for measuring data leakage and actual multi-modal training gain.
- **Relevance:** Motivates rigorous visual-necessity verification in benchmark design — a key quality requirement for multimodal-conv-bench to ensure that multi-turn visual reasoning cannot be short-circuited by language priors.
- **Link:** https://arxiv.org/abs/2403.20330

---

### 2.9 NaturalBench: Evaluating Vision-Language Models on Natural Adversarial Samples
- **Authors:** Baiqi Li, Zhiqiu Lin et al.
- **Year:** 2024 | **Venue:** NeurIPS 2024
- **Summary:** NaturalBench semi-automatically generates 10,000 human-verified VQA samples by identifying natural image-text pairs where the same question yields different answers across paired images. 53 state-of-the-art VLMs lag 50–70% behind humans, exposing severe biases including attribute binding, object relationship, and counting failures.
- **Relevance:** The paired-image design forcing visual grounding is methodologically relevant; the benchmark exposes confirmation bias and visual neglect — the same failure modes that multi-turn benchmarking should probe.
- **Link:** https://arxiv.org/abs/2410.14669

---

## 3. VLM Architectures

Key model families to evaluate on the benchmark, and their architecturally relevant properties for multi-turn settings.

---

### 3.1 GPT-4 Technical Report / GPT-4V(ision) System Card
- **Authors:** OpenAI (Josh Achiam et al.)
- **Year:** 2023 | **Venue:** OpenAI Technical Report (arXiv:2303.08774); System Card: September 2023
- **Summary:** GPT-4 is a large-scale Transformer-based model accepting image and text inputs, achieving human-expert-level performance on numerous professional benchmarks. GPT-4V extends multimodal capabilities to document images, photographs, diagrams, and screenshots. The System Card documents safety considerations and capabilities across visual grounding and multi-step visual reasoning.
- **Relevance:** Primary closed-source VLM the benchmark should evaluate; GPT-4V/4o represents the strongest commercially deployed multi-turn multimodal conversational model.
- **Link:** https://arxiv.org/abs/2303.08774

---

### 3.2 Flamingo: A Visual Language Model for Few-Shot Learning
- **Authors:** Jean-Baptiste Alayrac et al. (DeepMind)
- **Year:** 2022 | **Venue:** NeurIPS 2022
- **Summary:** Flamingo is an 80B-parameter VLM that bridges frozen vision and language models using Perceiver Resampler modules and gated cross-attention layers, enabling in-context few-shot learning from arbitrarily interleaved image-text sequences. It set new few-shot records across 16 benchmarks without task-specific fine-tuning. Its architecture natively handles multi-image, multi-turn conversation.
- **Relevance:** Architecturally pioneered multi-image interleaved input, the fundamental capability required for multi-turn visual dialogue; serves as an architectural baseline.
- **Link:** https://arxiv.org/abs/2204.14198

---

### 3.3 LLaVA: Visual Instruction Tuning
- **Authors:** Haotian Liu, Chunyuan Li, Qingyang Wu, Yong Jae Lee
- **Year:** 2023 | **Venue:** NeurIPS 2023 (Oral)
- **Summary:** LLaVA is the first model to use language-only GPT-4 to generate 158K multimodal instruction-following samples (conversations, descriptions, complex reasoning), then fine-tunes a CLIP vision encoder + Vicuna LLM with a simple linear projection. It achieves 85.1% of GPT-4 on a multimodal benchmark and 92.53% on ScienceQA, demonstrating that high-quality synthetic instruction data can produce powerful VLMs cheaply.
- **Relevance:** LLaVA and its variants are primary open-source baseline models for the benchmark; its GPT-4-generated instruction data pipeline is directly reusable for generating multi-turn benchmark conversations.
- **Link:** https://arxiv.org/abs/2304.08485

---

### 3.4 LLaVA-1.5: Improved Baselines with Visual Instruction Tuning
- **Authors:** Haotian Liu, Chunyuan Li, Yuheng Li, Yong Jae Lee
- **Year:** 2024 | **Venue:** CVPR 2024
- **Summary:** LLaVA-1.5 replaces LLaVA's linear projection with an MLP connector, scales to CLIP-ViT-L-336px resolution, and adds academic-task VQA data with response-formatting prompts, achieving state-of-the-art across 11 benchmarks using only 1.2M public data in 1 day on 8 A100s.
- **Relevance:** The primary strong open-source baseline to evaluate on multimodal-conv-bench; its modest architecture means multi-turn performance gaps are attributable to training data, not architecture.
- **Link:** https://openaccess.thecvf.com/content/CVPR2024/papers/Liu_Improved_Baselines_with_Visual_Instruction_Tuning_CVPR_2024_paper.pdf

---

### 3.5 InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning
- **Authors:** Wenliang Dai, Junnan Li, Dongxu Li et al. (Salesforce)
- **Year:** 2023 | **Venue:** NeurIPS 2023
- **Summary:** InstructBLIP builds on BLIP-2 by adding an instruction-aware Query Transformer that extracts visual features conditioned on the instruction, then systematically collects 26 public datasets transformed into instruction-following format across diverse tasks. Achieves state-of-the-art zero-shot performance on 13 held-out datasets.
- **Relevance:** Instruction-conditioned visual feature extraction is directly relevant to multi-turn settings where visual attention must track which aspects of an image are relevant given dialogue context.
- **Link:** https://arxiv.org/abs/2305.06500

---

### 3.6 Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond
- **Authors:** Jinze Bai, Shuai Bai et al. (Alibaba Cloud)
- **Year:** 2023 | **Venue:** arXiv 2023
- **Summary:** Qwen-VL introduces a bilingual (Chinese/English) large VLM with capabilities spanning image captioning, VQA, visual grounding, OCR, and document understanding, using a 3-stage training pipeline and a meticulously cleaned multilingual multimodal corpus. It accepts images, text, and bounding boxes as input.
- **Relevance:** Represents the strong open-source multilingual VLM family; its grounding and localization capabilities are especially relevant to entity tracking categories in the taxonomy.
- **Link:** https://arxiv.org/abs/2308.12966

---

### 3.7 Gemini: A Family of Highly Capable Multimodal Models
- **Authors:** Gemini Team, Google DeepMind
- **Year:** 2023/2024 | **Venue:** Technical Report (arXiv:2312.11805); Gemini 1.5: arXiv:2403.05530
- **Summary:** Gemini is natively multimodal, processing arbitrarily interleaved audio, visual, text, and code. Gemini Ultra was the first model to achieve human-expert performance on MMLU and advanced the state of the art on 30/32 benchmarks. Gemini 1.5 uses a mixture-of-experts architecture with a 1-million-token context window, enabling long-form multi-turn interleaved reasoning over very long video/image sequences.
- **Relevance:** Gemini's 1M-token context and native multimodal interleaving make it uniquely relevant for evaluating long-horizon, many-turn, many-image visual conversations.
- **Link:** https://arxiv.org/abs/2312.11805

---

## 4. Visual Dialogue Datasets

Pre-LLM visual dialogue datasets that established the task formulations our benchmark extends.

---

### 4.1 Visual Dialog (VisDial)
- **Authors:** Abhishek Das, Satwik Kottur et al.
- **Year:** 2017/2019 | **Venue:** CVPR 2017; v1.0 released 2019
- **Summary:** VisDial introduces the task of holding a meaningful dialogue with humans about visual content, creating a dataset of 1.2M dialog question-answer pairs on 120K COCO images with 10-round conversations. Evaluation uses MRR and dense-annotation-based NDCG. Version 1.0 added dense human relevance annotations across 100 candidate answers per turn.
- **Relevance:** The foundational multi-turn visual dialogue dataset; provides both the task formulation and evaluation protocol most directly relevant to multimodal-conv-bench.
- **Link:** https://arxiv.org/abs/1611.08669

---

### 4.2 GuessWhat?! Visual Object Discovery through Multi-Modal Dialogue
- **Authors:** Harm de Vries et al.
- **Year:** 2017 | **Venue:** CVPR 2017
- **Summary:** GuessWhat?! is a two-player game where a Questioner asks binary yes/no questions about a scene to identify a hidden target object, while an Oracle answers. The dataset contains 150K human-played games with 800K visual question-answer pairs on 66K COCO images. It is the first large-scale goal-oriented visual dialogue dataset.
- **Relevance:** Directly models the strategic information acquisition process (taxonomy category 6) where an agent must ask questions to identify visual targets.
- **Link:** https://arxiv.org/abs/1611.08481

---

### 4.3 CLEVR-Dialog: A Diagnostic Dataset for Multi-Round Reasoning in Visual Dialog
- **Authors:** Satwik Kottur, José MF Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach
- **Year:** 2019 | **Venue:** NAACL 2019
- **Summary:** CLEVR-Dialog generates 4.25M question-answer pairs across ~85K CLEVR images using a dialogue grammar grounded in CLEVR scene graphs, with 5 instances of 10-round dialogs per image. The grammar ensures controlled variation in entity attributes, spatial relationships, and counting across turns, providing diagnostic signals for each reasoning type.
- **Relevance:** The diagnostic approach — using structured scene graphs to generate compositional multi-round dialogues — is a direct methodological template for the synthetic generation component of multimodal-conv-bench.
- **Link:** https://arxiv.org/abs/1903.03166

---

### 4.4 The PhotoBook Dataset: Building Common Ground through Visually-Grounded Dialogue
- **Authors:** Janosch Haber, Tim Baumgärtner, Ece Takmaz, Lieke Gelderloos, Elia Bruni, Raquel Fernández
- **Year:** 2019 | **Venue:** ACL 2019
- **Summary:** PhotoBook collects 2,502 collaborative multi-round image-identification games between two participants, yielding 164,615 utterances. The key contribution is capturing how common ground is built across rounds — entities are described differently in later rounds as shared context accumulates. This models reference chain evolution across dialogue turns.
- **Relevance:** Directly models how entity references evolve across turns (taxonomy category 3 — Cross-Turn Entity Tracking and Reference Resolution), providing both data and evaluation methodology.
- **Link:** https://arxiv.org/abs/1906.01530

---

### 4.5 DVD: A Diagnostic Dataset for Multi-step Reasoning in Video Grounded Dialogue
- **Authors:** Hung Le, Chinnadhurai Sankar, Seungwhan Moon, Ahmad Beirami, Alborz Geramifard, Satwik Kottur
- **Year:** 2021 | **Venue:** ACL 2021
- **Summary:** DVD is built on 11K CATER synthetic videos, producing 100K+ dialogues and 1M+ QA pairs spanning 10-round conversations per video, with minimal bias and detailed annotations for spatial, temporal, object-tracking, and counting reasoning. Programmatic question generation isolates specific reasoning types and provides diagnostic insight into model failures.
- **Relevance:** Multi-step reasoning over video dialogue with temporal and spatial components maps directly to taxonomy categories 1, 3, and 4; its diagnostic design philosophy directly informs multimodal-conv-bench.
- **Link:** https://arxiv.org/abs/2101.00151

---

### 4.6 VSTAR: A Video-grounded Dialogue Dataset for Situated Semantic Understanding with Scene and Topic Transitions
- **Authors:** (Patrick Tssn et al.)
- **Year:** 2023 | **Venue:** ACL 2023
- **Summary:** VSTAR is constructed from 395 TV series (8,159 episodes) with annotations for scene and topic transitions, forming a dataset of multi-turn video-grounded dialogues. It proposes benchmarks for scene segmentation, topic segmentation, and video-grounded dialogue generation. The dataset captures natural scene changes and topic shifts in long-form dialogue.
- **Relevance:** Scene and topic transitions in long-form video dialogue directly model the incremental state tracking problem (taxonomy category 1) in naturalistic multi-turn settings.
- **Link:** https://aclanthology.org/2023.acl-long.276/

---

## 5. State Tracking in Visual/Grounded Settings

Works on tracking visual entity states, scene changes, and procedural sequences across images.

---

### 5.1 CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning
- **Authors:** Justin Johnson et al.
- **Year:** 2017 | **Venue:** CVPR 2017
- **Summary:** CLEVR contains 100K rendered images with ~1M auto-generated questions in natural language and functional-program form, testing attribute identification, counting, comparison, multi-hop attention, and logical operations. Every question's required reasoning skills are precisely annotated via the functional program. Revealed that state-of-the-art VQA models could not perform compositional visual reasoning.
- **Relevance:** CLEVR's controlled scene graph and functional program formalism is the basis for CLEVR-Dialog and is the architectural template for the synthetic state-tracking scenarios in multimodal-conv-bench.
- **Link:** https://arxiv.org/abs/1612.06890

---

### 5.2 SceneDiff: A Benchmark and Method for Multiview Object Change Detection
- **Authors:** (Multiple authors)
- **Year:** 2024/2025 | **Venue:** arXiv 2024
- **Summary:** SceneDiff is the first multiview change detection benchmark with object instance annotations, containing 350 diverse video pairs with thousands of changed objects. The method leverages pretrained 3D, segmentation, and image encoding models, aligning captures in 3D, extracting object regions, and comparing spatial and semantic features to detect changes.
- **Relevance:** Multi-view object change detection is the computer vision component of incremental state tracking across turns — directly relevant to taxonomy category 1.

---

### 5.3 STAR: A Benchmark for Situated Reasoning in Real-World Videos
- **Authors:** Bo Wu, Shoubin Yu, Zhenfang Chen, Joshua B. Tenenbaum, Chuang Gan
- **Year:** 2021 | **Venue:** NeurIPS 2021 (Datasets and Benchmarks)
- **Summary:** STAR evaluates situated reasoning in real-world action videos with four question types: Interaction, Sequence, Prediction, and Feasibility, using hyper-graphs connecting entities, actions, and relations. It exposes the inability of existing models to generalize beyond language shortcuts when scene state matters.
- **Relevance:** Sequence and prediction question types directly require tracking state changes across time — a core component of multi-turn incremental visual state tracking. STAR's Sequence questions require ordering actions from temporally disordered clips, while Prediction questions test anticipated outcomes from partial observations.
- **Link:** https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/5ef059938ba799aaa845e1c2e8a762bd-Abstract-round2.html

---

### 5.4 RecipeQA: A Challenge Dataset for Multimodal Comprehension of Cooking Recipes
- **Authors:** Semih Yagcioglu, Aykut Erdem et al.
- **Year:** 2018/2019 | **Venue:** EMNLP 2018
- **Summary:** RecipeQA tests procedural knowledge across multiple modalities (text, images, diagrams), requiring models to identify and link entities across steps and track ingredient states (e.g., raw tomato → roasted tomato). It introduced the challenge of tracking object states across sequential procedural steps.
- **Relevance:** State tracking across sequential procedural images is exactly what taxonomy category 1 requires; RecipeQA's entity-state annotation methodology informs the benchmark design.

---

## 6. Belief Revision and Update in LLMs/VLMs

Works on how language and vision-language models update (or fail to update) their beliefs when presented with new evidence.

---

### 6.1 Fundamental Problems with Model Editing: How Should Rational Belief Revision Work in LLMs?
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** This paper frames the model editing problem as belief revision and argues LLMs should maintain logically consistent outputs after knowledge updates. It shows that edits generalize poorly — updating one belief fails to propagate consistently to related beliefs — and that LLM probabilities diverge from Bayesian posteriors, indicating non-rational belief updating.
- **Relevance:** Directly addresses belief revision in LLMs, the theoretical framework for taxonomy category 2 (Belief Revision under Visual Evidence).
- **Link:** https://arxiv.org/abs/2406.19354

---

### 6.2 Failing to Falsify: Evaluating and Mitigating Confirmation Bias in Language Models
- **Authors:** (Multiple authors)
- **Year:** 2024/2025 | **Venue:** arXiv
- **Summary:** This paper demonstrates that LLMs exhibit confirmation bias by over-interpreting ambiguous evidence as confirming prior beliefs and under-updating when faced with contradictory evidence. Two interventions — Dual-Goal and Think-in-Opposites prompting — can meaningfully reduce this bias. Evaluated across reasoning tasks where initial hypothesis formation leads to biased evidence evaluation.
- **Relevance:** Confirmation bias in initial visual interpretations is the central failure mode of taxonomy category 2; the mitigation strategies can be used to design model ablations.

---

### 6.3 VLMs Are Biased
- **Authors:** (Multiple authors)
- **Year:** 2023/2024
- **Summary:** Documents that state-of-the-art VLMs achieve near-100% accuracy on familiar image subjects but only ~17% on counterfactual images with subtle modifications, revealing that models default to memorized knowledge rather than performing actual visual analysis. This is framed as confirmation bias — models confirm what they expect to see rather than what is shown.
- **Relevance:** Directly motivates taxonomy category 2 (Belief Revision under Visual Evidence) — models must revise beliefs based on actual visual input rather than prior expectations.

---

### 6.4 Towards Analyzing and Mitigating Sycophancy in Large Vision-Language Models
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** Studies sycophancy in LVLMs — the tendency to agree with misleading user-provided prior information rather than trusting visual evidence. Visual and cross-modal fusion modules increase output instability. Layer-wise attention modification (amplifying attention to visual tokens in higher layers) mitigates over-reliance on language priors.
- **Relevance:** Sycophancy is the multi-turn version of belief failure — when earlier turns or user suggestions anchor the model to incorrect beliefs it fails to update, precisely what taxonomy category 2 tests.
- **Link:** https://arxiv.org/abs/2408.11261

---

### 6.5 The Adaptability of Large Language Models Reasoning (on Belief Updating)
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** EMNLP 2024
- **Summary:** Investigates how LLMs update their reasoning when presented with contradictory information across turns, finding anti-Bayesian drift where models become more overconfident after encountering counter-arguments rather than rationally updating. Demonstrates that model size alone does not ensure rational belief revision.
- **Relevance:** Anti-Bayesian drift is the failure mode the benchmark's belief revision category measures; provides evaluation metrics (Bayesian Coherence Coefficient, Martingale Score) adaptable to multi-turn VLM evaluation.

---

## 7. Temporal and Causal Visual Reasoning

Works on understanding temporal ordering, causal chains, and event sequencing in visual content.

---

### 7.1 NExT-QA: Next Phase of Question-Answering to Explaining Temporal Actions
- **Authors:** Junbin Xiao, Xindi Shang, Angela Yao, Tat-Seng Chua
- **Year:** 2021 | **Venue:** CVPR 2021
- **Summary:** NExT-QA is a VideoQA benchmark with 5,440 videos and 48K multiple-choice questions specifically targeting causal (why/how) and temporal (before/after) reasoning. Analysis reveals that top models excel at surface description but fail at causal and temporal inference.
- **Relevance:** Causal and temporal question types are the foundation of taxonomy category 4 (Temporal and Causal Reasoning from Sequential Images); NExT-QA is a key baseline to compare against.
- **Link:** https://arxiv.org/abs/2105.08276

---

### 7.2 AGQA: A Benchmark for Compositional Spatio-Temporal Reasoning
- **Authors:** Madeleine Grunde-McLaughlin, Ranjay Krishna, Maneesh Agrawala
- **Year:** 2021/2022 | **Venue:** CVPR 2021; AGQA 2.0: arXiv 2022
- **Summary:** AGQA generates 192M QA pairs for 9.6K videos using handcrafted programs over spatio-temporal scene graphs, testing compositional reasoning including novel-composition generalization, indirect references, and multi-step reasoning chains. Human performance is 86%; best models reach only 47.74% and fail to generalize to unseen compositions.
- **Relevance:** Compositional spatio-temporal reasoning over event sequences is directly analogous to multi-turn incremental visual reasoning; the scene-graph program methodology is reusable for generating benchmark questions.
- **Link:** https://arxiv.org/abs/2103.16002

---

### 7.3 MECD: Unlocking Multi-Event Causal Discovery in Video Reasoning
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** NeurIPS 2024
- **Summary:** MECD introduces causal discovery as a prediction task — event A causes event B if A facilitates predicting B — and evaluates VLMs on identifying causal links between events in video. Models struggle significantly with multi-event causal chains requiring both temporal tracking and causal inference.
- **Relevance:** Multi-event causal discovery over video directly maps to taxonomy category 4; provides both benchmark design patterns and evaluation metrics for causal chains.
- **Link:** https://proceedings.neurips.cc/paper_files/paper/2024/file/a8320b6b9d95798dc286a867c44742a1-Paper-Conference.pdf

---

### 7.4 CausalVQA: A Physically Grounded Causal Reasoning Benchmark for Video Models
- **Authors:** Meta AI Research
- **Year:** 2025 | **Venue:** arXiv 2025
- **Summary:** CausalVQA presents five question types (counterfactual, hypothetical, anticipation, planning, descriptive) grounded in real physical scenarios from web videos. It tests whether models understand the physics of cause and effect, not just temporal correlation. Models significantly underperform on anticipation and planning questions.
- **Relevance:** Anticipation and planning question types require maintaining a causal model across observed and hypothetical future states — directly relevant to taxonomy category 4.

---

### 7.5 VIST: Visual Storytelling (Sequential Image Narrative Dataset)
- **Authors:** Ting-Hao Huang et al.
- **Year:** 2016 | **Venue:** NAACL 2016
- **Summary:** VIST (originally SIND) is the first dataset for sequential vision-to-language, containing 81,743 unique photos in 20,211 five-image sequences, each aligned to descriptive captions and a five-sentence narrative that contextualizes the sequence as a coherent story. This requires generating causally and temporally coherent narratives from image sequences.
- **Relevance:** Sequential image narrative generation is the generative counterpart to the benchmark's evaluation of multi-turn understanding; VIST images and stories can be adapted for generating incremental visual reasoning conversations. Also useful as a source of seed image sequences for data generation.
- **Link:** https://arxiv.org/abs/1604.03968

---

## 8. Coreference and Entity Tracking Across Turns

Works on resolving cross-turn references to entities in visual dialogue, including pronoun resolution and deictic expressions.

---

### 8.1 Visual Coreference Resolution in Visual Dialog Using Neural Module Networks
- **Authors:** Satwik Kottur, José Moura, Devi Parikh, Dhruv Batra, Marcus Rohrbach
- **Year:** 2018 | **Venue:** ECCV 2018
- **Summary:** Introduces neural module networks with Refer and Exclude modules for explicit, fine-grained visual coreference resolution in multi-turn dialogue — linking pronouns and noun phrases to the visual entities they refer to in a COCO image, grounding cross-turn references in actual image regions. Prior approaches handled coreference implicitly or at coarse question level only.
- **Relevance:** Directly models taxonomy category 3 (Cross-Turn Entity Tracking and Reference Resolution) — the first explicit model for visual entity coreference across dialogue turns.
- **Link:** https://arxiv.org/abs/1809.01816

---

### 8.2 What You See is What You Get: Visual Pronoun Coreference Resolution in Dialogues (VisPro)
- **Authors:** Hang Yu, Jing Zhang et al. (HKUST)
- **Year:** 2019 | **Venue:** EMNLP-IJCNLP 2019
- **Summary:** VisPro is a large-scale dataset of 29,722 pronouns from 5,000 VisDial v1.0 dialogues annotated for coreference resolution, along with a visual-aware PCR model (VisCoref) that resolves pronouns by grounding them in image regions across turns. In real dialogue, speakers use pronouns to refer to visible objects without prior introduction, making visual grounding of pronouns non-trivial.
- **Relevance:** VisPro is the primary dataset for the pronoun coreference component of taxonomy category 3; the annotation scheme can be extended to multi-image incremental settings.
- **Link:** https://arxiv.org/abs/1909.00421

---

### 8.3 Modeling Coreference Relations in Visual Dialog
- **Authors:** (Multiple authors)
- **Year:** 2021 | **Venue:** EACL 2021
- **Summary:** This paper models explicit coreference chains across multiple dialogue turns in VisDial, showing that models with explicit coreference modeling outperform memory-based implicit approaches on MRR and NDCG metrics. The coreference annotation includes both pronominal and nominal references across the full 10-turn conversation.
- **Relevance:** Extends coreference modeling to full-dialogue chains, critical for the multi-turn entity tracking the benchmark is designed to test.

---

### 8.4 Who Are You Referring To? Coreference Resolution in Image Narrations
- **Authors:** (University of Edinburgh group)
- **Year:** 2022 | **Venue:** arXiv 2022
- **Summary:** Introduces a dataset of annotated coreference chains with bounding boxes in image narrations, and proposes weakly supervised learning using only image-text pairs and linguistic priors (without explicit coreference annotations) to learn cross-reference resolution. Studies how coreference in image narrations differs from pure text.
- **Relevance:** Multimodal coreference with bounding box grounding is directly applicable to the entity tracking and deictic expression categories in the benchmark's taxonomy.

---

### 8.5 VD-PCR: Improving Visual Dialog with Pronoun Coreference Resolution
- **Authors:** (HKUST group)
- **Year:** 2022 | **Venue:** Pattern Recognition
- **Summary:** Integrates pronoun coreference resolution directly into the visual dialog pipeline, pruning dialogue history by resolving pronouns before passing context to the dialog model. Achieves significant improvement on VisDial NDCG by reducing ambiguity in history encoding. Demonstrates that pronoun resolution improves downstream answer retrieval quality.
- **Relevance:** Shows that proper coreference tracking improves multi-turn visual dialogue performance — motivating the benchmark's focus on entity tracking as an evaluable capability.
- **Link:** https://www.sciencedirect.com/science/article/abs/pii/S0031320322000218

---

## 9. Active and Strategic Information Acquisition

Works on embodied QA, active perception, and goal-driven visual question generation — the basis for taxonomy category 6.

---

### 9.1 EmbodiedQA: Embodied Question Answering
- **Authors:** Abhishek Das, Samyak Datta, Georgia Gkioxari, Stefan Lee, Devi Parikh, Dhruv Batra
- **Year:** 2018 | **Venue:** CVPR 2018
- **Summary:** EmbodiedQA spawns an agent at a random location in a 3D environment who must navigate to gather visual information and then answer a question — requiring language understanding, visual recognition, active perception, goal-driven navigation, commonsense reasoning, and long-term memory. A large-scale EQA dataset in the House3D environment is provided with 750K questions.
- **Relevance:** EmbodiedQA is the foundational task for taxonomy category 6 (Strategic Information Acquisition) — the agent must proactively seek visual information to answer questions.
- **Link:** https://arxiv.org/abs/1711.11543

---

### 9.2 IQA: Visual Question Answering in Interactive Environments
- **Authors:** Daniel Gordon, Aniruddha Kembhavi, Mohammad Rastegari, Joseph Redmon, Dieter Fox, Ali Farhadi
- **Year:** 2018 | **Venue:** CVPR 2018
- **Summary:** IQA extends EmbodiedQA to interactive environments where the agent must manipulate objects (e.g., open a refrigerator) as well as navigate. The IQUAD V1 dataset contains 75K questions in AI2-THOR photo-realistic indoor scenes. The Hierarchical Interactive Memory Network (HIMN) operates at multiple temporal abstraction levels to handle diverse interaction types.
- **Relevance:** Interactive object manipulation to answer questions models the active visual investigation component of taxonomy category 6 — models must decide what additional visual information to request.
- **Link:** https://arxiv.org/abs/1712.03316

---

### 9.3 OpenEQA: Embodied Question Answering in the Era of Foundation Models
- **Authors:** Arjun Majumdar, Anurag Ajay et al. (Meta FAIR)
- **Year:** 2024 | **Venue:** CVPR 2024
- **Summary:** OpenEQA provides 1,600+ open-vocabulary human-generated questions from 180 real-world environments, supporting both episodic memory EQA (answer from past visual observations) and active EQA (take actions to gather information). GPT-4V and other foundation models significantly lag behind human performance, particularly on spatial understanding. The automatic LLM-based evaluation protocol correlates strongly with human judgments.
- **Relevance:** OpenEQA's active EQA task directly models strategic visual information gathering (taxonomy category 6) and its LLM-based evaluation protocol is adaptable to evaluating VLMs in multi-turn settings.
- **Link:** https://openaccess.thecvf.com/content/CVPR2024/papers/Majumdar_OpenEQA_Embodied_Question_Answering_in_the_Era_of_Foundation_Models_CVPR_2024_paper.pdf

---

### 9.4 Goal-Oriented Visual Question Generation via Intermediate Rewards
- **Authors:** (Multiple authors — Das et al. line of work)
- **Year:** 2019–2021 | **Venue:** Various
- **Summary:** Building on GuessWhat?!, this line of work trains Questioner agents to ask minimal, maximally informative questions to identify visual targets. Reinforcement learning approaches with intermediate rewards for information gain outperform supervised baselines and demonstrate emergent question strategies.
- **Relevance:** Goal-driven question asking is the model behavior taxonomy category 6 aims to evaluate — whether VLMs can proactively acquire missing visual information through strategic questions.

---

### 9.5 Modeling Future Conversation Turns to Teach LLMs to Ask Clarifying Questions
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** Trains LLMs to predict the usefulness of future conversation turns to determine when and what to ask for clarification. Shows that models taught to anticipate future context generate more targeted, informative clarifying questions than baseline approaches.
- **Relevance:** Future-turn modeling is directly relevant to designing evaluations of proactive information acquisition in multi-turn VLM dialogue (taxonomy category 6).

---

## 10. Synthetic Data Generation for VLM Training

Works on automatically generating training and evaluation data for VLMs — directly relevant to the benchmark's text-first generation pipeline.

---

### 10.1 MIMIC-IT: Multi-Modal In-Context Instruction Tuning
- **Authors:** (NTU and Microsoft Research)
- **Year:** 2023 | **Venue:** arXiv 2023
- **Summary:** MIMIC-IT is a 2.8M multimodal instruction-response dataset with multi-modal in-context information, generated via the Syphus automatic annotation pipeline combining human expertise with GPT. Multiple images and videos are accepted as in-context input, enabling few-shot instruction following. The associated Otter model demonstrates strong multi-modal in-context learning.
- **Relevance:** MIMIC-IT's multi-image in-context instruction generation pipeline is directly reusable for generating multi-turn multi-image benchmark conversations for multimodal-conv-bench.
- **Link:** https://arxiv.org/abs/2306.05425

---

### 10.2 ShareGPT4V: Improving Large Multi-Modal Models with Better Captions
- **Authors:** Lin Chen, Jinsong Li, Xiaoyi Dong, Pan Zhang et al.
- **Year:** 2024 | **Venue:** ECCV 2024
- **Summary:** ShareGPT4V introduces 1.2M highly descriptive image captions generated via GPT-4V, covering world knowledge, spatial relationships, object properties, and aesthetic evaluations. A 100K high-quality GPT-4V-annotated seed set is expanded to 1.2M via a trained caption model. Fine-tuning on ShareGPT4V significantly improves LLaVA-7B, LLaVA-1.5-13B, and Qwen-VL-Chat.
- **Relevance:** GPT-4V-generated detailed captions can be used as a grounding layer for synthesizing multi-turn benchmark conversations with accurate visual descriptions.
- **Link:** https://arxiv.org/abs/2311.12793

---

### 10.3 SynthVLM: Towards High-Quality and Efficient Synthesis of Image-Caption Datasets
- **Authors:** (Multiple authors)
- **Year:** 2024 | **Venue:** ACM Multimedia 2024
- **Summary:** SynthVLM synthesizes high-quality image-caption pairs by generating detailed captions with an LLM and then synthesizing matching images with text-to-image models, creating training data without human annotation. Evaluated on multiple benchmarks showing competitive performance versus human-annotated datasets.
- **Relevance:** Text-to-image synthesis for benchmark generation provides a scalable alternative to manual image collection, directly relevant to the synthetic scenario generation pipeline for multimodal-conv-bench.

---

### 10.4 Synthetic Dialogue Dataset Generation using LLM Agents
- **Authors:** (Multiple authors)
- **Year:** 2023 | **Venue:** GEM Workshop 2023 (EMNLP)
- **Summary:** Proposes a two-agent LLM framework where one agent acts as the conversational system and another as the user, generating multi-turn dialogues for training purposes. The Multi-Lingual Dialogue Dataset (MLDD) generates 200K multi-turn samples. Shows that LLM-synthesized dialogue achieves quality competitive with human-authored dialogues on multiple metrics.
- **Relevance:** Two-agent LLM dialogue generation is directly applicable to synthesizing multi-turn visual reasoning conversations for the benchmark.

---

### 10.5 CoSyn: Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation
- **Authors:** Yue Yang et al.
- **Year:** 2024 | **Venue:** arXiv 2024
- **Summary:** CoSyn uses text-only LLMs' coding capabilities to automatically generate Python code that renders synthetic text-rich multimodal data (charts, tables, documents), producing 400K images and 2.7M instruction-tuning samples. Models trained on CoSyn data achieve state-of-the-art among open-source models and surpass GPT-4V and Gemini 1.5 Flash on text-rich understanding benchmarks.
- **Relevance:** Code-guided synthetic scene generation is applicable to creating controlled visual scenarios where specific visual properties need to be programmatically guaranteed — a core requirement for reliable benchmark construction.

---

## 11. Master Reference Table

| # | Short Title | Authors | Year | Venue | Category |
|---|-------------|---------|------|-------|----------|
| 1 | MMDialog | Feng et al. | 2023 | ACL | Multi-turn bench |
| 2 | MMDU | Liu et al. | 2024 | NeurIPS | Multi-turn bench |
| 3 | ConvBench | Liu et al. | 2024 | NeurIPS | Multi-turn bench |
| 4 | MULTIVERSE | Lee et al. | 2025 | ICCV | Multi-turn bench |
| 5 | MMCR | Multiple | 2025 | arXiv | Multi-turn bench |
| 6 | AlignMMBench | Multiple | 2024 | arXiv | Multi-turn bench |
| 7 | MIBench | Multiple | 2024 | EMNLP | Multi-turn bench |
| 8 | MMSearch | Multiple | 2025 | ICLR | Multi-turn bench |
| 9 | MMIE | Multiple | 2024 | arXiv | Multi-turn bench |
| 10 | MMMU | Yue et al. | 2024 | CVPR | Single-turn bench |
| 11 | MMBench | Liu et al. | 2024 | ECCV | Single-turn bench |
| 12 | ScienceQA | Lu et al. | 2022 | NeurIPS | Single-turn bench |
| 13 | VQAv2 | Goyal et al. | 2017 | CVPR | Single-turn bench |
| 14 | OK-VQA | Marino et al. | 2019 | CVPR | Single-turn bench |
| 15 | TextVQA | Singh et al. | 2019 | CVPR | Single-turn bench |
| 16 | SEED-Bench | Li et al. | 2024 | CVPR | Single-turn bench |
| 17 | MMStar | Chen et al. | 2024 | NeurIPS | Single-turn bench |
| 18 | NaturalBench | Li, Lin et al. | 2024 | NeurIPS | Single-turn bench |
| 19 | GPT-4V | OpenAI | 2023 | Tech Report | VLM arch |
| 20 | Flamingo | Alayrac et al. | 2022 | NeurIPS | VLM arch |
| 21 | LLaVA | Liu et al. | 2023 | NeurIPS | VLM arch |
| 22 | LLaVA-1.5 | Liu et al. | 2024 | CVPR | VLM arch |
| 23 | InstructBLIP | Dai et al. | 2023 | NeurIPS | VLM arch |
| 24 | Qwen-VL | Bai et al. | 2023 | arXiv | VLM arch |
| 25 | Gemini | Google DeepMind | 2023/24 | Tech Report | VLM arch |
| 26 | VisDial | Das et al. | 2017 | CVPR | Visual dialogue |
| 27 | GuessWhat?! | de Vries et al. | 2017 | CVPR | Visual dialogue |
| 28 | CLEVR-Dialog | Kottur et al. | 2019 | NAACL | Visual dialogue |
| 29 | PhotoBook | Haber et al. | 2019 | ACL | Visual dialogue |
| 30 | DVD | Le et al. | 2021 | ACL | Visual dialogue |
| 31 | VSTAR | Multiple | 2023 | ACL | Visual dialogue |
| 32 | CLEVR | Johnson et al. | 2017 | CVPR | State tracking |
| 33 | SceneDiff | Multiple | 2024 | arXiv | State tracking |
| 34 | STAR | Wu et al. | 2021 | NeurIPS | State tracking |
| 35 | RecipeQA | Yagcioglu et al. | 2018 | EMNLP | State tracking |
| 36 | Belief Revision in LLMs | Multiple | 2024 | arXiv | Belief revision |
| 37 | Confirmation Bias in LMs | Multiple | 2025 | arXiv | Belief revision |
| 38 | VLMs Are Biased | Multiple | 2024 | — | Belief revision |
| 39 | Sycophancy in LVLMs | Multiple | 2024 | arXiv | Belief revision |
| 40 | LLM Reasoning Adaptability | Multiple | 2024 | EMNLP | Belief revision |
| 41 | NExT-QA | Xiao et al. | 2021 | CVPR | Temporal/causal |
| 42 | AGQA | Grunde-McLaughlin et al. | 2021 | CVPR | Temporal/causal |
| 43 | MECD | Multiple | 2024 | NeurIPS | Temporal/causal |
| 44 | CausalVQA | Meta AI | 2025 | arXiv | Temporal/causal |
| 45 | VIST | Huang et al. | 2016 | NAACL | Temporal/causal |
| 46 | Visual Coref NMN | Kottur et al. | 2018 | ECCV | Coreference/entity |
| 47 | VisPro | Yu et al. | 2019 | EMNLP | Coreference/entity |
| 48 | Modeling Coref in VisDial | Multiple | 2021 | EACL | Coreference/entity |
| 49 | Coref in Image Narrations | Edinburgh | 2022 | arXiv | Coreference/entity |
| 50 | VD-PCR | HKUST | 2022 | Pattern Recog. | Coreference/entity |
| 51 | EmbodiedQA | Das et al. | 2018 | CVPR | Strategic acq. |
| 52 | IQA | Gordon et al. | 2018 | CVPR | Strategic acq. |
| 53 | OpenEQA | Majumdar et al. | 2024 | CVPR | Strategic acq. |
| 54 | Goal-Oriented VQG | Das et al. line | 2019–21 | Various | Strategic acq. |
| 55 | Clarifying Questions (LLMs) | Multiple | 2024 | arXiv | Strategic acq. |
| 56 | MIMIC-IT | NTU/Microsoft | 2023 | arXiv | Synthetic data |
| 57 | ShareGPT4V | Chen et al. | 2024 | ECCV | Synthetic data |
| 58 | SynthVLM | Multiple | 2024 | ACM MM | Synthetic data |
| 59 | LLM Synthetic Dialogue | Multiple | 2023 | GEM/EMNLP | Synthetic data |
| 60 | CoSyn | Yang et al. | 2024 | arXiv | Synthetic data |
