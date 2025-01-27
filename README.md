# QLoRA in Code Summarization: How Far Are We?
We conduct the first empirical investigation into QLoRA, focusing specifically on method-level code summarization as a representative task that requires the model to identify patterns and structures across both natural and programming languages. Our study evaluates three state-of the-art CLMs across two programming languages: Python and Java. 

#### How to replicate

* ##### Requirements
  We provide an Anaconda environment with all the necessary dependencies for running our code. The commands to generate an environment from our YAML file are below:
  1. `conda env create -n qlora -f conda_environment.yml`
  2. `conda activate qlora`

* ##### Datasets
  The dataset used for fine-tuning can be found here:
  - [Training Dataset](https://huggingface.co/datasets/google/code_x_glue_ct_code_to_text)
  - [Evaluation Dataset](https://huggingface.co/datasets/doejn771/code_x_glue_ct_code_to_text_java_python)

* ##### QLoRA fine-tuning 💻
  The code to fine-tune models using QLoRA is provided here:
  - [QLoRA Fine-Tuning](https://github.com/doejn771/qlora-summary-replication/tree/main/qlora)

* ##### Full fine-tuning ✍️
  The code for full parameter fine-tuning is provided here:
  - [Full Parameter Fine-Tuning](https://github.com/doejn771/qlora-summary-replication/tree/main/full_finetuning)
 
* ##### Evaluation 📊
  To run an analysis on the resulting fine-tuned models, you can use the scripts provided here:
  - [Analysis](https://github.com/doejn771/qlora-summary-replication/tree/main/model_analysis)

#### Results
* We've added the pickle files (.pkl) for our fine-tuned models to the following Google Drive link:
  - [Model Pickle Files](https://drive.google.com/drive/folders/1eHomJ9dq7_tmAOZLiADwRq8hU_Wrj4oS?usp=sharing)
