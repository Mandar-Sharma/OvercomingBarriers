# Overcoming-Barriers

This repository contains anonymously hosted code-base for the submission **Overcoming Barriers to Skill Injection in Language Modeling: Case Study in Arithmetic** to the Math-AI Workshop NIPS 2022.

Please follow the steps below to setup your environment to run this code-base:
1. Run ```pip install -r requirements.txt```
2. The requirements.txt specifically does not contain pip installation for PyTorch and Huggingface:
  - Please install [PyTorch](https://pytorch.org) appropriate for you CUDA setup.
  - In order to run GLUE tasks, a [source installation](https://huggingface.co/docs/transformers/installation) of Huggingface is required
3. Remember to set the DEVICE_ID appropriately in all the Python files according to your system setup.

So that the appropriate models are trained before evaluation is run, please follow the steps below to replicate the results of the paper:
1. Run ```python fisher_dsiltbert.py & python fisher_dsiltbert_cola.py. & python fisher_dsiltbert_mrpc.py```. This will generate matrices corresponding to Fisher scores for the corresponding models at ./Gradients/DistilBERT/
2. Run ```python arith.py & python arith_ewc.py```. This will train the Base + Arithmetic model and Our model. The models will be saved to the directory ./Models/
3. Run ```python evaluate_arithmetic.py``` for log-RMSE scores for the test-set of the Arithmetic dataset. Please note to specify which model you wish to use this for in ```model = DistilBertForMaskedLM.from_pretrained('')```
4. Download the [GLUE Datasets](https://gluebenchmark.com) into ./Glue/$TASK$ folders.
5. Run ```python run_glue.py --model_name_or_path distilbert-base-uncased --task_name cola --max_seq_length 512 --train_file ./Glue/CoLA/train.tsv --validation_file ./Glue/CoLA/dev.tsv --do_train --do_eval --output_dir ./GlueOut/Base/CoLA --overwrite_output_dir --logging_steps 50 --save_steps 200```. Note, to evaluate glue on the base + arithmetic model & our model, change the --model_name_or_path argument accordingly.
6. With the models trained and the Fisher scores generated, please use the Fisher_Plot.ipynb and NN_Weights_tSNE.ipynb notebooks to generate the plots presented in the paper.

Thank you!

If you find our work to be helpful to your research/projects, please don't forget to cite us and star this repository! :)

Regards,
Overcoming Barries Authors
