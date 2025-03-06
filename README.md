# LLMCDSR
This is the official implementation of the TOIS paper "LLMCDSR: Leveraging Large Language Models for Cross-Domain Sequential Recommendation"

## Requirements
- torch == 2.0.1
- transformers == 4.31.0
- higher == 0.2.1

## Data
The processed data used in our work (i.e., Movie-Book and Food-kitchen) are in `./data`.

For convinience, we also release the pre-trained parameters and the textual embeddings of the generated interactions by LLMs. These files can be downloaded at: https://drive.google.com/file/d/1Wn0l60PBN_ZZ84S6KxhZFVom3Tqi-hG6


## Candidate-Free Cross-Domain Interaction Generation

*One can use the ready-made textual embeddings downloaded above to skip this.*

1. Run to get the generations of LLMs.
```shell
cd ./generation
python candidate_generate_icl.py --base_model={LLM_model_path} --output_name={generation_dir_name}
```
2. As generations may not be standard listed results, use LLM again to parse the names of the generated interaction items:
```shell
python parse_items.py --base_model={LLM_model_path} --data_name={generation_dir_name}
```
3. Get the textual embeddings of the parsed generations:
```shell
python get_candidate_embeddings.py --model_path={text_embedding_model_path} --task={dataset} --generation_path={parsed_generation_path}
```

## Collaborative-Textual Contrastive Pre-Training
*One can use the ready-made pre-trained parameters downloaded above to skip this.*

First, get the textual embeddings of the items in the dataset.
```shell
cd ./generation
python get_item_embedding.py --task={dataset} --domain={A|B} --model_path={text_embedding_model_path}
```
Then, run the following commands to pre-train.
```shell
cd ./pre-train
python main.py --dataset={dataset} --domain={A|B}
```
After that, copy the trained parameters into `./pretrained_parameters` fold with the name of `{dataset}_projection_{A|B}.pt`

## Relevance-Aware Meta Recall Network
Once having prepared the needed ingradients, one can simply run the code to train the model and evaluate the performance:
```shell
python main.py --dataset={dataset}
```

## Requirements
If you find our work useful for your research, please cite:
```
@article{xin2025llmcdsr,
  title={LLMCDSR: Enhancing Cross-Domain Sequential Recommendation with Large Language Models},
  author={Xin, Haoran and Sun, Ying and Wang, Chao and Xiong, Hui},
  journal={ACM Transactions on Information Systems},
  year={2025},
  publisher={ACM New York, NY}
}
```
