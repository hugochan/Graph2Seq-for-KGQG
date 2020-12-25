# Graph2Seq-for-KGQG
Code &amp; data accompanying the paper ["Toward Subgraph Guided Knowledge Graph Question Generation with Graph Neural Networks"](https://arxiv.org/abs/2004.06015).


## Architecture

![Model architecture.](images/arch.png)

## Get started


### Prerequisites
This code is written in python 3. You will need to install a few python packages in order to run the code.
We recommend you to use `virtualenv` to manage your python packages and environments.
Please take the following steps to create a python virtual environment.

* If you have not installed `virtualenv`, install it with ```pip install virtualenv```.
* Create a virtual environment with ```virtualenv venv```.
* Activate the virtual environment with `source venv/bin/activate`.
* Install the package requirements with `pip install -r requirements.txt`.


In order to compute the meteor score, please download the required data from [here](https://github.com/xinyadu/nqg/blob/master/qgevalcap/meteor/data/paraphrase-en.gz) and put it under the src/core/evaluation/meteor/data folder.


### Run the QG model
* Download the pretrained GloVe word ebeddings [glove.840B.300d.zip](http://nlp.stanford.edu/data/wordvecs/glove.840B.300d.zip) and move `glove.840B.300d.txt` to the `data` folder in this repo.
* Download the data from [here](https://1drv.ms/u/s!AjiSpuwVTt09gVsFilSx0NpJlid-?e=1TKqfG) and move it to the `data` folder in this repo.
* Cd into the `src` folder
* Run the QG model and report the performance
    ```
        python main.py -config config/mhqg-wq/graph2seq.yml
        python main.py -config config/mhqg-pq/graph2seq.yml
    ```
* You can finetune the above trained QG model using RL by running the following command:
    ```
        python main.py -config config/mhqg-wq/rl_graph2seq.yml
        python main.py -config config/mhqg-pq/rl_graph2seq.yml
    ```
* You can find the output data in the `out_dir` folder specified in the config file.


## Reference

If you found this code useful, please consider citing the following paper:

Chen, Yu, Lingfei Wu, and Mohammed J. Zaki. "Toward Subgraph Guided Knowledge Graph Question Generation with Graph Neural Networks." arXiv preprint arXiv:2004.06015 (2020).


    @article{chen2020toward,
      title={Toward Subgraph Guided Knowledge Graph Question Generation with Graph Neural Networks},
      author={Chen, Yu and Wu, Lingfei and Zaki, Mohammed J.},
      journal={arXiv preprint arXiv:2004.06015},
      year={2020}
    }
