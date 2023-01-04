#  👨‍💻运用卷积神经网络使用词序进行文本分类

## 💡关于

本项目使用 python 3.10 和 PyTorch 0.4.1 创建

## 📜数据集和使用模型

- [Quest_CNN](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/quest_cnn)：用于文本数据中问题识别的多通道深度卷积神经网络
- [KIM_CNN](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/kim_cnn)： 用于句子分类的卷积神经网络
- [XML_CNN](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/xml_cnn)：极端多标签文本分类的深度学习
- [Seq_cnn](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/seq_cnn)：使用卷积神经网络有效使用词序进行文本分类
- [FastText](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/FastText)：高效文本分类的技巧包
- [CHAR_CNN](https://github.com/Xuyan-cmd/Effective-Use-of-Word-Order-for-Text-Categorization/tree/main/neural_network/char_cnn) :字符级卷积网络

对于每个模型，我在文件夹中提供了额外的自述文件，包含有关如何运行每个模型的说明

## 🚀快速上手

### 提示

建议在虚拟环境中安装和运行代码。

### 创建 Conda 虚拟环境

首先，从此[链接下载 Anaconda](https://www.anaconda.com/distribution/)

其次，用python 3.10创建一个conda环境。

```javascript
$ conda create -n cnn37 python=3.10
```
重新启动终端会话后，你可以激活 conda 环境：
```javascript
$ conda activate cnn37
```
### 安装所需的python包

在项目根目录下，运行以下命令安装所需的包。

```javascript
pip install -r requirements.txt
```

最后，需要下载 NLTK 库中的停用词：

```javascript
python
import nltk
nltk.download()
```

### 下载预训练嵌入

**谷歌预训练嵌入**

为了对词嵌入（或语义嵌入）使用预训练嵌入，需要将 `GoogleNews-vectors-negative300.bin.gz` 下载到文件夹*`embedding_input/google_embedding`*

```
wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

### 参数调整

为了调整每个模型的超参数，你需要创建一个 json 文件，例如*`search_spaces/cnn.json`*文件并将其添加到*`search_spaces/`*

之后，运行：

```javascript
python3 param_json.py --model_name "model_name"  -fn "results_file_name" - -jf "search_spaces/model.json" -st search_trials
```

最终结果将保存在*dataset_output/hyperpameters/*中，它将创建三个文件：

- `results_file_name.csv` ：包含每个搜索试验的所有最终 F1 分数
- `results_file_name.json` ：包含模型的最佳超参数
- `results_file_name_param.csv` ：模型的参数数量

之后，运行：

```javascript
python3 main_iterations.py --model_name "model_name"  -fn "results_file_name"
```

最终结果将保存在*`dataset_output/results/`*中，它将创建两个文件：

- `results_file_name.csv` ：包含每种子的所有结果，测试集的均值和标准差
- `results_file_name_val.csv` ：包含每种子的所有结果，试验集的均值和标准差

为了查看可以更改的所有参数以进行其他实验：

```javascript
python main_iterations.py -help
 

usage: main_iterations.py [-h] [-modn MODEL_NAME] [-fn DATA_FINAL_NAME]
                          [-dn DATA_NAME] [-ner NER] [-df DATA_FILE]
                          [-dft DATA_FILE_TEST] [-dd DATA_FILE_DEV]
                          [-e EMBD_FILE] [-e_mimic EMBD_FILE_MIMIC]
                          [-e_flag EMBEDDING_FLAG] [-cn CLASS_NUMBER]
                          [-tr TRAINING_RATIO] [-tv TEST_VAL_RATIO]
                          [-l EMBEDDING_SIZE] [-opt OPTIM] [-b BATCH_SIZE]
                          [-n NUM_ITERS] [-lr LEARNING_RATE]
                          [-wd WEIGHT_DECAY] [-usemb USE_EMBEDDING]
                          [-tr_e TRAINING_EMBEDDING] [-sp SAVE_PATH]
                          [-pr PRINTING_LOSS] [-multi KMULTICHANNEL]
                          [-dr DROPOUT] [-fm FEATURE_MAPS]
                          [-fs [FILTER_SIZES [FILTER_SIZES ...]]]
                          [-z HIDDEN_SIZE] [-qm QUESTION_NAME]
                          [-qml QUESTION_NAME_LABEL]

optional arguments:
  -h, --help            show this help message and exit
  -modn MODEL_NAME, --model_name MODEL_NAME
                        name of the anmodel we are using
  -fn DATA_FINAL_NAME, --data_final_name DATA_FINAL_NAME
                        result name.
  -dn DATA_NAME, --data_name DATA_NAME
                        Dataset name.
  -ner NER, --ner NER   whether we use ner or re task
  -df DATA_FILE, --data_file DATA_FILE
                        Path to dataset.
  -dft DATA_FILE_TEST, --data_file_test DATA_FILE_TEST
                        Path to dataset test set.
  -dd DATA_FILE_DEV, --data_file_dev DATA_FILE_DEV
                        Path to dataset.
  -e EMBD_FILE, --embd_file EMBD_FILE
                        Path to Embedding File of google.
  -e_mimic EMBD_FILE_MIMIC, --embd_file_mimic EMBD_FILE_MIMIC
                        Path to Embedding File of mimic.
  -e_flag EMBEDDING_FLAG, --embedding_flag EMBEDDING_FLAG
                        1 use google embedding, 2 use mimic dataset, 3 random
                        start
  -cn CLASS_NUMBER, --class_number CLASS_NUMBER
                        Number of class
  -tr TRAINING_RATIO, --training_ratio TRAINING_RATIO
                        Ratio of training set.
  -tv TEST_VAL_RATIO, --test_val_ratio TEST_VAL_RATIO
                        Ratio of testing/validation set.
  -l EMBEDDING_SIZE, --embedding_size EMBEDDING_SIZE
                        embedding size
  -opt OPTIM, --optim OPTIM
                        optimization
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        Batch Size.
  -n NUM_ITERS, --num_iters NUM_ITERS
                        Number of iterations/epochs.
  -lr LEARNING_RATE, --learning_rate LEARNING_RATE
                        Learning rate for optimizer.
  -wd WEIGHT_DECAY, --weight_decay WEIGHT_DECAY
                        weight decay
  -usemb USE_EMBEDDING, --use_embedding USE_EMBEDDING
                        if we use pre-training embedding
  -tr_e TRAINING_EMBEDDING, --training_embedding TRAINING_EMBEDDING
                        If we will continue the training of embedding.
  -sp SAVE_PATH, --save_path SAVE_PATH
                        path where the model will be saved
  -pr PRINTING_LOSS, --printing_loss PRINTING_LOSS
                        whether we print the training loss in each epoch
  -multi KMULTICHANNEL, --kmultichannel KMULTICHANNEL
                        whether we use mutlichannel for Kim
  -dr DROPOUT, --dropout DROPOUT
                        dropout for cnn_text
  -fm FEATURE_MAPS, --feature_maps FEATURE_MAPS
                        size of feature map for each filter
  -fs [FILTER_SIZES [FILTER_SIZES ...]], --filter_sizes [FILTER_SIZES [FILTER_SIZES ...]]
                        size for each filter
  -z HIDDEN_SIZE, --hidden_size HIDDEN_SIZE
                        Number of Units in LSTM layer.
  -qm QUESTION_NAME, --question_name QUESTION_NAME
                        name of the column that contain questions
  -qml QUESTION_NAME_LABEL, --question_name_label QUESTION_NAME_LABEL
```








