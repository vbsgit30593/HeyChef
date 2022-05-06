# HeyChef
Our base code is inpired by the facebook research titled - inversecooking which can be found @https://github.com/facebookresearch/inversecooking
We have also generated structural representations using the code inpired by SGN which can be found @https://github.com/hwang1996/SGN

We have made necessary modifications on the base inverse cooking code to make it to work on our setup.
Following are the steps we followed (inspired by inversecooking) - 
# Install conda on colab
* All the steps can be found in the attached IPynb file.
# Install the dependencies from requirements.txt
* pip install -r requirements.txt

# Putting necessary data in place and building vocab.
* Please refer inversecooking repo for instructions 

# LMDB file generation
We couldn't place the generated data on the git repo but it can be generated using the following command. Please follow the directory structure that's provided in this repo
python utils/ims2file.py --recipe1m_path path_to_dataset

* Before training the model, we need to generate the subsampled data. For that, we first mount the data into colab session using RATARMOUNT. Once thats done then we extract subsampled image data using the newly curated JSONified layers. All of these can be found in the notebook

ratarmount recipe1M_images_train.tar /content/drive/MyDrive/inversecooking/dataset_path/images/train_data/ -o nonempty

The data can then be unmounted from the session storage using the following command
fusermount -u /content/drive/MyDrive/inversecooking/dataset_path/images/train_data/

# Training
## Stage 1 - Ingredient prediction from images
python train.py --model_name im2ingr --batch_size 100 --finetune_after 0 --ingrs_only \
--es_metric iou_sample --loss_weight 0 1000.0 1.0 1.0 \
--learning_rate 1e-4 --num_epochs 25 --scale_learning_rate_cnn 1.0 \
--save_dir ../checkpoints --recipe1m_dir dataset_path --aux_data_dir /content/drive/MyDrive/inversecooking/data

## Stage 2 - Recipe generation from images and ingredients (loading from 1.)
python train.py --model_name model --batch_size 64 --recipe_only --transfer_from im2ingr \
--save_dir ../checkpoints --recipe1m_dir dataset_path --notensorboard --num_epochs 20

# Evaluation
python sample.py --model_name model --save_dir ../checkpoints --recipe1m_dir path_to_dataset --greedy --eval_split test
* This generates Accuracy and F1 for ingredients, IoU, precision_all and recall_all

We couldn't upload our best model to the git repo due to size limitations but it can be generated using the steps mentioned above. 

# Qualitative analysis
The code for qualitative analysis between pretrained and subsampled model can be found in the attached notebook.

# Possible execution errors on colab and workarounds
### RuntimeError: CUDA out of memory
* Try reducing the batch size till you get rid of this issue

### nltk error

import nltk

nltk.download('punkt')

### Issue in using generated pickle during qualitative analysis
* Try adding the Vocabulary class within the current context

