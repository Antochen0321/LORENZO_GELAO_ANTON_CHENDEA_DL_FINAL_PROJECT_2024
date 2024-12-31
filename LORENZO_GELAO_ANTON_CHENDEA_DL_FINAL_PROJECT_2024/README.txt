This work is made by Anton Chendea and Lorenzo Gelao.

### README

Every code part is independant and works for every task (indicated in the file name).

It's important to put all codes in the same folders as Aptos and DeepDRiD datasets and models (path parameter in codes are based on that).

Task a)

To train a model on the dataset, just run the code, we let the model which had the best result for us (resnet18), it can be change in the class "MyModel".
Fined tuned models and results are saved in files with "part_a" followed by the name of the model performing in the file name.

Task b) 

Same as part a. In this task it's a manual pretrained which is very long (2-3 hours), it's possible to put in commentary the pretraining to test in easier way the training on main dataset.

Task c)

In this part, parameter of attention type can be change in the main part (at the end of the code) between two commentary #CHANGE PARAMETER HERE.
There are 3 models based on resnet34 which are adapted the basic model to attention mechanism.

Task d) 

This part needs to put 3 trained models, one densenet, one resnet and one efficientnet.

Theses models must be called:
model_b_resnet.pth
model_b_densenet.pth
model_b_efficientnet.pth

There are 2 different codes, nothing must be change for these two codes. After running, results for every method we tested will be return.

Task e)

There are two file, code_part_e_wrong is the implementation based on fine-tuned models, this implementation is not working (results are false).
The code_part_e file is just an example of adapted file (here is the part a code adapted) with a train function which is saving metrics into a global array used to plot it at the end of the code. It just need to be run to have an example of this implementation effect.