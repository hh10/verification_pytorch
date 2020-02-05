# verification_pytorch
Cleaned code for work in my master's project on "Provable Verification of NNs"

#Example runs for various experiments:<br/>

1 - To find custom epsilon balls on synthetic dataset:<br/>
	python bounds_syn.py --datasets syn --in_dim 2 --out_dim 2 --dataset_file data/sample_syn_dataset/07_15_1214 --models 2D --model_files data/sample_models/model_for_2D_synthetic_data_07_15_1214/v449_best.pth --batch_size 1 --max_iter 30 --lr 0.01 --mode [semi_nonuniform, uniform, nonuniform]

2 - To find custom epsilon balls on images:<br/>
	python bounds_images.py --datasets mnist --models conv_2layer --model_files data/sample_models/mnist_small_0_3.pth --batch_size 1 --max_iter 300 --lr 0.001 --mode uniform

3 - To provably robustly train a network:<br/>
	python train_main.py --datasets mnist --method wong --warm_start 0 --model_files <specify the model file is required pre-training, else leave for training from scratch> --batch_size 32 --epsilon 0.1 --models conv_2layer --plateau_length 3 --schedule_length 15 --epochs 100

4 - To verify a trained network:<br/>
	python verify_networks.py --datasets mnist --methods wong --model_files data/sample_models/mnist_small_0_3.pth --epsilons 0.1 --models conv_2layer --batch_size 32

5 - To adversarially train a network:<br/>
	python adv_trainer.py --model conv_2layer --batch_size 16 --epsilon 0.2

6 - To store data (bounds) on intermediate activation layers for various analysis, change variables as follows in the 'convex adversarial' package (this package is copied from https://github.com/locuslab/convex_adversarial and modified/extended as per our observations/understanding).<br/>
	- For stability of intermediate activation layer vs increase in epsilon of their neighbourhood, set STABILITY = True in dual_layers.py and set argument --intermediate True.<br/>
		-- It will store the 'l1' change in the activation layers (D matrix corresponding to the activation layer for a given epsilon w.r.t the D matrix of the same layer for epsilon=0.0001 {dd_clean}) in the dicts 'act_change_layers' and 'act_change_layers_per_class' dicts.<br/>
		-- This change will be stored in files named as './exp_res/stability/l1_act_change_filter_'+str(index)+'.npy', where index determines the the state of the activation layer after each subsequent layer applied on it (ToDO: explaining index better, for now, set it to 19 for conv_2layer, 55 for conv_4layer).<br/>
	- For storing max/min activation bounds across the training dataset for OOD analysis, set OOD = True
		-- It will store the max/min bounds for all activation layers in files named  as './exp_res/OOD/max_act_'+str(relu_index)+'.npy', where relu_index is the number of relu layer in the network.<br/>
	- For accounting for correlations by computing bounding hyperplanes after each affine layer, set CH = True (CH for Convex Hull). # try only on linear net

See https://colab.research.google.com/drive/1bMep60RhSZrZ0N5m0_6ejCZO0Ss_2BM6 for SDP example.
