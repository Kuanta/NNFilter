from Networks.ARNet import ARNet
from Networks.IdentificationNetwork import IdentificationNet
from DeepTorch import Trainer as trn
from DeepTorch.Datasets.SequentialDataset import SequentialDataset
from DeepTorch.Datasets.NumericalDataset import NumericalDataset
from DeepTorch.Functions.regularizations import sparse_regularization, L2_regularization
import torch
import torch.nn as nn
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt

TRAIN = True
RESUME_TRAINING = True
TEST = True
NON_LINEAR = False
MODEL_NAME = "./models/nn/nn_filter"
if NON_LINEAR:
    DATASET = "./data/nl_dataset.mat"
    MODEL_NAME = MODEL_NAME + "_nl"
else:
    DATASET = "./data/dataset3.mat"

# Load data
dataset = loadmat(DATASET)
train = dataset["tr_out"][0][0]
val = dataset["val_out"][0][0]
test = dataset["test_out"][0][0]

tr_reg = np.array([train["y2"]])
tr_reg = np.transpose(tr_reg).squeeze(0)
tr_labels = np.array(train["y1"])

val_reg = np.array([val["y2"]])
val_reg = np.transpose(val_reg).squeeze(0)
val_labels = np.array(val["y1"])

test_reg = np.array([test["y2"]])
test_reg = np.transpose(test_reg).squeeze(0)
test_labels = np.array(test["y1"])

train_dataset = SequentialDataset(tr_reg, tr_labels, n_points=1000, batch_size=-1)
validation_dataset = SequentialDataset(val_reg, val_labels, n_points=1000, batch_size=-1)
test_dataset = SequentialDataset(test_reg, test_labels, n_points=1)
test_dataset.batch_index = 0

net = IdentificationNet(1, 1, 3)
net.to("cuda:0")
net.reset()

def loss_fnc(pred, batch_labels):
    return torch.mean(torch.square(pred - batch_labels))

if TRAIN:
    trn_opts = trn.TrainingOptions()
    if NON_LINEAR:
        trn_opts.batch_size = 5000
        trn_opts.learning_rate = 0.001
        trn_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        trn_opts.learning_rate_drop_factor = 1
        trn_opts.learning_rate_drop_step_count = 100
        trn_opts.n_epochs = 5000
        trn_opts.l2_reg_constant = 1E-2
        trn_opts.saved_model_name = MODEL_NAME
        trn_opts.save_model = True
        trn_opts.optimizer_type = trn.OptimizerType.Adam
        trn_opts.verbose_freq = 500
    else:

        #These optiosn work best for linear case
        trn_opts.batch_size = -1
        trn_opts.learning_rate = 0.001
        trn_opts.learning_rate_drop_type = trn.SchedulerType.StepLr
        trn_opts.learning_rate_drop_factor = 0.9
        trn_opts.learning_rate_drop_step_count = 50
        trn_opts.n_epochs = 1000
        trn_opts.l2_reg_constant = 0.0001
        trn_opts.saved_model_name = MODEL_NAME
        trn_opts.save_model = True
        trn_opts.shuffle_data = False
        trn_opts.optimizer_type = trn.OptimizerType.Adam
        trn_opts.epoch_reset_callback = net.reset
        trn_opts.checkpoint_frequency = 500
        trn_opts.regularization_method = None
        trn_opts.shuffle_data = True
        trn_opts.regularization_hyperparams = [0.01, 0.1]
        trn_opts.validation_batch_size = -1
        trn_opts.weight_decay = 0

    trainer = trn.Trainer(net, trn_opts)
    if RESUME_TRAINING:
        net.load_state_dict(torch.load(MODEL_NAME))
    print(train_dataset.get_iter_count())
    trainer.train(loss_fnc, train_dataset, validation_set=validation_dataset)


    print("Done")

if TEST:
    net.load_state_dict(torch.load(MODEL_NAME))
    net.n_points = 1
    net.reset()
    net.output_feedback = True
    regs, labels = test_dataset.get_batch(-1, 0, "cuda:0")
    test_preds = net.forward(regs).detach().cpu().numpy().squeeze(0)
    test_labels = labels.detach().cpu().numpy().squeeze(0)
    # Regular Test
    # test_regs, test_labels = test_dataset.get_batch(-1, 0, device="cuda:0")
    # test_preds = net.forward(test_regs).detach().cpu().numpy()
    mse = np.mean(np.square(test_labels - test_preds))
    plt.figure()
    plt.plot(np.array(test_labels))
    plt.plot(np.array(test_preds))
    plt.plot(np.array(test["y2"]))
    plt.legend(("Labels", "Prediction", "Y2"))
    plt.title("Regular Test - MSE:{}".format(mse))
    plt.show()



    # In place Test 1
    '''
    In this test, y2 inputs are fed into F(s) and the results are compared with y1
    '''
    # net.reset()  # Reset states
    # test_inputs = np.transpose([test["y2"]]).squeeze(0)
    # past_preds = [0, 0, 0]
    # preds = []
    # for i in range(len(test["tout"])):
    #     # Input must be of rank 3
    #     curr_input = np.expand_dims(np.concatenate((test_inputs[i,:], np.array(past_preds)), axis=0), axis=0)
    #     curr_input = np.expand_dims(curr_input, axis=0)
    #     curr_input = torch.tensor(curr_input).to("cuda:0").float()
    #     pred = net.forward(curr_input).item()
    #     preds.append(pred)
    #     past_preds.insert(0, pred)
    #     past_preds.pop(-1)
    #
    # mse = np.mean(np.square(np.array(test["y1"]) - np.array(preds)))
    # plt.figure()
    # plt.plot(np.array(test["y1"]))
    # plt.legend("Y1")
    # plt.plot(np.array(preds))
    # plt.legend("Prediction")
    # plt.plot(np.array(test["y2"]))
    # plt.legend("Y2")
    # plt.title("In place test 1 - MSE:{}".format(mse))
    # plt.show()

    # # In place Test 2
    # '''
    # In this test, a step input is fed into the F(s)*T2(s) system and the results are compared to T1(s)
    # '''
    # net.reset()
    # diff_eq_coeffs = [0,   0.0828,   -0.0750,    1.5987,   -0.6065];
    # test_inputs = np.transpose([test["u"], test["u_1"], test["u_2"], test["u_3"]]).squeeze(0)
    # past_r2s = [0, 0, 0]
    # past_y2 = [0, 0]
    # control_signals = []
    # preds = []
    # for i in range(len(test["tout"])):
    #     # F(s)
    #     curr_input = np.expand_dims(np.concatenate((test_inputs[i,:], np.array(past_r2s)), axis=0),axis=0)
    #     r2 = net.forward(torch.tensor(curr_input).to("cuda:0").float()).item()
    #     control_signals.append(r2)
    #     past_r2s.insert(0, r2)
    #     past_r2s.pop(-1),
    #
    #     # T2(s)
    #     curr_r2 = np.array([r2, past_r2s[0], past_r2s[1]])
    #     curr_r2 = np.concatenate((curr_r2, np.array(past_y2)), axis=0)
    #     if NON_LINEAR:
    #         curr_r2 = np.sin(curr_r2)
    #     _y2 = np.dot(curr_r2, diff_eq_coeffs)
    #     past_y2.insert(0, _y2)
    #     past_y2.pop(-1)
    #     preds.append(_y2)
    #
    # mse = np.mean(np.square(np.array(test["y1"]) - np.array(preds)))
    # plt.figure()
    # plt.plot(np.array(test["y1"]))
    # plt.legend("Y1")
    # plt.plot(np.array(preds))
    # plt.legend("Prediction")
    # plt.plot(np.array(test["y2"]))
    # plt.legend("Y2")
    # plt.title("In Place Test 2 - MSE:{}".format(mse))
    # plt.show()
    #
    # plt.figure()
    # plt.plot(np.array(control_signals))
    # plt.title("Control Signal in In Place Test 2")
    # plt.show()
