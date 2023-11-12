import pandas as pd
from gmf import GMFEngine
from mlp import MLPEngine
from cnn import CNNEngine
from neumf import NeuMFEngine
from data import SampleGenerator
import argparse

# TODO Parse flags
parser = argparse.ArgumentParser()
parser.add_argument("--model", default=None, help="model to train")
args = parser.parse_args()

# Load Data
print("load data")
data = pd.read_csv("./data/processed/combined1.csv")

print("Range of userId is [{}, {}]".format(data.userId.min(), data.userId.max()))
print("Range of itemId is [{}, {}]".format(data.itemId.min(), data.itemId.max()))
print("Range of rating is [{}, {}]".format(data.rating.min(), data.rating.max()))

num_users = data.userId.max() + 1
num_items = data.itemId.max() + 1

# DataLoader for training
print("initialise sample generator")
sample_generator = SampleGenerator(ratings=data)
evaluate_data = sample_generator.evaluate_data

# Specify the exact model
print("start traing model")

gmf_config = {
    "alias": "gmf",
    "num_epoch": 1,
    "batch_size": 1024,
    # 'optimizer': 'sgd',
    # 'sgd_lr': 1e-3,
    # 'sgd_momentum': 0.9,
    # 'optimizer': 'rmsprop',
    # 'rmsprop_lr': 1e-3,
    # 'rmsprop_alpha': 0.99,
    # 'rmsprop_momentum': 0,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 8,
    "num_negative": 4,
    "l2_regularization": 0,  # 0.01
    "use_cuda": False,
    "device_id": 0,
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

mlp_config = {
    "alias": "mlp",
    "num_epoch": 1,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 8,
    "num_negative": 4,
    "layers": [
        16,
        64,
        32,
        16,
        8,
    ],  # layers[0] is the concat of latent user vector & latent item vector
    "l2_regularization": 0.0000001,  # MLP model is sensitive to hyper params
    "use_cuda": False,
    "device_id": 7,
    "pretrain": False,
    "pretrain_mf": "checkpoints/{}".format(
        "gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model"
    ),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

cnn_config = {
    "alias": "cnn",
    "num_epoch": 1,
    "batch_size": 1024,
    "optimizer": "sgd",
    "sgd_lr": 1e-3,
    "sgd_momentum": 0.9,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim": 9,
    "num_negative": 4,
    "channels": [
        1,
        32,
        16,
    ],
    "l2_regularization": 0.01,
    "use_cuda": False,
    "device_id": 7,
    "pretrain": False,
    "pretrain_mf": "checkpoints/{}".format(
        "gmf_factor8neg4_Epoch100_HR0.6391_NDCG0.2852.model"
    ),
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

neumf_config = {
    "alias": "neumf",
    "num_epoch": 1,
    "batch_size": 1024,
    "optimizer": "adam",
    "adam_lr": 1e-3,
    "num_users": num_users,
    "num_items": num_items,
    "latent_dim_mf": 8,
    "latent_dim_mlp": 8,
    "latent_dim_cnn": 9,
    "num_negative": 4,
    "layers": [
        16,
        32,
        16,
        8,
    ],  # layers[0] is the concat of latent user vector & latent item vector
    "channels": [
        1,
        32,
        16,
    ],
    "l2_regularization": 0.01,
    "use_cuda": False,
    "device_id": 7,
    "pretrain": True,
    "pretrain_mf": "checkpoints/gmf_Epoch0_HR0.4798_NDCG0.2213.model"
    
    # "checkpoints/{}".format(
    #     "gmf_Epoch0_HR0.4798_NDCG0.2213.model"
    # )
     ,
    "pretrain_mlp": "checkpoints/mlp_Epoch0_HR0.8971_NDCG0.6011.model"
    
    # "checkpoints/{}".format(
    #     "mlp_Epoch0_HR0.8971_NDCG0.6011.model"
    # )
    ,
    "pretrain_cnn": "checkpoints/cnn_Epoch0_HR0.4752_NDCG0.2222.model"
    
    # "checkpoints/{}".format(
    #     "cnn_Epoch0_HR0.4752_NDCG0.2222.model"
    # )
    ,
    "model_dir": "checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model",
}

config, engine = None, None
if args.model == "gmf":
    config = gmf_config
    engine = GMFEngine(config)
elif args.model == "mlp":
    config = mlp_config
    engine = MLPEngine(config)
elif args.model == "cnn":
    config = cnn_config
    engine = CNNEngine(config)
elif args.model == "neumf":
    config = neumf_config
    engine = NeuMFEngine(config)

assert config != None
assert engine != None

for epoch in range(config["num_epoch"]):
    print("Epoch {} starts".format(epoch))
    print("-" * 80)
    train_loader = sample_generator.instance_a_train_loader(
        config["num_negative"], config["batch_size"]
    )
    engine.train_an_epoch(train_loader, epoch_id=epoch)
    hit_ratio, ndcg = engine.evaluate(evaluate_data, epoch_id=epoch)
    engine.save(config["alias"], epoch, hit_ratio, ndcg)