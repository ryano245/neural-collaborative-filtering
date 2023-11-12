import torch
from engine import Engine


class CNN(torch.nn.Module):
    def __init__(self, config):
        super(CNN, self).__init__()
        self.config = config
        self.num_users = config["num_users"]
        self.num_items = config["num_items"]
        self.latent_dim = config["latent_dim"]

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )

        self.layers = torch.nn.ModuleList()
        for _, (in_channel, out_channel) in enumerate(
            zip(config["channels"][:-1], config["channels"][1:])
        ):
            self.layers.append(
                torch.nn.Conv2d(
                    in_channel, out_channel, kernel_size=3, stride=1, padding=1
                )
            )
            self.layers.append(torch.nn.BatchNorm2d(out_channel))

        self.affine_output = torch.nn.Linear(
            in_features=config["channels"][-1] * (self.latent_dim**2),
            out_features=1,
        )
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)

        matrix = torch.einsum(
            "bi,bj->bij", user_embedding, item_embedding
        )  # batch outer product
        matrix = matrix.unsqueeze(
            1
        )  # expand to [batch_size, in_channels, height, width]

        for idx in range(len(self.layers) // 2):
            matrix = self.layers[idx](matrix)  # convolutional layer
            matrix = self.layers[idx + 1](matrix)  # batch normalisation

        vector = torch.flatten(matrix, start_dim=1)

        logits = self.affine_output(vector)
        rating = self.logistic(logits)
        return rating

    def init_weight(self):
        pass


class CNNEngine(Engine):
    """Engine for training & evaluating CNN model"""

    def __init__(self, config):
        self.model = CNN(config)
        super(CNNEngine, self).__init__(config)
        print(self.model)