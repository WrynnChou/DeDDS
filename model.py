from sklearn.linear_model import LogisticRegression


def get_model(name = "LogisticRegression"):

    if name  == "LogisticRegression":
        model = LogisticRegression()

<<<<<<< HEAD
    return model
=======
    if name == "ANN":

        model = Ann()

    if name == "ANN2":

        model = Ann2()


    return model

class Ann(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(6, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out

class Ann2(nn.Module):

    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.model = nn.Sequential(
            nn.Linear(4, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out

if __name__ == '__main__':

    ann = Ann()
    input = torch.zeros(6)
    output = ann(input)

    print(output)
>>>>>>> 461b612 (README)
