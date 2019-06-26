import torch.nn as nn



class Test(nn.Module):

    def __init__(self):
        super(Test, self).__init__()

        self.e1 = nn.Embedding(5, 10)

    def forward(self, *input):
        return 0


if __name__ == "__main__":
    t = Test()

    it = iter(t.parameters())

    print(next(it))

