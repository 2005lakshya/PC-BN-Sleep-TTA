"""Small 1D CNN that scores one 30 s EEG epoch (3000 samples at 100 Hz) into 5 stages.

It deliberately sees a single epoch, with no neighbouring context, so that every bit of
temporal reasoning in the experiments comes from the test-time methods being compared.
BatchNorm layers are kept because most test-time adaptation methods act on them.
"""
from torch import nn


def block(c_in, c_out, k, stride=1):
    return nn.Sequential(nn.Conv1d(c_in, c_out, k, stride, padding=k // 2, bias=False),
                         nn.BatchNorm1d(c_out), nn.ReLU(inplace=True))


class SleepCNN(nn.Module):
    def __init__(self, n_classes=5, width=64, dropout=0.5):
        super().__init__()
        self.encoder = nn.Sequential(
            block(1, width, 50, stride=6), nn.MaxPool1d(8, 8), nn.Dropout(dropout),
            block(width, 2 * width, 8), block(2 * width, 2 * width, 8),
            block(2 * width, 2 * width, 8), nn.MaxPool1d(4, 4),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(dropout))
        self.head = nn.Linear(2 * width, n_classes)

    def forward(self, x):
        return self.head(self.encoder(x))

    def bn_params(self):
        return [p for m in self.modules() if isinstance(m, nn.BatchNorm1d)
                for p in (m.weight, m.bias)]
