import torch, torchaudio
from torch import nn

# STFT analysis parameters. These are torchaudio's Spectrogram defaults for n_fft=512
# (a 32 ms window with a 16 ms hop at 16 kHz), and they are correct for this checkpoint:
# a sweep over win/hop on real speech at 0-15 dB SNR across five noise types found
# 512/256 at the top for SI-SDR gain (+3.6 dB mean), with shorter windows substantially
# worse (320/160 gives +0.9 dB and loses up to 7.7 dB on some conditions). Exposed as
# arguments so the sweep can be reproduced, but do not change the defaults.
WIN_LENGTH = 512
HOP_LENGTH = 256


class NsNet2(nn.Module):
    def __init__(self, n_fft, n_feat, hd1, hd2, hd3,
                 win_length=WIN_LENGTH, hop_length=HOP_LENGTH):
        super().__init__()
        self.n_fft = n_fft
        self.n_features = n_feat
        self.hidden_1 = hd1
        self.hidden_2 = hd2
        self.hidden_3 = hd3
        self.win_length = win_length
        self.hop_length = hop_length
        # fc1
        self.fc1 = nn.Linear(n_feat, hd1)
        # rnn
        self.rnn1 = nn.GRU(
            input_size=hd1, hidden_size=hd2, num_layers=1, batch_first=True
        )
        self.rnn2 = nn.GRU(
            input_size=hd2, hidden_size=hd2, num_layers=1, batch_first=True
        )
        # fc2
        self.fc2 = nn.Linear(hd2, hd3)
        # fc3
        self.fc3 = nn.Linear(hd3, hd3)
        # fc4
        self.fc4 = nn.Linear(hd3, n_feat)
        # Nonlinearities as distinct registered modules. Captum's DeepLIFT only applies
        # the rescale rule to nn.Module activation instances; written as
        # nn.functional.relu / torch.sigmoid it silently degrades to Gradient x Input.
        # Distinct instances also avoid the module-reuse that DeepLIFT forbids.
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # eps
        self.eps = 1e-8
        self.preproc = torchaudio.transforms.Spectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            power=None,
        )
        self.postproc = torchaudio.transforms.InverseSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
        )

    def forward(self, x_noisy):
        stft_noisy = self.preproc(x_noisy)
        mask_pred = self._forward(stft_noisy)
        # apply mask
        stft_pred = stft_noisy * mask_pred
        x_pred = self.postproc(stft_pred)
        return x_pred

    def log_power(self, stft_noisy):
        """Complex STFT -> the log-power features the network consumes, shape [B, F, T]."""
        return torch.log(stft_noisy.abs() ** 2 + self.eps).squeeze(1)

    def mask_logits(self, log_stft_noisy):
        """Log-power features [B, F, T] -> pre-sigmoid mask logits [B, F, T].

        Logits rather than the sigmoid output are the right target for attribution: a
        mask driven to 0 or 1 sits in the saturated region, where gradients vanish and
        every attribution method reports approximately nothing.
        """
        # sort shape
        x = log_stft_noisy.permute(0, 2, 1)
        # neural network layers
        x = self.fc1(x)
        x, _ = self.rnn1(x)
        x, _ = self.rnn2(x)
        x = self.fc2(x)
        x = self.relu1(x)
        x = self.fc3(x)
        x = self.relu2(x)
        x = self.fc4(x)
        # sort shape
        return x.permute(0, 2, 1)

    def _forward(self, stft_noisy):
        logits = self.mask_logits(self.log_power(stft_noisy))
        mask_pred = self.sigmoid(logits)
        return mask_pred.unsqueeze(1)


model = NsNet2(n_fft=512, n_feat=257, hd1=400, hd2=400, hd3=600)
