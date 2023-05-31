import torch
import torch.nn as nn
from networks.basic_modules_Masked_AE_Ocean import Inception,Inception_1d

# Time Evolution
class Temporal_Convolution(nn.Module):
    def __init__(self, channel_in = 5, channel_hid = 256, N_T = 4, incep_ker=[3,5,7,11], groups=1):
        super(Temporal_Convolution, self).__init__()
        self.T = channel_in
        self.N_T = N_T
        enc_layers = [Inception(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, input_state):
        S, l, c_hid = input_state.shape
        B = S // self.T
        input_state = input_state.reshape(B,self.T,l,c_hid) # Time-independent Encoder

        # encoder
        skips = []
        state = input_state
        for i in range(self.N_T):
            state = self.enc[i](state)
            if i < self.N_T - 1:
                skips.append(state)

        # decoder
        state = self.dec[0](state)
        for i in range(1, self.N_T):
            state = self.dec[i](torch.cat([state, skips[-i]], dim=1))
        
        state = state.reshape(S,l,c_hid)
        return state


class Temporal_Convolution_1d(nn.Module):
    def __init__(self, channel_in = 5, channel_hid = 256, N_T = 4, incep_ker=[3,5,7,11], groups=1):
        super(Temporal_Convolution_1d, self).__init__()
        self.T = channel_in
        self.N_T = N_T
        enc_layers = [Inception_1d(channel_in, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            enc_layers.append(Inception_1d(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        enc_layers.append(Inception_1d(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))

        dec_layers = [Inception_1d(channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups)]
        for i in range(1, N_T-1):
            dec_layers.append(Inception_1d(2*channel_hid, channel_hid//2, channel_hid, incep_ker= incep_ker, groups=groups))
        dec_layers.append(Inception_1d(2*channel_hid, channel_hid//2, channel_in, incep_ker= incep_ker, groups=groups))

        self.enc = nn.Sequential(*enc_layers)
        self.dec = nn.Sequential(*dec_layers)

    def forward(self, input_state):
        # encoder
        skips = []
        state = input_state
        for i in range(self.N_T):
            state = self.enc[i](state)
            if i < self.N_T - 1:
                skips.append(state)

        # decoder
        state = self.dec[0](state)
        for i in range(1, self.N_T):
            state = self.dec[i](torch.cat([state, skips[-i]], dim=1))        
        return state
