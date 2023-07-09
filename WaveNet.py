import torch
import torch.nn as nn

class WaveNet_Finetune(nn.Module):
    def __init__(self, 
                 N_in_channels=61, 
                 in_size = (720, 1440), 
                 out_size = (360, 720), 
                 N_out_channels = 2,
                 use_last_step = True):
        """
        N_in_channels: number of channels of backbone model, default is 61
        in_size: size of output from backbone model, default is (720, 1440)
        out_size: size of output from this model 

        N_out_channels: number of channels of this model
        """
        super().__init__()

        self.N_in_channels  = N_in_channels
        self.N_out_channels = N_out_channels
        self.in_size        = in_size
        self.out_size       = out_size
        self.use_last_step  = use_last_step

        # Downscaling to Target Size
        self.conv0 = torch.nn.Conv2d(N_in_channels, N_in_channels, kernel_size=1, stride=2, padding=0)

        # Mapping to Target Channel Dimension
        if use_last_step:
            self.conv = nn.Conv2d(N_in_channels+N_out_channels, N_out_channels, kernel_size=3, stride=1, padding=0, bias=True)
        else:
            self.conv = nn.Conv2d(N_in_channels, N_out_channels, kernel_size=3, stride=1, padding=0, bias=True)

        self.act = nn.ReLU()
        self.norm = nn.GroupNorm(1, N_out_channels)

    def forward(self, x, x_last_step = 0):

        # Downscaling to Target Size
        x = self.conv0(self.act(x))
        print(x.shape, x_last_step.shape)

        if self.use_last_step:
            x = torch.cat((x_last_step, x), dim=1)

        # Mapping to Target Prediction
        x = self.conv(x)
        x = nn.functional.adaptive_avg_pool2d(x, (self.out_size))
        x = self.act(self.norm(x))

        return x
    
if __name__ == "__main__":

    model = WaveNet_Finetune(N_in_channels=61, 
                             in_size = (720, 1440), 
                             out_size = (360, 720), 
                             N_out_channels = 2,
                             use_last_step = True)
    sample = torch.randn(1, 61, 720, 1440)
    last_step = torch.randn(1, 2, 360, 720)
    result = model(sample, last_step)
    print(result.shape)s
    # print(torch.norm(result))
