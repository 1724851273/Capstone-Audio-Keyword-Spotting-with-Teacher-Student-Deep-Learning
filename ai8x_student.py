import ai8x
import torch.nn as nn
from torchaudio.functional import create_dct
import torch

class KWSInfNet(nn.Module):
    
    """
    Student for zero shot KWS
    
    """

    def __init__(
            self,
            emb_size=128,
            num_channels=128,
            bias=True,
            **kwargs

    ):
        super().__init__()
        #TODO: The effect of dropout should be studied
        self.drop = nn.Dropout(p=0.2)       
        self.dct = ai8x.FusedConv1dAbs(num_channels, 128, 1, stride=1, padding=0,
                                                bias=False, **kwargs)
                                                
        dct_coefs = create_dct(n_mfcc=128, n_mels=128, norm=None)
        with torch.no_grad():
            self.dct.op.weight = nn.Parameter(dct_coefs.transpose(0,1)[0:128,:].unsqueeze(2)/14, requires_grad=False)
        # Time: 192 Feature :128
        self.voice_conv1 = ai8x.FusedConv1dReLU(num_channels, 128, 1, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T: 192 F: 128
        self.voice_conv2 = ai8x.FusedConv1dReLU(128, 128, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T: 190 F : 128
        self.voice_conv3 = ai8x.FusedMaxPoolConv1dReLU(128, 64, 3, stride=1, padding=1,
                                                       bias=bias, **kwargs)
        # T: 95 F : 64
        self.voice_conv4 = ai8x.FusedConv1dReLU(64, 48, 3, stride=1, padding=0,
                                                bias=bias, **kwargs)
        # T : 93 F : 48
        self.kws_conv1 = ai8x.FusedMaxPoolConv1dReLU(48, 64, 3, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T: 46 F : 64
        self.kws_conv2 = ai8x.FusedConv1dReLU(64, 128, 3, stride=1, padding=0,
                                              bias=bias, **kwargs)
        # T: 44 F : 128
        self.kws_conv3 = ai8x.FusedAvgPoolConv1dReLU(128, 128, 3, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T : 22 F: 128
        self.kws_conv4 = ai8x.FusedMaxPoolConv1dReLU(128, 128, 3, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T : 11 F: 128
        self.kws_conv5 = ai8x.FusedMaxPoolConv1dReLU(128, 128, 6, stride=1, padding=1,
                                                     bias=bias, **kwargs)
        # T : 2 F: 128 -> flatten to 256
        self.fc = ai8x.Linear(256, emb_size, bias=bias, wide=True, **kwargs)

    def forward(self, x):  # pylint: disable=arguments-differ
        """Forward prop"""
        # Run CNN
        x = self.dct(x)
        x = self.voice_conv1(x)
        x = self.voice_conv2(x)
        #x = self.drop(x)
        x = self.voice_conv3(x)
        x = self.voice_conv4(x)
        #x = self.drop(x)
        x = self.kws_conv1(x)
        x = self.kws_conv2(x)
        #x = self.drop(x)
        x = self.kws_conv3(x)
        x = self.kws_conv4(x)
        x = self.kws_conv5(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x
    