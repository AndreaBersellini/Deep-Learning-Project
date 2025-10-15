import torch
import torch.nn as nn

# ----- START DEFINITION OF CONVOLUTIONAL BLOCKS -----

def conv_block(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.ReLU())
    return conv

def conv_block_v1(in_channels, out_channels):
    conv = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        #nn.AvgPool2d(kernel_size = 5, stride = 1, padding = 2),
        nn.Dropout(0.5),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.Dropout(0.5),
        nn.ReLU())
    return conv

# ----- END DEFINITION OF CONVOLUTIONAL BLOCKS -----

# ----- START DEFINITION OF BASELINE UNET -----

class Unet(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, conv_features : list = [64, 128, 256, 512, 1024]):
        super(Unet, self).__init__()

        # CHANNELS OF THE INPUT AND OUTPUT IMAGES
        self.in_channels = in_channels
        self.out_channels = out_channels

        # CHANNELS USED DURING THE CONVOLUTION
        self.conv_features = conv_features

        # LAYERS INITIALIZATION
        self.con_path = nn.ModuleList()
        self.exp_path = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ----- CONTRACTING PATH INITIALIZATION -----

        conv_input = self.in_channels

        for conv_output in self.conv_features[:-1]:
            self.con_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- BOTTLENECK INITIALIZATION -----

        self.bottleneck = conv_block(conv_input, conv_features[-1])
        conv_input = self.conv_features[-1]

        # ----- EXPANSIVE PATH INITIALIZATION -----

        for conv_output in reversed(self.conv_features[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(conv_input, conv_output, kernel_size=2, stride=2))
            self.exp_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- FINAL CONVOLUTION INITIALIZATION -----

        conv_output = self.out_channels
        self.final_layer = nn.Conv2d(conv_input, conv_output, kernel_size=1, stride=1, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        skip_conns = []

        # ----- CONTRACTING PATH -----

        for conv_block in self.con_path:
            x = conv_block(x)
            skip_conns.append(x)
            x = self.max_pool(x)

        # ----- BOTTLENECK -----

        x = self.bottleneck(x)

        # ----- EXPANSIVE PATH -----

        skip_conns = list(reversed(skip_conns))

        for idx, (conv_block, conv_transpose) in enumerate(zip(self.exp_path, self.up_convs)):
            x = conv_transpose(x)
            x = torch.cat((skip_conns[idx], x), dim = 1)
            x = conv_block(x)
        
        # ----- FINAL CONVOLUTION -----

        x = self.final_layer(x)
        
        x = self.sigmoid(x)

        return x

# ----- END DEFINITION OF BASELINE UNET -----

# ----- START DEFINITION OF UNET WITH RAW LOGITS OUTPUT -----

class UnetNoSigmoid(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, conv_features : list = [64, 128, 256, 512, 1024]):
        super(UnetNoSigmoid, self).__init__()

        # CHANNELS OF THE INPUT AND OUTPUT IMAGES
        self.in_channels = in_channels
        self.out_channels = out_channels

        # CHANNELS USED DURING THE CONVOLUTION
        self.conv_features = conv_features

        # LAYERS INITIALIZATION
        self.con_path = nn.ModuleList()
        self.exp_path = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ----- CONTRACTING PATH INITIALIZATION -----

        conv_input = self.in_channels

        for conv_output in self.conv_features[:-1]:
            self.con_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- BOTTLENECK INITIALIZATION -----

        self.bottleneck = conv_block(conv_input, conv_features[-1])
        conv_input = self.conv_features[-1]

        # ----- EXPANSIVE PATH INITIALIZATION -----

        for conv_output in reversed(self.conv_features[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(conv_input, conv_output, kernel_size=2, stride=2))
            self.exp_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- FINAL CONVOLUTION INITIALIZATION -----

        conv_output = self.out_channels
        self.final_layer = nn.Conv2d(conv_input, conv_output, kernel_size=1, stride=1, padding = 0)

    def forward(self, x):

        skip_conns = []

        # ----- CONTRACTING PATH -----

        for conv_block in self.con_path:
            x = conv_block(x)
            skip_conns.append(x)
            x = self.max_pool(x)

        # ----- BOTTLENECK -----

        x = self.bottleneck(x)

        # ----- EXPANSIVE PATH -----

        skip_conns = list(reversed(skip_conns))

        for idx, (conv_block, conv_transpose) in enumerate(zip(self.exp_path, self.up_convs)):
            x = conv_transpose(x)
            x = torch.cat((skip_conns[idx], x), dim = 1)
            x = conv_block(x)
        
        # ----- FINAL CONVOLUTION -----

        x = self.final_layer(x)
        
        # removal of sigmoid -> logit output

        return x

# ----- END DEFINITION OF UNET WITH RAW LOGITS OUTPUT -----

# ----- START DEFINITION OF UNET WITHOUT SKIP CONNECTIONS -----

class UnetNoSkip(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, conv_features : list = [64, 128, 256, 512, 1024]):
        super(UnetNoSkip, self).__init__()

        # CHANNELS OF THE INPUT AND OUTPUT IMAGES
        self.in_channels = in_channels
        self.out_channels = out_channels

        # CHANNELS USED DURING THE CONVOLUTION
        self.conv_features = conv_features

        # LAYERS INITIALIZATION
        self.con_path = nn.ModuleList()
        self.exp_path = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ----- CONTRACTING PATH INITIALIZATION -----

        conv_input = self.in_channels

        for conv_output in self.conv_features[:-1]:
            self.con_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- BOTTLENECK INITIALIZATION -----

        self.bottleneck = conv_block(conv_input, conv_features[-1])
        conv_input = self.conv_features[-1]

        # ----- EXPANSIVE PATH INITIALIZATION -----

        for conv_output in reversed(self.conv_features[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(conv_input, conv_output, kernel_size=2, stride=2))
            self.exp_path.append(conv_block(conv_output, conv_output)) # convolution "out_features -> out_features" to account for missing skip connections
            conv_input = conv_output

        # ----- FINAL CONVOLUTION INITIALIZATION -----

        conv_output = self.out_channels
        self.final_layer = nn.Conv2d(conv_input, conv_output, kernel_size=1, stride=1, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # ----- CONTRACTING PATH -----

        for conv_block in self.con_path:
            x = conv_block(x)
            x = self.max_pool(x)

        # ----- BOTTLENECK -----

        x = self.bottleneck(x)

        # ----- EXPANSIVE PATH -----

        for idx, (conv_block, conv_transpose) in enumerate(zip(self.exp_path, self.up_convs)):
            x = conv_transpose(x)
            x = conv_block(x)
        
        # ----- FINAL CONVOLUTION -----

        x = self.final_layer(x)
        
        x = self.sigmoid(x)

        return x

# ----- END DEFINITION OF UNET WITHOUT SKIP CONNECTIONS -----

# ----- START DEFINITION OF IMPROVED UNET -----

class UnetImproved(nn.Module):
    def __init__(self, in_channels : int, out_channels : int, conv_features : list = [64, 128, 256, 512, 1024]):
        super(UnetImproved, self).__init__()

        # CHANNELS OF THE INPUT AND OUTPUT IMAGES
        self.in_channels = in_channels
        self.out_channels = out_channels

        # CHANNELS USED DURING THE CONVOLUTION
        self.conv_features = conv_features

        # LAYERS INITIALIZATION
        self.con_path = nn.ModuleList()
        self.exp_path = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        self.max_pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        # ----- CONTRACTING PATH INITIALIZATION -----

        conv_input = self.in_channels

        for conv_output in self.conv_features[:-1]:
            self.con_path.append(conv_block(conv_input, conv_output))
            conv_input = conv_output

        # ----- BOTTLENECK INITIALIZATION -----

        self.bottleneck = conv_block(conv_input, conv_features[-1])
        conv_input = self.conv_features[-1]

        # ----- EXPANSIVE PATH INITIALIZATION -----

        for conv_output in reversed(self.conv_features[:-1]):
            self.up_convs.append(nn.ConvTranspose2d(conv_input, conv_output, kernel_size=2, stride=2))
            self.exp_path.append(conv_block_v1(conv_input, conv_output))
            conv_input = conv_output

        # ----- FINAL CONVOLUTION INITIALIZATION -----

        conv_output = self.out_channels
        self.final_layer = nn.Conv2d(conv_input, conv_output, kernel_size=1, stride=1, padding = 0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        skip_conns = []

        # ----- CONTRACTING PATH -----

        for conv_block in self.con_path:
            x = conv_block(x)
            skip_conns.append(x)
            x = self.max_pool(x)

        # ----- BOTTLENECK -----

        x = self.bottleneck(x)

        # ----- EXPANSIVE PATH -----

        skip_conns = list(reversed(skip_conns))

        for idx, (conv_block, conv_transpose) in enumerate(zip(self.exp_path, self.up_convs)):
            x = conv_transpose(x)
            x = torch.cat((skip_conns[idx], x), dim = 1)
            x = conv_block(x)
        
        # ----- FINAL CONVOLUTION -----

        x = self.final_layer(x)
        
        x = self.sigmoid(x)

        return x
    
# ----- END DEFINITION OF IMPROVED UNET -----