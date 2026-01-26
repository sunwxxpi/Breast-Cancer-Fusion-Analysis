import torch
import torch.nn as nn
from torchvision import models


class StackWiseIntegrationModel(nn.Module):
    def __init__(self, mode='both'):
        super(StackWiseIntegrationModel, self).__init__()
        
        self.mode = mode
        
        self.model_ft = models.densenet161(weights='DEFAULT')
        in_ftrs = self.model_ft.classifier.in_features
        self.model_ft.classifier = nn.Linear(in_ftrs, 3)
        
        # self.model_ft = models.regnet_y_8gf(weights='DEFAULT')
        # in_ftrs = self.model_ft.fc.in_features
        # self.model_ft.fc = nn.Linear(in_ftrs, 5)
        
        # self.model_ft = models.shufflenet_v2_x1_5(weights='DEFAULT')
        # in_ftrs = self.model_ft.fc.in_features
        # self.model_ft.fc = nn.Linear(in_ftrs, 5)
        
        # self.model_ft = models.vgg19(weights='DEFAULT')
        # in_ftrs = self.model_ft.classifier[6].in_features
        # self.model_ft.classifier[6] = nn.Linear(in_ftrs, 5)
        
        # self.model_ft = models.efficientnet_v2_s(weights='DEFAULT')
        # num_features = self.model_ft.classifier[1].in_features
        # self.model_ft.classifier[1] = nn.Linear(num_features, 5)

        print(self.model_ft.__class__.__name__)

        if self.mode == 'both':
            self.conv1x1 = nn.Conv2d(6, 3, kernel_size=1, stride=1, padding=0)

    def forward(self, b_mode_input, se_mode_input):
        if self.mode == 'b_mode':
            x = b_mode_input
        elif self.mode == 'se_mode':
            x = se_mode_input
        else: # both
            x = torch.cat((b_mode_input, se_mode_input), dim=1)
            x = self.conv1x1(x)
        
        x = self.model_ft(x)
        
        return x