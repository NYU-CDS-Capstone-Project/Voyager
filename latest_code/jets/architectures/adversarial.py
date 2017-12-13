import torch
import torch.nn as nn
import torch.nn.functional as F

class AdversarialParticleEmbedding(nn.Module):
    def __init__(self, batch=None, **kwargs):
        super().__init__()
        #self.transform = particle_transform(features=features, hidden=hidden, **kwargs)

        activation_string = 'relu'
        self.activation = getattr(F, activation_string)

        self.fc1 = nn.Linear(1, 1)
        self.fc2 = nn.Linear(1, 1)
        self.fc3 = nn.Linear(1, 3)

        gain = nn.init.calculate_gain(activation_string)
        nn.init.xavier_uniform(self.fc1.weight, gain=gain)
        nn.init.xavier_uniform(self.fc2.weight, gain=gain)
        nn.init.xavier_uniform(self.fc3.weight, gain=gain)
        nn.init.constant(self.fc3.bias, 1)


    def forward(self, jets, **kwargs):
        #out_stuff = self.transform(jets, **kwargs)
        return_extras = kwargs.pop('return_extras', False)
        #if return_extras:
        #    h, extras = out_stuff
        #else:
        #    h = out_stuff
                
        h = self.fc1(jets)
        h = self.activation(h)

        h = self.fc2(h)
        h = self.activation(h)

        #h = F.sigmoid(self.fc3(h))
        h = F.log_softmax(self.fc3(h))
        if return_extras:
            #return (h>=0.5).type(torch.FloatTensor), extras
            return h, extras
        else:
            #return (h>=0.5).type(torch.FloatTensor)
            return h