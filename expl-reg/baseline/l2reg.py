import torch

class L2RegWrapper(torch.nn.Module):
    def __init__(self, model):
        super(L2RegWrapper, self).__init__()
        #self.model = model
        self.weight_list_original = self.get_weight_dict(model)
        self.weight_list = self.get_weight(model)

    def get_weight(self, model):
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def get_weight_dict(self, model):
        weight_dict = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight_dict[name] = torch.tensor(param.data)
        return weight_dict

    def regularization_loss(self, weight_list):
        reg_loss = 0
        for name, w in weight_list:

            w_original = self.weight_list_original[name]
            loss = torch.dist(w, w_original, p=2)
            reg_loss += loss
        return reg_loss

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list)
        return reg_loss