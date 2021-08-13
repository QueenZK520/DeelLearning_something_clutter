""" 载入预训练权值，避开修改尺寸的子model """
pre_weights = torch.load("load_model_path")  # , map_location=device
# delete classifier weights
pre_dict = {k: v for k, v in pre_weights.items() if self.net.state_dict()[k].numel() == v.numel()}  # numel(): get the length of the input tensor.
missing_keys, unexpected_keys = self.net.load_state_dict(pre_dict, strict=False)
print("missing_keys:{},\n unexpected_keys:{}\n".format(missing_keys, unexpected_keys))

##########################################################################################

base, head = [], []
for name, param in self.model.named_parameters():
    if 'segNet' in name:
        param.requires_grad = False  # 固定权值
        base.append(param)
    else:
        head.append(param)
self.optimizer = torch.optim.Adam([{'params': head}], lr=cfg.lr, betas=cfg.betas, eps=cfg.eps, weight_decay=cfg.weight_decay)

#########################################################################################
