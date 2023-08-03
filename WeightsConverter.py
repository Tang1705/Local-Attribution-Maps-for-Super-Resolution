import torch

state_dict = torch.load("./ModelZoo/models/basicvsr_reds4_20120409-0e599677.pth", map_location="cpu")["state_dict"]

new_dict = {}
for key, value in state_dict.items():
    new_key = key.split("generator.")[1]
    new_dict[new_key] = value

torch.save(new_dict,"./ModelZoo/models/basicvsr_reds4.pth")
