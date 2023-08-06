import torch

state_dict = torch.load("./TTVSR_Vimeo90K.pth", map_location="cpu")
print(state_dict.keys())

# new_dict = {}
# for key, value in state_dict.items():
#     if  "step" not in key:
#         new_key = key.split("generator.")[1]
#         new_dict[new_key] = value
#
# torch.save(new_dict,"./ttvsr_vimeo90k.pth")
