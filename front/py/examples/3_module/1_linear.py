from deepx.nn.modules import Linear

model = Linear(10, 5)
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")
