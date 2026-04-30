import torch
import torchvision.models as models
from torch.fx.experimental.proxy_tensor import make_fx

model = models.vgg16().eval()
x = torch.randn(1, 3, 224, 224)

gm = make_fx(model)(x)
ops = set()

for node in gm.graph.nodes:
    if node.op == "call_function":
        ops.add(node.target.__name__)

print(sorted(ops))
