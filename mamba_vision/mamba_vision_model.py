import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

class MambaVision(nn.Module):
    def __init__(self):
        super().__init__()

        # ---- stem stage ----
        self.stem_conv1 = nn.Conv2d(in_channels=3, out_channels=32,
                                    kernel_size=3, stride=2, padding=1,
                                    bias=False)
        self.stem_bn1 = nn.BatchNorm2d(32, eps=1e-4)
        self.stem_conv2 = nn.Conv2d(in_channels=32, out_channels=80,
                                    kernel_size=3, stride=2, padding=1,
                                    bias=False)
        self.stem_bn2 = nn.BatchNorm2d(80, eps=1e-4)

        # ---- stage 1: first conv stage ----
        self.s1_0_conv1 = nn.Conv2d(in_channels=80, out_channels=80,
                                    kernel_size=3, stride=1, padding=1)
        self.s1_0_norm1 = nn.BatchNorm2d(80, eps=1e-5)
        self.s1_0_conv2 = nn.Conv2d(in_channels=80, out_channels=80,
                                    kernel_size=3, stride=1, padding=1)
        self.s1_0_norm2 = nn.BatchNorm2d(80, eps=1e-5)
        self.ds1 = nn.Conv2d(in_channels=80, out_channels=160,
                             kernel_size=3, stride=2, padding=1,
                             bias=False)
        
        # ---- stage 2: second conv stage with 3 repeat blocks ----
        for i in range(3):
            setattr(self, f"s2_{i}_conv1", nn.Conv2d(in_channels=160,
                                                     out_channels=160,
                                                     kernel_size=3,
                                                     stride=1, padding=1))
            setattr(self, f"s2_{i}_norm1", nn.BatchNorm2d(160, eps=1e-5))
            setattr(self, f"s2_{i}_conv2", nn.Conv2d(in_channels=160,
                                                     out_channels=160,
                                                     kernel_size=3,
                                                     stride=1, padding=1))
            setattr(self, f"s2_{i}_norm2", nn.BatchNorm2d(160, eps=1e-5))     
        self.ds2 = nn.Conv2d(in_channels=160, out_channels=320,
                             kernel_size=3, stride=2, padding=1,
                             bias=False)
        
        # ---- stage 3: first hybrid stage with layout MMMMAAAA ----
        for i in range(4):
            stage_name = f"s3_{i}"
            self._init_mamba(stage_name, dim=320)
            self._init_mlp(stage_name, dim=320)
        for i in range(4,8):
            stage_name = f"s3_{i}"
            self._init_attention(stage_name, dim=320, head=8)
            self._init_mlp(stage_name, dim=320)
        self.ds3 = nn.Conv2d(in_channels=320, out_channels=640,
                             kernel_size=3, stride=2, padding=1,
                             bias=False)

        # ---- stage 4: second hybrid stage with layout MMAA ----
        for i in range(2):
            stage_name = f"s4_{i}"
            self._init_mamba(stage_name, dim=640)
            self._init_mlp(stage_name, dim=640)
        for i in range(2,4):
            stage_name = f"s4_{i}"
            self._init_attention(stage_name, dim=640, head=16)
            self._init_mlp(stage_name, dim=640)

        self.norm = nn.BatchNorm2d(640, eps=1e-5)
        self.classifier = nn.Linear(640, 1000)
        self.gelu = nn.GELU(approximate="tanh")

        # ---- Model meta data ----
        self.meta = {'mean': [0.485, 0.456, 0.406], #RGB
                     'std': [0.229, 0.224, 0.225],
                     'image_size': [224, 224, 3],
                     'stage_size': [1, 3, 8, 4],
                     'layers': ['stem_conv1', 'stem_bn1', 'stem_conv2', 'stem_bn2', 
                                's1_0_conv1', 's1_0_norm1', 's1_0_conv2', 's1_0_norm2', 'ds1', 
                                's2_0_conv1', 's2_0_norm1', 's2_0_conv2', 's2_0_norm2', 
                                's2_1_conv1', 's2_1_norm1', 's2_1_conv2', 's2_1_norm2', 
                                's2_2_conv1', 's2_2_norm1', 's2_2_conv2', 's2_2_norm2', 'ds2', 
                                's3_0_norm1', 's3_0_in_project', 's3_0_x_project', 's3_0_delta_project', 's3_0_out_project', 's3_0_norm2', 's3_0_mlp_fc1', 's3_0_mlp_fc2', 
                                's3_1_norm1', 's3_1_in_project', 's3_1_x_project', 's3_1_delta_project', 's3_1_out_project', 's3_1_norm2', 's3_1_mlp_fc1', 's3_1_mlp_fc2', 
                                's3_2_norm1', 's3_2_in_project', 's3_2_x_project', 's3_2_delta_project', 's3_2_out_project', 's3_2_norm2', 's3_2_mlp_fc1', 's3_2_mlp_fc2', 
                                's3_3_norm1', 's3_3_in_project', 's3_3_x_project', 's3_3_delta_project', 's3_3_out_project', 's3_3_norm2', 's3_3_mlp_fc1', 's3_3_mlp_fc2', 
                                's3_4_norm1', 's3_4_qkv', 's3_4_project', 's3_4_norm2', 's3_4_mlp_fc1', 's3_4_mlp_fc2', 
                                's3_5_norm1', 's3_5_qkv', 's3_5_project', 's3_5_norm2', 's3_5_mlp_fc1', 's3_5_mlp_fc2', 
                                's3_6_norm1', 's3_6_qkv', 's3_6_project', 's3_6_norm2', 's3_6_mlp_fc1', 's3_6_mlp_fc2', 
                                's3_7_norm1', 's3_7_qkv', 's3_7_project', 's3_7_norm2', 's3_7_mlp_fc1', 's3_7_mlp_fc2', 'ds3', 
                                's4_0_norm1', 's4_0_in_project', 's4_0_x_project', 's4_0_delta_project', 's4_0_out_project', 's4_0_norm2', 's4_0_mlp_fc1', 's4_0_mlp_fc2', 
                                's4_1_norm1', 's4_1_in_project', 's4_1_x_project', 's4_1_delta_project', 's4_1_out_project', 's4_1_norm2', 's4_1_mlp_fc1', 's4_1_mlp_fc2', 
                                's4_2_norm1', 's4_2_qkv', 's4_2_project', 's4_2_norm2', 's4_2_mlp_fc1', 's4_2_mlp_fc2', 
                                's4_3_norm1', 's4_3_qkv', 's4_3_project', 's4_3_norm2', 's4_3_mlp_fc1', 's4_3_mlp_fc2', 
                                'norm', 'classifier']}

    #########################
    # Mamba
    #
    def _init_mamba(self, stage_name, dim):
        # the size of hidden state in SSM
        state = 8
        # the rank for producing delta in SSM
        delta_rank = math.ceil(dim / 16)
        setattr(self, f"{stage_name}_in_project", nn.Linear(dim, dim,
                                                         bias=False))
        setattr(self, f"{stage_name}_x_project", nn.Linear(dim // 2,
                                                        delta_rank + 2 * state,
                                                        bias=False))
        setattr(self, f"{stage_name}_delta_project", nn.Linear(delta_rank, dim // 2,
                                                         bias=True))
        setattr(self, f"{stage_name}_out_project", nn.Linear(dim, dim,
                                                         bias=False))
        setattr(self, f"{stage_name}_conv1d_x1", nn.Conv1d(in_channels=dim // 2,
                                                          out_channels=dim // 2,
                                                          kernel_size=3,
                                                          groups=dim // 2,
                                                          bias=False))
        setattr(self, f"{stage_name}_conv1d_x2", nn.Conv1d(in_channels=dim // 2,
                                                          out_channels=dim // 2,
                                                          kernel_size=3,
                                                          groups=dim // 2,
                                                          bias=False))
        A = repeat(torch.arange(1, state + 1, dtype=torch.float32), " n -> d n", d=dim // 2).contiguous()
        setattr(self, f"{stage_name}_A_log", nn.Parameter(torch.log(A)))
        setattr(self, f"{stage_name}_D", nn.Parameter(torch.ones(dim // 2)))

    def _scan(self, x, delta, A, B, C, D, delta_bias):
        x, delta, A, B, C, D = (i.float() for i in (x, delta, A, B, C, D))
        delta = F.softplus(delta + delta_bias[..., None].float())
        batch, dim, state = x.shape[0], A.shape[0], A.shape[1]
        dA = torch.exp(torch.einsum("bdl,dn->bdln", delta, A))
        dBx = torch.einsum("bdl,bnl,bdl->bdln", delta, B, x)
        h = x.new_zeros(batch, dim, state)
        length_of_input = x.shape[2]
        y = []
        # there are dim SSMs, one channel one SSM
        for i in range(length_of_input):
            # h = Ah + Bx
            h = dA[:, :, i] * h + dBx[:, :, i]
            # y = Ch
            y.append(torch.einsum("bdn,bn->bd", h, C[:, :, i]))
        return torch.stack(y, 2) + x * D[None, :, None]
    
    # mambavision's mamba
    def _mamba(self, x, in_project, x_project, delta_project, out_project,
               conv1d_x1, conv1d_x2, A_log, D):
        # window partition lead to x's shape become (batch length dim)
        L = x.shape[1]
        A = -torch.exp(A_log.float())

        half_of_input = in_project.out_features // 2
        x1_x2 = rearrange(in_project(x), "b l d -> b d l")
        x1, x2 = x1_x2.chunk(2, dim=1)
        # the branch
        x1 = F.silu(F.conv1d(input=x1, weight=conv1d_x1.weight,
                             bias=None, padding="same", groups=half_of_input))
        x2 = F.silu(F.conv1d(input=x2, weight=conv1d_x2.weight,
                             bias=None, padding="same", groups=half_of_input))
        
        delta, B, C = torch.split(x_project(rearrange(x1, "b d l -> (b l) d")),
                                 [delta_project.in_features, A_log.shape[1], A_log.shape[1]],
                                  dim=-1)
        delta = rearrange(delta_project(delta), "(b l) d -> b d l", l=L)
        B = rearrange(B, "(b l) n -> b n l", l=L).contiguous()
        C = rearrange(C, "(b l) n -> b n l", l=L).contiguous()
        y = self._scan(x1, delta, A, B, C, D, delta_project.bias)
        y = rearrange(torch.cat([y, x2], dim=1), "b d l -> b l d")
        return out_project(y)

    #########################
    # Attention
    #
    def _init_attention(self, stage_name, dim, head):
        setattr(self, f"{stage_name}_qkv", nn.Linear(dim, dim * 3, bias=True))
        setattr(self, f"{stage_name}_project", nn.Linear(dim, dim, bias=True))

    def _attention(self, x, qkv, project, head):
        # 3, 196, 320
        # 196 words which are represented by 320 dim vector
        b, l, d = x.shape
        slice = d // head
        # b, l, 3, head, slice = b, 196, 3, 8, 40
        # 960 = 3 * 8 * 40
        q, k, v = qkv(x).reshape(b, l, 3, head, slice).permute(2, 0, 3, 1, 4).unbind(0)
        output = F.scaled_dot_product_attention(q, k, v)
        return project(output.transpose(1,2).reshape(b, l, d))
    
    #########################
    # MLP
    #   
    def _init_mlp(self, stage_name, dim):
        setattr(self, f"{stage_name}_norm1", nn.LayerNorm(dim))
        setattr(self, f"{stage_name}_norm2", nn.LayerNorm(dim))
        setattr(self, f"{stage_name}_mlp_fc1", nn.Linear(dim, dim * 4, bias=True))
        setattr(self, f"{stage_name}_mlp_fc2", nn.Linear(dim * 4, dim, bias=True))

    def forward(self, x):
        # stem
        x = F.relu(self.stem_bn1(self.stem_conv1(x)))
        x = F.relu(self.stem_bn2(self.stem_conv2(x)))
        # b,80,56,56

        # stage 1 conv then downsample
        x = x + self.s1_0_norm2(self.s1_0_conv2(self.gelu(self.s1_0_norm1(self.s1_0_conv1(x)))))
        x = self.ds1(x)
        # b,160,28,28
        
        # stage 2 conv then downsample
        x = x + self.s2_0_norm2(self.s2_0_conv2(self.gelu(self.s2_0_norm1(self.s2_0_conv1(x)))))
        x = x + self.s2_1_norm2(self.s2_1_conv2(self.gelu(self.s2_1_norm1(self.s2_1_conv1(x)))))
        x = x + self.s2_2_norm2(self.s2_2_conv2(self.gelu(self.s2_2_norm1(self.s2_2_conv1(x)))))
        x = self.ds2(x)
        # b,320,14,14

        # this following code is for replacing window partition
        # which is not needed in mambavision-T
        # b,320,14,14 -> b,320,196 -> b,196,320
        x = x.flatten(2).transpose(1, 2)

        # stage 3 hybrid MMMMAAAA
        # M
        x = x + self._mamba(self.s3_0_norm1(x), self.s3_0_in_project, self.s3_0_x_project,
                            self.s3_0_delta_project, self.s3_0_out_project,
                            self.s3_0_conv1d_x1, self.s3_0_conv1d_x2,
                            self.s3_0_A_log, self.s3_0_D)
        x = x + self.s3_0_mlp_fc2(self.gelu(self.s3_0_mlp_fc1(self.s3_0_norm2(x))))
        x = x + self._mamba(self.s3_1_norm1(x), self.s3_1_in_project, self.s3_1_x_project,
                            self.s3_1_delta_project, self.s3_1_out_project,
                            self.s3_1_conv1d_x1, self.s3_1_conv1d_x2,
                            self.s3_1_A_log, self.s3_1_D)
        x = x + self.s3_1_mlp_fc2(self.gelu(self.s3_1_mlp_fc1(self.s3_1_norm2(x))))
        x = x + self._mamba(self.s3_2_norm1(x), self.s3_2_in_project, self.s3_2_x_project,
                            self.s3_2_delta_project, self.s3_2_out_project,
                            self.s3_2_conv1d_x1, self.s3_2_conv1d_x2,
                            self.s3_2_A_log, self.s3_2_D)
        x = x + self.s3_2_mlp_fc2(self.gelu(self.s3_2_mlp_fc1(self.s3_2_norm2(x))))
        x = x + self._mamba(self.s3_3_norm1(x), self.s3_3_in_project, self.s3_3_x_project,
                            self.s3_3_delta_project, self.s3_3_out_project,
                            self.s3_3_conv1d_x1, self.s3_3_conv1d_x2,
                            self.s3_3_A_log, self.s3_3_D)
        x = x + self.s3_3_mlp_fc2(self.gelu(self.s3_3_mlp_fc1(self.s3_3_norm2(x))))

        # A
        x = x + self._attention(self.s3_4_norm1(x), self.s3_4_qkv, self.s3_4_project, 8)
        x = x + self.s3_4_mlp_fc2(self.gelu(self.s3_4_mlp_fc1(self.s3_4_norm2(x))))
        x = x + self._attention(self.s3_5_norm1(x), self.s3_5_qkv, self.s3_5_project, 8)
        x = x + self.s3_5_mlp_fc2(self.gelu(self.s3_5_mlp_fc1(self.s3_5_norm2(x))))
        x = x + self._attention(self.s3_6_norm1(x), self.s3_6_qkv, self.s3_6_project, 8)
        x = x + self.s3_6_mlp_fc2(self.gelu(self.s3_6_mlp_fc1(self.s3_6_norm2(x))))
        x = x + self._attention(self.s3_7_norm1(x), self.s3_7_qkv, self.s3_7_project, 8)
        x = x + self.s3_7_mlp_fc2(self.gelu(self.s3_7_mlp_fc1(self.s3_7_norm2(x))))
        
        # reverse window partition
        x = x.transpose(1,2).reshape(x.shape[0],320,14,14)
        x = self.ds3(x)
        
        # window partition
        x = x.flatten(2).transpose(1, 2)
        
        # stage 4 hybrid MMAA
        # M
        x = x + self._mamba(self.s4_0_norm1(x), self.s4_0_in_project, self.s4_0_x_project,
                            self.s4_0_delta_project, self.s4_0_out_project,
                            self.s4_0_conv1d_x1, self.s4_0_conv1d_x2,
                            self.s4_0_A_log, self.s4_0_D)
        x = x + self.s4_0_mlp_fc2(self.gelu(self.s4_0_mlp_fc1(self.s4_0_norm2(x))))
        x = x + self._mamba(self.s4_1_norm1(x), self.s4_1_in_project, self.s4_1_x_project,
                            self.s4_1_delta_project, self.s4_1_out_project,
                            self.s4_1_conv1d_x1, self.s4_1_conv1d_x2,
                            self.s4_1_A_log, self.s4_1_D)
        x = x + self.s4_1_mlp_fc2(self.gelu(self.s4_1_mlp_fc1(self.s4_1_norm2(x))))
        
        # A
        x = x + self._attention(self.s4_2_norm1(x), self.s4_2_qkv, self.s4_2_project, 8)
        x = x + self.s4_2_mlp_fc2(self.gelu(self.s4_2_mlp_fc1(self.s4_2_norm2(x))))
        x = x + self._attention(self.s4_3_norm1(x), self.s4_3_qkv, self.s4_3_project, 8)
        x = x + self.s4_3_mlp_fc2(self.gelu(self.s4_3_mlp_fc1(self.s4_3_norm2(x))))
        
        # reverse window partition
        x = x.transpose(1,2).reshape(x.shape[0],640,7,7)
        
        # classification
        x = self.norm(x)
        x = torch.flatten(F.adaptive_avg_pool2d(x,1),1)
        return self.classifier(x)
    
def _map_name(name):
    raw_name = name.split(".")
    if name.startswith("patch_embed.conv_down."):
        m = {"0": "stem_conv1", "1": "stem_bn1", "3": "stem_conv2", "4": "stem_bn2"}[raw_name[2]]
        return m + "." + ".".join(raw_name[3:])
    if raw_name[0] == "norm":
        return name
    if raw_name[0] == "head":
        return "classifier." + ".".join(raw_name[1:])
    
    stage_number = int(raw_name[1]) + 1
    if raw_name[2] == "downsample":
        return f"ds{stage_number}." + ".".join(raw_name[5:])
        
    block = int(raw_name[3])
    mid = raw_name[4]
    end = ".".join(raw_name[5:])
    stage = f"s{stage_number}_{block}"
    if stage_number in (1,2):
        return f"{stage}_{mid}." + end
    if mid in ("norm1", "norm2"):
        return f"{stage}_{mid}." + end
    if mid =="mlp":
        return f"{stage}_mlp_{raw_name[5]}." + ".".join(raw_name[6:])
    
    mix = raw_name[5]
    rest = ".".join(raw_name[6:])
    rename = {
        "in_proj": "in_project",
        "x_proj": "x_project",
        "dt_proj": "delta_project",
        "out_proj": "out_project",
        "conv1d_x": "conv1d_x1",
        "conv1d_z": "conv1d_x2",
        "proj": "project"
    }
    mix = rename.get(mix, mix)
    mix_prefix = f"{stage}_{mix}"

    return mix_prefix + ("." + rest if rest else "")

def load_pretrained(model, weight_path, strict=True):
    weights = torch.load(weight_path, map_location="cpu")
    state_dict = weights["state_dict"]
    if next(iter(state_dict)).startswith("module."):
        state_dict = {key[7:]: value for key, value in state_dict.items()}
    new_dict = {_map_name(key): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(new_dict, strict=strict)
    print(f"missing={len(missing)} unexpected={len(unexpected)}\n")
    return model

def get_execution_order(model, input):
    order = []
    hooks = []

    def make_hook(name):
        def hook(module, input, output):
            order.append(name)
        return hook
    
    for name, module in model.named_modules():
        leaf = len(list(module.children()))
        trainable = any(p.requires_grad for p in module.parameters())
        if name and leaf == 0 and trainable:
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        model(input)
    
    for hook in hooks:
        hook.remove()
    
    return order

if __name__ == "__main__":
    weight_path = "mambavision_tiny_1k.pth"
    weights = torch.load(weight_path, map_location="cpu")
    image_path = "dinosaur.jpg"

    #['epoch', 'arch', 'state_dict', 'optimizer', 'version', 'args', 'amp_scaler', 'metric']
    # print(list(weights.keys()))
    # print(weights["state_dict"])
    
    mamba = MambaVision().eval()
    mamba_T = load_pretrained(mamba, weight_path)
    # print(mamba_T.meta['layers'])
    
    h, w = mamba_T.meta['image_size'][0], mamba_T.meta['image_size'][1]
    
    image = Image.open(image_path).convert('RGB')
    image = transforms.Resize((h, w))(image)
    
    tensor = transforms.ToTensor()(image)

    mean = torch.tensor(mamba_T.meta['mean'], dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(mamba_T.meta['std'], dtype=torch.float32).view(3, 1, 1)
    tensor = (tensor - mean) / std
    image = tensor.unsqueeze(0)

    logits = mamba_T(image)
    probs = F.softmax(logits, dim=1)

    top5_probs, top5_indices = torch.topk(probs, 5, dim=1)
    print("Top-5 predictions:")
    for i in range(5):
        print(f"Class {top5_indices[0, i].item():4d}"
              f" probability: {top5_probs[0, i].item():.6f}")
    
    with open("imagenet_classes.txt") as f:
        labels = [line.strip() for line in f]
    print(labels[probs.argmax(1).item()])

    # order = get_execution_order(mamba_T, image)
    # print(order)