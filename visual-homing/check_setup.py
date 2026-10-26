import pickle, sys, types
import torch, yaml
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
print("PyTorch", torch.__version__, "| GPU:", torch.cuda.get_device_name(0) if device == "cuda" else "none")

# --- Grounding DINO ---
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
dino = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(device).eval()
inputs = proc(images=Image.new("RGB", (640, 480), "gray"), text=[["a door"]], return_tensors="pt").to(device)
with torch.no_grad():
    dino(**inputs)
print("Grounding DINO OK")

# --- ViNT ---
sys.path.insert(0, "visualnav-transformer/train")
from vint_train.models.vint.vint import ViNT

cfg = yaml.safe_load(open("visualnav-transformer/train/config/vint.yaml"))
vint = ViNT(context_size=cfg["context_size"], len_traj_pred=cfg["len_traj_pred"],
            learn_angle=cfg["learn_angle"], obs_encoder=cfg["obs_encoder"],
            obs_encoding_size=cfg["obs_encoding_size"], late_fusion=cfg["late_fusion"],
            mha_num_attention_heads=cfg["mha_num_attention_heads"],
            mha_num_attention_layers=cfg["mha_num_attention_layers"],
            mha_ff_dim_factor=cfg["mha_ff_dim_factor"])

class _Placeholder:                       # stands in for saved training objects we don't need
    def __init__(self, *a, **k): pass
    def __setstate__(self, state): pass

class _Forgiving(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError):
            return _Placeholder

forgiving = types.ModuleType("forgiving_pickle")
forgiving.Unpickler = _Forgiving
forgiving.load = lambda f, **kw: _Forgiving(f, **kw).load()

ckpt = torch.load("weights/vint.pth", map_location="cpu", weights_only=False, pickle_module=forgiving)
saved = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
saved = getattr(saved, "module", saved)
result = vint.load_state_dict(saved.state_dict(), strict=False)
vint = vint.to(device).eval()

width, height = cfg["image_size"]
obs = torch.zeros(1, 3 * (cfg["context_size"] + 1), height, width, device=device)
goal = torch.zeros(1, 3, height, width, device=device)
with torch.no_grad():
    dist, waypoints = vint(obs, goal)
print(f"ViNT OK | missing weights: {len(result.missing_keys)} | waypoints shape: {tuple(waypoints.shape)}")

if device == "cuda":
    print(f"GPU memory used by both models: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
