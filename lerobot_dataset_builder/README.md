# LeRobot OXE → LeRobot Dataset: Conversion & Stats Guide

This README walks you through installing **LeRobot**, configuring an OXE-to-LeRobot mapping, converting replayed data into the LeRobot format, and computing replay quality stats.

---

## 1) Installation

> You can follow the official LeRobot instructions, or use the quick-start below.

### Quick-start (Conda)

```bash
# Create & activate env
conda create -y -n lerobot2 python=3.10
conda activate lerobot2

# System dependency used by video I/O
conda install -y -c conda-forge ffmpeg

# Get LeRobot at a pinned commit
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 61b0eea
pip install -e .

# Runtime deps for this converter
pip install tensorflow-datasets
pip install gcsfs
python -m pip install -U tensorflow
```

> **Notes**
>
> * On macOS Apple Silicon, replace the last line with `python -m pip install -U tensorflow-macos`.
> * If you only want CPU TensorFlow on Linux/Windows: `python -m pip install -U tensorflow`.
> * If you need GPU on Linux: `python -m pip install -U "tensorflow[and-cuda]"`.

---

## 2) Configure your OXE dataset mapping

Edit `config.py` and add an entry to the `RLDS_TO_LEROBOT_DATASET_CONFIGS` dictionary for your OXE dataset.
This specifies:

* the **robot** type,
* **image\_size**,
* and how **RLDS fields** map to **LeRobot features** (dtype/shape).

Example:

```python
RLDS_TO_LEROBOT_DATASET_CONFIGS = {
    "toto": {
        "robot": "panda",
        "image_size": (480, 640),
        "rlds_to_lerobot_mappings": [
            {
                "rlds_path": "/steps/observation/image",
                "lerobot_path": "observation.images.image",
                "dtype": "video",
                "shape": (480, 640, 3),
            },
            {
                "rlds_path": "/steps/observation/natural_language_instruction",
                "lerobot_path": "natural_language_instruction",
                "dtype": "string",
                "shape": (1,),
            },
            {
                "rlds_path": "/steps/observation/state",
                "lerobot_path": "observation.state",
                "dtype": "float32",
                "shape": (7,),
            },
        ],
    },
    # ... add your datasets here ...
}
```

> Include the fields you need (images, natural language, states, etc.).
> Ensure `dtype`/`shape` match what your RLDS actually stores.

---

## 3) Convert replayed data → LeRobot dataset

### CLI

```bash
python lerobot_converter.py --dataset <dataset> \
  --split <train|val|test> --robots <replayed robots> \
  --start <episode #> --end <episode #> --fps <int> \
  --out_root <output root for LeRobot dataset> \
  --trg_root <root of your replay dataset>
```

**Parameter tips**

* `--dataset` : your OXE dataset key (e.g., `fractal20220817_data`), which must exist in `RLDS_TO_LEROBOT_DATASET_CONFIGS`.
* `--robots`  : one or more robot names to include (space-separated).
* `--trg_root`: the **parent** directory of your replay dataset.
  If the replay lives at `/path/to/oxe-aug/toto`, pass `--trg_root /path/to/oxe-aug`.

### Example

```bash
python lerobot_converter.py --dataset fractal20220817_data \
  --split train --robots panda sawyer ur5e jaco kinova3 kuka_iiwa xarm7 \
  --start 0 --end 10 --fps 30 \
  --out_root lerobot_dataset \
  --trg_root video
```

---

## 4) Replay quality stats (per-repo)

Compute error-rate and diagnostics for a single converted LeRobot repo:

```bash
python replay_stats.py \
  --root <where LeRobot dataset is stored> \
  --repo <dataset repo_id> \
  --threshold <float> \
  --metric <l2|l1|...>
```

### Example

```bash
python replay_stats.py \
  --root lerobot_dataset \
  --repo jaco_play_train_0_10 \
  --threshold 0.01 --metric l2
```

---

## 5) Batched stats (multiple chunked repos → one summary)

If your dataset was exported in multiple chunks, summarize them all:

```bash
python batched_replay_stats.py \
  --root <where LeRobot dataset is stored> \
  --prefix <dataset name prefix> \
  --threshold <float> --metric <l2|l1|...> \
  --csv <output CSV path> \
  --outdir <directory for figures>
```

### Example

```bash
python batched_replay_stats.py \
  --root lerobot_dataset \
  --prefix kaist_nonprehensile_converted_externally_to_rlds \
  --threshold 0.01 --metric l2 \
  --csv kaist_nonprehensile_converted_externally_to_rlds_stats.csv \
  --outdir dataset_stats
```

## Using the selective LeRobot dataloader (lazy video decode)

This loader lets you read **only the columns you need** from a local LeRobot repo, and **optionally** decode frames **just for specific video keys** at each sampled timestamp. It’s fast for state-only work and still convenient when you want a frame per step.

### What it returns

For each sampled frame index, you’ll get a dict containing:

* `episode_index` (int)
* `timestamp` (float seconds)
* Any **non-video** keys you requested (from parquet)
* Any **video** keys you requested, decoded on-the-fly as **`torch.float32` CHW tensors in \[0,1]**

> Video keys are detected from `ds.base.meta.video_keys`.

---

### 1) Discover keys (features & video keys)

```python
from selective_lerobot_dataloader import _load_info, discover_keys
from lerobot.datasets.lerobot_dataset import LeRobotDataset

root = "lerobot_dataset"
repo = "jaco_play_train_0_10"

# All feature names declared by the dataset (from meta/info.json)
info = _load_info(root, repo)
print("All feature keys:", discover_keys(info))

# Which keys are videos? Ask the dataset meta directly.
base = LeRobotDataset(repo_id=repo, root=f"{root}/{repo}", download_videos=False)
print("Video keys:", getattr(base.meta, "video_keys", []))
print("HF columns:", base.hf_dataset.column_names)
```

---

### 2) Non-video only (fast path, no decoding)

```python
from selective_lerobot_dataloader import make_selective_loader

loader = make_selective_loader(
    root="lerobot_dataset",
    repo_id="jaco_play_train_0_10",
    keys=["observation.state", "action.eef_velocity"],  # choose parquet keys
    batch_size=256,
    shuffle=True,
    num_workers=4,
    fmt="torch",               # numeric -> torch tensors
    download_videos=False,     # we won't decode video
)

for batch in loader:
    states = batch["observation.state"]        # torch.Tensor [B, D]
    eef_vel = batch["action.eef_velocity"]     # torch.Tensor [B, ?]
    epi = batch["episode_index"]               # torch.Tensor [B]
    ts  = batch["timestamp"]                   # torch.Tensor [B]
    # training / analytics ...
```

> If your selected **non-video** keys include strings (e.g., instructions), use `fmt="numpy"` so strings don’t get forced into torch.

---

### 3) Add **video** columns (lazy per-sample decode)

```python
from selective_lerobot_dataloader import make_selective_loader

# Suppose your dataset has the video key "observation.images.image"
loader = make_selective_loader(
    root="lerobot_dataset",
    repo_id="jaco_play_train_0_10",
    keys=["observation.state", "observation.images.image"],  # add a video key
    batch_size=64,
    shuffle=False,
    num_workers=4,
    fmt="torch",               # parquet numerics -> torch
    download_videos=False,     # videos already local; set True to auto-fetch from Hub
)

for batch in loader:
    img = batch["observation.images.image"]   # torch.FloatTensor [B, C, H, W] in [0,1]
    st  = batch["observation.state"]          # torch.FloatTensor [B, D]
    # model step ...
```

* Frames are pulled via `decode_video_frames(vpath, [timestamp], tolerance_s, backend)`.
* By default, the loader converts frames to **CHW float32 \[0,1]**.
* If you need **BGR** for OpenCV, open the file and **uncomment** the BGR line in `_frame_to_chw_float01` (already in the code as a hint).

---

### 4) Restrict by episode(s)

```python
# Iterate only episodes 0, 5, 7 (inclusive frame ranges)
loader = make_selective_loader(
    root="lerobot_dataset",
    repo_id="jaco_play_train_0_10",
    keys=["observation.state", "observation.images.image"],
    episodes=[0, 5, 7],
    batch_size=128,
    num_workers=4,
    fmt="torch",
)

for batch in loader:
    # batch["episode_index"] tells you which episode each frame came from
    ...
```

---

### 5) Mixed types (include strings)

```python
# Include text; use fmt="numpy" to avoid torch converting strings
loader = make_selective_loader(
    root="lerobot_dataset",
    repo_id="jaco_play_train_0_10",
    keys=["natural_language_instruction", "observation.state"],
    batch_size=128,
    fmt="numpy",
)

for batch in loader:
    instrs = batch["natural_language_instruction"]  # list[str] / np.array of str
    states = batch["observation.state"]             # np.ndarray [B, D]
```

---

### 6) Performance tips

* **num\_workers**: Increase to parallelize collation; start with 4–8 and tune.
* **persistent\_workers=True** (already set): avoids worker respawn overhead.
* **pin\_memory=True** (already set): useful if you move batches to GPU.
* Video decode is the main cost. Keep batch size moderate when including frames.
* Access to **local** videos is fastest. If your repo is remote, set `download_videos=True` during construction so LeRobot fetches them once.

### 7) Minimal training loop sketch

```python
import torch
from selective_lerobot_dataloader import make_selective_loader

loader = make_selective_loader(
    root="lerobot_dataset",
    repo_id="jaco_play_train_0_10",
    keys=["observation.images.image", "action.eef_velocity"],
    batch_size=32,
    shuffle=True,
    num_workers=4,
    fmt="torch",
)

for batch in loader:
    x = batch["observation.images.image"]   # [B, C, H, W], float32 [0,1]
    y = batch["action.eef_velocity"]        # [B, D]
    # x, y -> your model
```
