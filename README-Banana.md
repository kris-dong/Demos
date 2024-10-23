# Demo: I Want Banana

## Setup

Install Python dependencies

```bash
pip install -r requirements.txt
```

Download pretrained perception model checkpoints

```bash
./scripts/banana_download_checkpoints.sh
```

Clone and install Depth Anything V2 Python package

```bash
mkdir -p ./shared/
git clone git@github.com:T-K-233/Depth-Anything-V2.git ./shared/Depth-Anything-V2/
pip install -e ./shared/Depth-Anything-V2/
```

## Run

```bash
python ./scripts/tests/banana_perception.py
```

