# openai-glide-lib
Organizing the notebooks from the official glide-text2im project into reusable python libraries

## Setup

Follow instructions in the [glide-text2im](https://github.com/openai/glide-text2im) to install the base Glide library.

### Additional dependencies

```bash
pip install pyyaml
```

## Usage

### text2im

```bash
python text2im.py --prompt="An oil painting of a corgi" --output_file="corgi.png"
```

## clip-guided text2im

```bash
python clip_guided.py --prompt="An oil painting of a corgi" --output_file="corgi.png"
```

## inpaint

```bash
python inpaint.py --source_file="src.png" --source_mask_file="mask.png" --output_file="corgi.png" --prompt="A corgi in a field"
```
