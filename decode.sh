uv run -m cs336_basics.train.decode \
    --checkpoint_dir checkpoints/tinystories_lr_3e-2 \
    --tokenizer_path data/tiny_stories/tokenizer.pkl \
    --max_length 256 \
    --temperature 0.7 \
    --topp 0.9 \
    --eos_token '<|endoftext|>' \
    --prompt 'Hello, how are you doing today' 