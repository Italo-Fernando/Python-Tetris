# Python-Tetris
Tetris using Python and Pygame

![tetris](screenshot/1.png "Tetris")

## NEAT AI (enhancement)

Use the project virtual environment to train and watch the AI:

```bash
.venv/bin/python train_ai.py --generations 25 --population 40
.venv/bin/python play_ai.py
```

If no trained genome exists, `play_ai.py` trains one before opening the game. Training runs without a graphical window and saves checkpoints in `checkpoints/`. Resume from a checkpoint with:

```bash
.venv/bin/python train_ai.py --resume checkpoints/neat-checkpoint-5
```

Press `P` to pause or resume. Use the up and down arrows to change the fall speed.
