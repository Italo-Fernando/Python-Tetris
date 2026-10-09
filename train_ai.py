import argparse
import pickle
from pathlib import Path
from statistics import mean

import neat

from tetris_ai import play_episode


ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "neat_config.ini"


def evaluate_genomes(genomes, config, seeds, max_pieces):
    for _, genome in genomes:
        network = neat.nn.FeedForwardNetwork.create(genome, config)
        results = [play_episode(network, seed, max_pieces) for seed in seeds]
        genome.fitness = mean(result[0] for result in results)


def train(generations, population_size, episodes, max_pieces, output_path, resume=None):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    config.pop_size = population_size
    population = (
        neat.Checkpointer.restore_checkpoint(str(resume))
        if resume
        else neat.Population(config)
    )
    population.add_reporter(neat.StdOutReporter(True))
    population.add_reporter(neat.StatisticsReporter())
    checkpoint_dir = ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    population.add_reporter(
        neat.Checkpointer(generation_interval=5, filename_prefix=str(checkpoint_dir / "neat-checkpoint-"))
    )
    seeds = [101, 307, 911][:episodes]

    def evaluate(current_genomes, current_config):
        evaluate_genomes(current_genomes, current_config, seeds, max_pieces)

    winner = population.run(evaluate, generations)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as genome_file:
        pickle.dump(winner, genome_file)

    winner_network = neat.nn.FeedForwardNetwork.create(winner, config)
    results = [play_episode(winner_network, seed, max_pieces) for seed in seeds]
    print(
        "Best genome evaluation: "
        f"{mean(result[1] for result in results):.1f} lines, "
        f"{mean(result[2] for result in results):.1f} pieces per episode"
    )
    print(f"Saved best genome to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train a NEAT network to play Tetris.")
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=40)
    parser.add_argument("--episodes", type=int, choices=range(1, 4), default=2)
    parser.add_argument("--max-pieces", type=int, default=180)
    parser.add_argument("--output", type=Path, default=ROOT / "best_genome.pkl")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    train(
        args.generations,
        args.population,
        args.episodes,
        args.max_pieces,
        args.output,
        args.resume,
    )


if __name__ == "__main__":
    main()