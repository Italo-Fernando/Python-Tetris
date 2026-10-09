import argparse
import pickle
from pathlib import Path
from random import Random, randrange

import neat
import pygame

from tetris_ai import HEIGHT, PIECES, WIDTH, empty_board, placements
from train_ai import CONFIG_PATH, train


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "best_genome.pkl"
TILE = 45
GAME_SIZE = WIDTH * TILE, HEIGHT * TILE
WINDOW_SIZE = 750, 940
FPS = 60


def load_network(model_path):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    with model_path.open("rb") as genome_file:
        genome = pickle.load(genome_file)
    return neat.nn.FeedForwardNetwork.create(genome, config)


def get_color():
    return randrange(30, 256), randrange(30, 256), randrange(30, 256)


def main():
    parser = argparse.ArgumentParser(description="Watch the NEAT Tetris player.")
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population", type=int, default=40)
    args = parser.parse_args()

    if not args.model.exists():
        print("No trained genome found. Training before opening the game...")
        train(args.generations, args.population, 2, 180, args.model)

    network = load_network(args.model)
    pygame.init()
    screen = pygame.display.set_mode(WINDOW_SIZE)
    pygame.display.set_caption("NEAT Tetris AI")
    game_surface = pygame.Surface(GAME_SIZE)
    clock = pygame.time.Clock()
    background = pygame.image.load(str(ROOT / "img/bg.jpg")).convert()
    game_background = pygame.image.load(str(ROOT / "img/bg2.jpg")).convert()
    title_font = pygame.font.Font(str(ROOT / "font/font.ttf"), 58)
    font = pygame.font.Font(str(ROOT / "font/font.ttf"), 30)
    small_font = pygame.font.Font(str(ROOT / "font/font.ttf"), 22)
    title = title_font.render("TETRIS", True, pygame.Color("darkorange"))
    score_title = font.render("score", True, pygame.Color("green"))
    lines_title = font.render("lines", True, pygame.Color("green"))

    rng = Random()
    board = empty_board()
    piece_index = rng.randrange(len(PIECES))
    next_piece_index = rng.randrange(len(PIECES))
    color = get_color()
    next_color = get_color()
    score = 0
    total_lines = 0
    best_score = 0
    paused = False
    game_over_at = None
    animation_frame = 0
    fall_speed = 1.0

    def choose_move():
        options = placements(board, piece_index, network)
        return max(options, key=lambda option: option.value) if options else None

    placement = choose_move()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_UP:
                    fall_speed = min(3.0, fall_speed + 0.25)
                elif event.key == pygame.K_DOWN:
                    fall_speed = max(0.5, fall_speed - 0.25)

        if not paused and game_over_at is None:
            if placement is None:
                game_over_at = pygame.time.get_ticks()
                best_score = max(best_score, score)
            else:
                animation_frame += 1
                target_y = min(y for _, y in placement.cells)
                drop_distance = target_y + 3
                duration = max(12, int((target_y + HEIGHT) * 3 / fall_speed))
                if animation_frame >= duration:
                    board = placement.board
                    total_lines += placement.lines
                    score += {0: 0, 1: 100, 2: 300, 3: 700, 4: 1500}[placement.lines]
                    piece_index = next_piece_index
                    next_piece_index = rng.randrange(len(PIECES))
                    color = next_color
                    next_color = get_color()
                    placement = choose_move()
                    animation_frame = 0

        if game_over_at is not None and pygame.time.get_ticks() - game_over_at > 1600:
            board = empty_board()
            score = 0
            total_lines = 0
            piece_index = rng.randrange(len(PIECES))
            next_piece_index = rng.randrange(len(PIECES))
            color = get_color()
            next_color = get_color()
            placement = choose_move()
            animation_frame = 0
            game_over_at = None

        screen.blit(background, (0, 0))
        game_surface.blit(game_background, (0, 0))
        for x in range(WIDTH):
            for y in range(HEIGHT):
                if board[y][x]:
                    pygame.draw.rect(game_surface, (230, 170, 55), (x * TILE, y * TILE, TILE - 2, TILE - 2))
                pygame.draw.rect(game_surface, (40, 40, 40), (x * TILE, y * TILE, TILE, TILE), 1)

        if placement is not None and game_over_at is None:
            target_y = min(y for _, y in placement.cells)
            drop_distance = target_y + 3
            duration = max(12, int((target_y + HEIGHT) * 3 / fall_speed))
            remaining = max(0.0, 1.0 - animation_frame / duration)
            vertical_offset = int(drop_distance * remaining)
            for x, y in placement.cells:
                draw_y = y - vertical_offset
                if draw_y >= 0:
                    pygame.draw.rect(
                        game_surface,
                        color,
                        (x * TILE, draw_y * TILE, TILE - 2, TILE - 2),
                    )

        screen.blit(game_surface, (20, 20))
        screen.blit(title, (490, -5))
        screen.blit(score_title, (535, 645))
        screen.blit(font.render(str(score), True, pygame.Color("white")), (540, 685))
        screen.blit(lines_title, (535, 745))
        screen.blit(font.render(str(total_lines), True, pygame.Color("white")), (540, 785))
        screen.blit(small_font.render("next", True, pygame.Color("white")), (555, 155))
        for x, y in PIECES[next_piece_index]:
            pygame.draw.rect(
                screen,
                next_color,
                ((x + 6) * (TILE // 2) + 480, (y + 3) * (TILE // 2) + 180, TILE // 2 - 2, TILE // 2 - 2),
            )
        screen.blit(small_font.render(f"best {best_score}", True, pygame.Color("gold")), (520, 860))
        screen.blit(small_font.render(f"speed {fall_speed:.2f}x", True, pygame.Color("white")), (520, 890))

        if paused:
            screen.blit(font.render("PAUSED", True, pygame.Color("white")), (280, 450))
        elif game_over_at is not None:
            screen.blit(font.render("GAME OVER", True, pygame.Color("white")), (270, 450))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()


if __name__ == "__main__":
    main()
