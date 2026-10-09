from dataclasses import dataclass
from random import Random

WIDTH, HEIGHT = 10, 20
INPUTS = 33

PIECES = (
    ((-1, 0), (-2, 0), (0, 0), (1, 0)),
    ((0, -1), (-1, -1), (-1, 0), (0, 0)),
    ((-1, 0), (-1, 1), (0, 0), (0, -1)),
    ((0, 0), (-1, 0), (0, 1), (-1, -1)),
    ((0, 0), (0, -1), (0, 1), (-1, -1)),
    ((0, 0), (0, -1), (0, 1), (1, -1)),
    ((0, 0), (0, -1), (0, 1), (-1, 0)),
)

Board = list[list[int]]


@dataclass(frozen=True)
class Placement:
    cells: tuple[tuple[int, int], ...]
    board: Board
    lines: int
    landing_y: int
    value: float


def empty_board() -> Board:
    return [[0 for _ in range(WIDTH)] for _ in range(HEIGHT)]


def _normalize(cells: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_x = min(x for x, _ in cells)
    min_y = min(y for _, y in cells)
    return tuple(sorted((x - min_x, y - min_y) for x, y in cells))


def orientations(piece_index: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    cells = _normalize(PIECES[piece_index])
    result = []
    for _ in range(4):
        if cells not in result:
            result.append(cells)
        cells = _normalize(tuple((-y, x) for x, y in cells))
    return tuple(result)


def _fits(board: Board, cells: tuple[tuple[int, int], ...], x: int, y: int) -> bool:
    return all(
        0 <= x + cell_x < WIDTH
        and 0 <= y + cell_y < HEIGHT
        and not board[y + cell_y][x + cell_x]
        for cell_x, cell_y in cells
    )


def _board_features(board: Board) -> tuple[list[int], list[int], int, int, int, int]:
    heights = []
    holes_by_column = []
    for x in range(WIDTH):
        occupied = [y for y in range(HEIGHT) if board[y][x]]
        heights.append(HEIGHT - occupied[0] if occupied else 0)
        holes_by_column.append(
            sum(not board[y][x] for y in range(occupied[0], HEIGHT)) if occupied else 0
        )
    total_holes = sum(holes_by_column)
    bumpiness = sum(abs(heights[x] - heights[x + 1]) for x in range(WIDTH - 1))
    return heights, holes_by_column, total_holes, bumpiness, max(heights), sum(heights)


def _features(
    board: Board, piece_index: int, lines: int, landing_y: int
) -> tuple[float, ...]:
    heights, holes, total_holes, bumpiness, max_height, aggregate_height = _board_features(board)
    piece_identity = tuple(float(index == piece_index) for index in range(len(PIECES)))
    values = (
        *(height / HEIGHT for height in heights),
        *(count / HEIGHT for count in holes),
        aggregate_height / (WIDTH * HEIGHT),
        bumpiness / (WIDTH * HEIGHT),
        max_height / HEIGHT,
        total_holes / (WIDTH * HEIGHT),
        lines / 4,
        landing_y / HEIGHT,
        *piece_identity,
    )
    return values


def placements(board: Board, piece_index: int, network) -> list[Placement]:
    options = []
    for shape in orientations(piece_index):
        shape_width = max(x for x, _ in shape) + 1
        for x in range(WIDTH - shape_width + 1):
            y = 0
            if not _fits(board, shape, x, y):
                continue
            while _fits(board, shape, x, y + 1):
                y += 1

            placed = [row[:] for row in board]
            cells = tuple((x + cell_x, y + cell_y) for cell_x, cell_y in shape)
            for cell_x, cell_y in cells:
                placed[cell_y][cell_x] = 1
            full_rows = sum(all(row) for row in placed)
            remaining = [row for row in placed if not all(row)]
            result = [[0 for _ in range(WIDTH)] for _ in range(full_rows)] + remaining
            value = network.activate(_features(result, piece_index, full_rows, y))[0]
            options.append(Placement(cells, result, full_rows, y, value))
    return options


def play_episode(network, seed: int, max_pieces: int = 250) -> tuple[float, int, int]:
    board = empty_board()
    rng = Random(seed)
    total_lines = 0
    placed_count = 0

    for _ in range(max_pieces):
        piece_index = rng.randrange(len(PIECES))
        options = placements(board, piece_index, network)
        if not options:
            break
        selected = max(options, key=lambda placement: placement.value)
        board = selected.board
        total_lines += selected.lines
        placed_count += 1

    _, _, holes, _, max_height, _ = _board_features(board)
    fitness = total_lines * 100.0 + placed_count * 0.5 - holes * 2.0 - max_height * 0.25
    return fitness, total_lines, placed_count