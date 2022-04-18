from typing import Tuple
import cv2
import numpy as np
import tkinter as tk
from environment_agent_modules import (
    create_nonclustered_tile_grid,
    SwarmAgent,
    Direction,
)

TILE_SIZE = 25
MID_POINT_OFFSET = round(TILE_SIZE / 2)
root = tk.Tk()
SCREEN_MID_POINT = (
    round(root.winfo_screenwidth() / 4),
    round(root.winfo_screenheight() / 2),
)


def setup_tile_grid_for_display(tile_grid: np.ndarray) -> np.ndarray:
    display_grid = np.zeros(
        (
            tile_grid.shape[0] * TILE_SIZE,
            tile_grid.shape[1] * TILE_SIZE,
            3,
        ),
        np.uint8,
    )

    for row in range(tile_grid.shape[0]):
        for col in range(tile_grid.shape[1]):
            if not (tile_grid[row, col]["colour"]):
                display_grid[
                    row * TILE_SIZE : row * TILE_SIZE + TILE_SIZE,
                    col * TILE_SIZE : col * TILE_SIZE + TILE_SIZE,
                ] = (
                    255,
                    255,
                    255,
                )

    return display_grid


def map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
    agent_direction: int, agent_position: Tuple[int, int]
) -> Tuple[int, int]:
    return {
        Direction.UP.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row),
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
        ),
        Direction.RIGHT.value: lambda row, column: (
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
            (column, row - MID_POINT_OFFSET),
        ),
        Direction.DOWN.value: lambda row, column: (
            (column - MID_POINT_OFFSET, row - TILE_SIZE),
            (column - MID_POINT_OFFSET, row),
        ),
        Direction.LEFT.value: lambda row, column: (
            (column, row - MID_POINT_OFFSET),
            (column - TILE_SIZE, row - MID_POINT_OFFSET),
        ),
    }[agent_direction](*agent_position)


def draw_swarm_agent_on_tile_grid(
    display_tile_grid: np.ndarray,
    swarm_agent: SwarmAgent,
) -> np.ndarray:
    (
        start_point,
        end_point,
    ) = map_swarm_agent_direction_position_to_display_tile_grid_arrow_coordinates(
        agent_direction=swarm_agent.current_direction_facing,
        agent_position=[(value + 1) * TILE_SIZE for value in swarm_agent.current_cell],
    )

    return cv2.arrowedLine(
        display_tile_grid,
        start_point,
        end_point,
        (0, 0, 255),
        1,
        tipLength=0.4,
    )


def move_and_show_window(x: int, y: int, winname: str, img: np.ndarray) -> None:
    cv2.namedWindow(winname)
    cv2.moveWindow(winname, x, y)
    cv2.imshow(winname, img)
    cv2.waitKey(400)


def main() -> None:
    tile_grid = create_nonclustered_tile_grid(20, 20)
    swarm_agents = [
        SwarmAgent(id=i, starting_cell=tile_grid[i, i], needs_models_loaded=True)
        for i in range(20)
    ]

    display_location = (
        round(SCREEN_MID_POINT[0] - (tile_grid.shape[1] * TILE_SIZE / 4)),
        round(SCREEN_MID_POINT[1] - (tile_grid.shape[0] * TILE_SIZE / 8)),
    )

    for i in range(100):
        img = setup_tile_grid_for_display(tile_grid)

        for agent in swarm_agents:
            agent.navigate(tile_grid=tile_grid)
            draw_swarm_agent_on_tile_grid(img, agent)

        move_and_show_window(*display_location, winname="Tile Grid", img=img)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
