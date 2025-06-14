import gymnasium as gym
from termcolor import colored
from ocatari.ram.extract_ram_info import (
    detect_objects_ram,
    init_objects,
)
from ocatari.vision.extract_vision_info import detect_objects_vision

try:
    # ALE (Arcade Learning Environment) is required for running Atari environments.
    import ale_py
except ModuleNotFoundError:
    raise ModuleNotFoundError(
        '\nALE is required when using the ALE env wrapper. Try `pip install "gymnasium[atari, accept-rom-license]"`\n'
    )


class OCAtari(gym.Wrapper):

    def __init__(self, env: gym.Env, mode="ram", hud=False):
        super().__init__(env)
        self.game_name = (
            env.spec.id.split("-")[0].split("No")[0].split("Deterministic")[0]  # type: ignore
        )
        self.mode = mode
        self.hud = hud

        global init_objects
        if mode == "vision":
            self.detect_objects = self._detect_objects_vision
            self.objects = init_objects(self.game_name, self.hud, vision=True)
        elif mode == "ram":
            self.detect_objects = self._detect_objects_ram
            self.objects = init_objects(self.game_name, self.hud)
        elif mode == "both":
            self.detect_objects = self._detect_objects_both
            self.objects = init_objects(self.game_name, self.hud)
        else:
            raise ValueError("Undefined mode for information extraction")

    def step(self, *args, **kwargs):
        obs, reward, terminated, truncated, info = self.env.step(*args, **kwargs)
        self.detect_objects()
        return obs, reward, terminated, truncated, info

    def _detect_objects_ram(self):
        detect_objects_ram(
            self.objects,
            self.env.unwrapped.ale.getRAM(),  # type: ignore
            self.game_name,
            self.hud,
        )

    def _detect_objects_vision(self):
        detect_objects_vision(
            self.objects,
            self.env.unwrapped.ale.getScreenRGB(),  # type: ignore
            self.game_name,
            self.hud,
        )

    def _detect_objects_both(self):
        self._detect_objects_ram()
        self._detect_objects_vision()

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self.objects = init_objects(
            self.game_name, self.hud, vision=self.mode == "vision"
        )
        self.detect_objects()
        return obs, info

    def close(self, *args, **kwargs):
        super().close(*args, **kwargs)
