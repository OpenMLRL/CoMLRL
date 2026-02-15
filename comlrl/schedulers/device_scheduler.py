from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple, Union

import torch

DeviceSpec = Union[str, Sequence[str]]


class DeviceScheduler:
    @staticmethod
    def assign_devices(
        num_agents: int,
        agent_devices: Optional[DeviceSpec],
        critic_devices: Optional[DeviceSpec],
        *,
        use_separate_critic: bool,
    ) -> Tuple[List[torch.device], List[torch.device]]:
        agent_list = DeviceScheduler.resolve_devices(
            agent_devices, num_agents, kind="agent_devices"
        )
        if use_separate_critic:
            critic_spec = (
                critic_devices if critic_devices is not None else agent_devices
            )
            critic_list = DeviceScheduler.resolve_devices(
                critic_spec, num_agents, kind="critic_devices"
            )
        else:
            critic_list = list(agent_list)
        return agent_list, critic_list

    @staticmethod
    def assign_shared_critic_device(
        agent_devices: Sequence[torch.device],
        critic_devices: Optional[DeviceSpec],
    ) -> torch.device:
        if critic_devices is None:
            return agent_devices[0]
        return DeviceScheduler.resolve_devices(
            critic_devices, 1, kind="critic_devices"
        )[0]

    @staticmethod
    def resolve_devices(
        spec: Optional[DeviceSpec],
        num_devices: int,
        *,
        kind: str = "devices",
    ) -> List[torch.device]:
        if num_devices < 1:
            raise ValueError(f"{kind} count must be >= 1.")

        if spec is None or (isinstance(spec, str) and spec.lower() == "auto"):
            return DeviceScheduler._auto_devices(num_devices)

        if isinstance(spec, str):
            return [torch.device(spec)] * num_devices

        if isinstance(spec, Sequence):
            if len(spec) == 0:
                raise ValueError(f"{kind} must be a non-empty list or 'auto'.")
            if len(spec) == 1:
                return [torch.device(spec[0])] * num_devices
            if len(spec) != num_devices:
                raise ValueError(
                    f"{kind} length ({len(spec)}) must be 1 or {num_devices}."
                )
            return [torch.device(s) for s in spec]

        raise ValueError(f"Unsupported {kind} spec: {spec!r}.")

    @staticmethod
    def devices_disjoint(device_groups: Iterable[Sequence[torch.device]]) -> bool:
        seen = set()
        for group in device_groups:
            for device in group:
                key = DeviceScheduler._device_key(device)
                if key in seen:
                    return False
                seen.add(key)
        return True

    @staticmethod
    def _device_key(device: torch.device) -> str:
        if device.type == "cuda":
            return f"cuda:{device.index}"
        return device.type

    @staticmethod
    def _auto_devices(num_devices: int) -> List[torch.device]:
        if not torch.cuda.is_available():
            return [torch.device("cpu")] * num_devices

        count = int(torch.cuda.device_count())
        if count < 1:
            return [torch.device("cpu")] * num_devices
        if count < num_devices:
            return [torch.device("cuda:0")] * num_devices

        indices = list(range(count))
        return [torch.device(f"cuda:{idx}") for idx in indices[:num_devices]]
