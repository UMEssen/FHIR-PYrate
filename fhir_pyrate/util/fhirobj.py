import json
from types import SimpleNamespace
from typing import Any


class FHIRObj(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if isinstance(val, dict):
                setattr(self, key, FHIRObj(**val))
            elif isinstance(val, list):
                setattr(
                    self, key, [FHIRObj(**v) if isinstance(v, dict) else v for v in val]
                )

    def __getattr__(self, item: str) -> None:
        return None

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    def to_json(self) -> str:
        return json.dumps(self, default=lambda o: o.__dict__)

    def to_dict(self) -> Any:
        return json.loads(self.to_json())
