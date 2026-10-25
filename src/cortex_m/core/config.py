"""
CORTEX-M Typed Configuration System.
Implements Layer 1 configuration with deployment profiles, validation, and precedence.
"""

import json
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from .errors import ConfigurationError


class DeploymentProfile(str, Enum):
    LOCAL_RESEARCH = "local_research"
    LOCAL_ASSISTANT = "local_assistant"
    DEVELOPER_WORKSTATION = "developer_workstation"
    TEAM_SERVER = "team_server"
    ENTERPRISE_PRIVATE = "enterprise_private"
    EDGE = "edge"


@dataclass
class ToolsConfig:
    python_sandbox: bool = True
    web_search: bool = True
    calculator: bool = True
    calendar: bool = True
    git_patching: bool = True
    sandbox_timeout_seconds: float = 5.0


@dataclass
class MemoryConfig:
    beta: float = 8.0
    learning_rate: float = 1.0
    synaptic_decay: float = 0.01
    hallucination_threshold: float = 0.50
    working_memory_depth: int = 3
    state_dimension: int = 64
    zero_database_mode: bool = True


@dataclass
class ModelConfig:
    engine_mode: str = "cortex_native"  # cortex_native, slm_ollama, openai_compatible
    slm_endpoint: str = "http://localhost:11434/api/generate"
    slm_model: str = "llama3.2"
    openai_endpoint: str = "https://api.openai.com/v1/chat/completions"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    stream_responses: bool = True


@dataclass
class CortexConfig:
    profile: DeploymentProfile = DeploymentProfile.DEVELOPER_WORKSTATION
    active_tenant_id: str = "default_user"
    models: ModelConfig = field(default_factory=ModelConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    require_human_approval_for_patches: bool = True
    enable_epistemic_abstention: bool = True

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "CortexConfig":
        """Load configuration from file or environment with fallback to defaults."""
        path = config_path or cls._resolve_default_path()
        if path and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                return cls._from_dict(data)
            except Exception as e:
                raise ConfigurationError(f"Failed to load config from {path}: {e}")
        return cls()

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> "CortexConfig":
        models_data = data.get("models", {})
        memory_data = data.get("memory", {})
        tools_data = data.get("tools", {})

        profile_val = data.get("profile", DeploymentProfile.DEVELOPER_WORKSTATION.value)
        try:
            profile = DeploymentProfile(profile_val)
        except ValueError:
            profile = DeploymentProfile.DEVELOPER_WORKSTATION

        return cls(
            profile=profile,
            active_tenant_id=data.get("active_tenant_id", "default_user"),
            models=ModelConfig(**{k: v for k, v in models_data.items() if k in ModelConfig.__dataclass_fields__}),
            memory=MemoryConfig(**{k: v for k, v in memory_data.items() if k in MemoryConfig.__dataclass_fields__}),
            tools=ToolsConfig(**{k: v for k, v in tools_data.items() if k in ToolsConfig.__dataclass_fields__}),
            require_human_approval_for_patches=data.get("require_human_approval_for_patches", True),
            enable_epistemic_abstention=data.get("enable_epistemic_abstention", True),
        )

    def save(self, path: Optional[Path] = None) -> None:
        """Persist configuration to disk."""
        target_path = path or self._resolve_default_path()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile.value,
            "active_tenant_id": self.active_tenant_id,
            "models": asdict(self.models),
            "memory": asdict(self.memory),
            "tools": asdict(self.tools),
            "require_human_approval_for_patches": self.require_human_approval_for_patches,
            "enable_epistemic_abstention": self.enable_epistemic_abstention,
        }

    @staticmethod
    def _resolve_default_path() -> Path:
        if os.environ.get("CORTEX_CONFIG_PATH"):
            return Path(os.environ["CORTEX_CONFIG_PATH"])
        cwd_cfg = Path("configs/default.json")
        if cwd_cfg.exists():
            return cwd_cfg.resolve()
        return Path(__file__).resolve().parent.parent.parent.parent / "configs" / "default.json"
