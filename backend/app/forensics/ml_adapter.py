"""Adaptador opcional de modelos de localización de manipulaciones (TruFor / CAT-Net / DocTamper).

Desactivado por defecto (ENABLE_ML_LOCALIZATION=false). No descarga pesos. Para integrar un modelo
real, implemente `LocalizationModel.predict` y registre la clase en `load_model` (los pesos se
proveen vía una ruta en la variable ML_MODEL_PATH, fuera de la imagen Docker por defecto).
"""
import os
from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from . import settings


class LocalizationModel(ABC):
    name = "abstract"

    @abstractmethod
    def predict(self, image: np.ndarray) -> np.ndarray:
        """Devuelve un mapa de calor float32 en [0, 1] con el mismo alto/ancho que `image`."""


class StubModel(LocalizationModel):
    """Implementación nula: devuelve un mapa de ceros. Sirve para probar la integración."""
    name = "stub"

    def predict(self, image: np.ndarray) -> np.ndarray:
        return np.zeros(image.shape[:2], np.float32)


class TruForModel(LocalizationModel):  # pragma: no cover - ejemplo de integración
    """Ejemplo: requiere `torch` y el repo TruFor (https://github.com/grip-unina/TruFor) instalados
    y los pesos en ML_MODEL_PATH. Se importa de forma diferida para no exigir dependencias pesadas."""
    name = "trufor"

    def __init__(self, weights: str):
        import torch  # noqa: F401
        raise NotImplementedError("Complete la carga de pesos de TruFor según su repositorio oficial.")

    def predict(self, image: np.ndarray) -> np.ndarray:
        raise NotImplementedError


def load_model() -> Optional[LocalizationModel]:
    if not settings.ml_enabled():
        return None
    kind = os.getenv("ML_MODEL", "stub").lower()
    if kind == "trufor":
        return TruForModel(os.getenv("ML_MODEL_PATH", ""))
    return StubModel()
