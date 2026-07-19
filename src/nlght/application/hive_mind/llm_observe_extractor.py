# Copyright (c) 2026 Amphidrom GmbH. All rights reserved.
# See LICENSE in the repository root for license terms.

"""Backward-compat re-export — use entity_extractor.EntityExtractor directly."""

from nlght.application.hive_mind.entity_extractor import EntityExtractor as LLMObserveExtractor

__all__ = ["LLMObserveExtractor"]
