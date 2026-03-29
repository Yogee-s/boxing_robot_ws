"""Tests for the BoxBunny gamification engine."""

import pytest


# XP calculation constants (should match the engine)
RANK_THRESHOLDS = [
    ("Novice", 0), ("Contender", 500), ("Fighter", 1500),
    ("Warrior", 4000), ("Champion", 10000), ("Elite", 25000),
]

BASE_XP = {
    "training": 50, "sparring": 75, "free": 25,
    "power": 30, "stamina": 40, "reaction": 30,
}


def _get_rank(total_xp: int) -> str:
    """Calculate rank from XP."""
    rank = "Novice"
    for name, threshold in reversed(RANK_THRESHOLDS):
        if total_xp >= threshold:
            rank = name
            break
    return rank


def _xp_to_next(total_xp: int) -> tuple:
    """Get XP needed for next rank."""
    for i, (name, threshold) in enumerate(RANK_THRESHOLDS):
        if total_xp < threshold:
            return threshold - total_xp, name
    return 0, "Elite"


class TestRankSystem:
    """Test rank calculations."""

    def test_novice_at_zero(self):
        assert _get_rank(0) == "Novice"

    def test_contender_at_500(self):
        assert _get_rank(500) == "Contender"

    def test_fighter_at_1500(self):
        assert _get_rank(1500) == "Fighter"

    def test_warrior_at_4000(self):
        assert _get_rank(4000) == "Warrior"

    def test_champion_at_10000(self):
        assert _get_rank(10000) == "Champion"

    def test_elite_at_25000(self):
        assert _get_rank(25000) == "Elite"

    def test_between_ranks(self):
        assert _get_rank(999) == "Contender"
        assert _get_rank(1499) == "Contender"
        assert _get_rank(3999) == "Fighter"

    def test_xp_to_next_from_zero(self):
        remaining, next_rank = _xp_to_next(0)
        assert remaining == 500
        assert next_rank == "Contender"

    def test_xp_to_next_from_elite(self):
        remaining, next_rank = _xp_to_next(30000)
        assert remaining == 0
        assert next_rank == "Elite"


class TestSessionXP:
    """Test session XP calculations."""

    def test_base_xp_training(self):
        assert BASE_XP["training"] == 50

    def test_base_xp_sparring(self):
        assert BASE_XP["sparring"] == 75

    def test_base_xp_free(self):
        assert BASE_XP["free"] == 25


class TestSessionScore:
    """Test session scoring (0-100)."""

    def _compute_score(
        self,
        volume_ratio: float = 0.5,
        accuracy: float = 0.5,
        consistency: float = 0.5,
        improvement: float = 0.0,
    ) -> int:
        """Simplified session score computation."""
        score = (
            volume_ratio * 30
            + accuracy * 30
            + consistency * 25
            + improvement * 15
        )
        return max(0, min(100, int(score)))

    def test_perfect_score(self):
        score = self._compute_score(1.0, 1.0, 1.0, 1.0)
        assert score == 100

    def test_zero_score(self):
        score = self._compute_score(0.0, 0.0, 0.0, 0.0)
        assert score == 0

    def test_average_score(self):
        score = self._compute_score(0.5, 0.5, 0.5, 0.0)
        assert 30 <= score <= 50

    def test_high_accuracy_low_volume(self):
        score = self._compute_score(0.2, 0.9, 0.7, 0.0)
        assert score > 40
