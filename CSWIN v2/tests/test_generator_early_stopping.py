import pytest

from hsi_model.train_generator import _update_early_stopping


def test_early_stopping_tracks_improvements_and_patience():
    best, bad_epochs, should_stop = _update_early_stopping(
        current_mrae=0.30,
        best_mrae=float("inf"),
        bad_epochs=0,
        patience=2,
        min_delta=1e-4,
        epoch=10,
        warmup_epochs=5,
    )
    assert best == pytest.approx(0.30)
    assert bad_epochs == 0
    assert not should_stop

    best, bad_epochs, should_stop = _update_early_stopping(
        current_mrae=0.30005,
        best_mrae=best,
        bad_epochs=bad_epochs,
        patience=2,
        min_delta=1e-4,
        epoch=11,
        warmup_epochs=5,
    )
    assert bad_epochs == 1
    assert not should_stop

    _, bad_epochs, should_stop = _update_early_stopping(
        current_mrae=0.31,
        best_mrae=best,
        bad_epochs=bad_epochs,
        patience=2,
        min_delta=1e-4,
        epoch=12,
        warmup_epochs=5,
    )
    assert bad_epochs == 2
    assert should_stop


def test_early_stopping_does_not_count_warmup_epochs():
    best, bad_epochs, should_stop = _update_early_stopping(
        current_mrae=0.31,
        best_mrae=0.30,
        bad_epochs=0,
        patience=1,
        min_delta=1e-4,
        epoch=3,
        warmup_epochs=5,
    )

    assert best == pytest.approx(0.30)
    assert bad_epochs == 0
    assert not should_stop


def test_early_stopping_can_be_disabled_with_zero_patience():
    _, bad_epochs, should_stop = _update_early_stopping(
        current_mrae=0.31,
        best_mrae=0.30,
        bad_epochs=0,
        patience=0,
        min_delta=1e-4,
        epoch=10,
        warmup_epochs=0,
    )

    assert bad_epochs == 1
    assert not should_stop
