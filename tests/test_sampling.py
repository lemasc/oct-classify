from oct_classify.training.sampling import SourceClassBalancedBatchSampler


def test_source_class_balanced_sampler_gives_equal_source_quotas() -> None:
    sampler = SourceClassBalancedBatchSampler(
        ["duke", "duke", "paima", "paima"], [0, 1, 0, 1], 4, 2, seed=7
    )

    batches = list(sampler)

    assert all(sum(index < 2 for index in batch) == 2 for batch in batches)
    assert all(sum(index >= 2 for index in batch) == 2 for batch in batches)


def test_source_class_balanced_sampler_rejects_uneven_batch_size() -> None:
    try:
        SourceClassBalancedBatchSampler(["a", "b"], [0, 0], 3, 1, seed=7)
    except ValueError as error:
        assert "divide evenly" in str(error)
    else:
        raise AssertionError("Expected uneven source quota to be rejected")
