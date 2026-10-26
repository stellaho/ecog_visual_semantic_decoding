from torch.utils.data import ConcatDataset


def split_recordings(recordings, test_id):
    ids = [name.removesuffix(".pt") for name in recordings]
    test_id = test_id.removesuffix(".pt")
    if len(ids) != len(set(ids)):
        raise ValueError("Recording IDs must be unique")
    if test_id not in ids or test_id == "Val01":
        raise ValueError(f"Invalid outer test recording: {test_id}")
    test_key = next(name for name in recordings if name.removesuffix(".pt") == test_id)
    train = {
        name: data
        for name, data in recordings.items()
        if name.removesuffix(".pt") not in {test_id, "Val01"}
    }
    if len(train) < 2:
        raise ValueError("Nested LORO requires at least two outer-training recordings")
    return train, recordings[test_key]


def get_loro_splits(recordings, loro_test_id):
    train_recordings, test_set = split_recordings(recordings, loro_test_id)
    folds = []
    for val_id, val_set in train_recordings.items():
        train_set = ConcatDataset(
            [data for name, data in train_recordings.items() if name != val_id]
        )
        folds.append((train_set, val_set))
    return test_set, folds
