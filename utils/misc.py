from datasets import load_dataset


def train_test_val_split(dataset, test_size=0.2, val_size=0.1, stratify_by=None):
    """
  Splits a dataset loaded with `load_dataset` into training, validation, and test sets.

  Args:
      dataset: The dataset loaded from `load_dataset`.
      test_size: Proportion of data for the test set (default 0.2).
      val_size: Proportion of data for the validation set (default 0.1). Must be between 0 and 1.
      stratify_by: Column name to use for stratified splitting (maintains class balance).

  Returns:
      A dictionary with keys 'train', 'val', and 'test' containing the respective datasets.
  """

    # Input validation
    if val_size < 0 or val_size > 1:
        raise ValueError("Validation size must be between 0 and 1.")

    # Ensure test_size + val_size <= 1
    if test_size + val_size > 1:
        raise ValueError("Combined test and validation size cannot exceed 1.")

    # Split training and test sets (with stratification if specified)
    train_test = dataset.train_test_split(test_size=test_size)
    train_set = train_test["train"]
    test_set = train_test["test"]

    # Further split training set into training and validation sets (with stratification)
    if val_size > 0:
        val_split = train_set.train_test_split(test_size=val_size / (1 - test_size))
        train_set = val_split["train"]
        val_set = val_split["test"]
    else:
        val_set = None

    return {"train": train_set, "val": val_set, "test": test_set}
