from alpaca.system.utils.artifact import Artifact


class SelectionReport(Artifact):
    """
    Report of selected datapoints (at one query step)
    """

    def __init__(self, pool_size: int = None, selected_ids: list[int] = None, **kwargs):
        super().__init__(**kwargs)
        self.pool_size: int = pool_size
        self.selected_ids: int = selected_ids


class PerformanceReport(Artifact):
    """
    Report of a model's performance (RMSE loss)
    """

    def __init__(
        self,
        train_size: int = None,
        pool_size: int = None,
        test_size: int = None,
        train_rmse: float = None,
        pool_rmse: float = None,
        test_rmse: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.train_size: int = train_size
        self.pool_size: int = pool_size
        self.test_size: int = test_size
        self.train_rmse: float = train_rmse
        self.pool_rmse: float = pool_rmse
        self.test_rmse: float = test_rmse


class IterationReport(Artifact):
    """
    Accumulates the various reports generated during one active learning iteration.
    """

    def __init__(
        self,
        iteration: int = None,
        model_fit: Artifact = None,
        performance: PerformanceReport = None,
        selection: SelectionReport = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.iteration: int = iteration
        self.model_fit: Artifact = model_fit
        self.performance: PerformanceReport = performance
        self.selection: SelectionReport = selection


class EpochReport(Artifact):
    """
    Iteratively trained models generate one epoch report after every epoch.

    Typically the fit method of such models return an ArtifactSeries of EpochReports.

    To get a list of the individual EpochReports from the ArtifactSeries use its "artifacts" property.
    """

    def __init__(
        self,
        epoch: int = None,
        minibatch_size: int = None,
        train_size: int = None,
        train_loss: float = None,
        validation_size: int = None,
        validation_rmse: float = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.epoch: int = epoch
        self.minibatch_size: int = minibatch_size
        self.train_size: int = train_size
        self.train_loss: float = train_loss
        self.validation_size: int = validation_size
        self.validation_rmse: float = validation_rmse
