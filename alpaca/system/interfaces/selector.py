from alpaca.system.reports import SelectionReport
from alpaca.system.utils.stopwatch import StopWatch
from alpaca.system.interfaces.dataset import TrainPoolSet
from alpaca.system.interfaces.model import ModelInterface


class SelectorInterface:
    RequiredModelType = None
    RequiredDataLoaderType = None

    def _derived_select_batch(
        self, model: ModelInterface, train_pool_set: TrainPoolSet, batch_size: int
    ) -> list[int]:
        """
        Select a batch of pool datapoints for labeling.
        Does not do the labeling itself, so TrainPoolSet is not changed by this call.

        @params model : The model in case it is needed for labeling
        @params train_pool_set : Training and pool datasets, see class TrainPoolSet
        @params batch_size : Number of points to label

        @returns : List of indices of the selected pool datapoints
        """
        raise NotImplementedError()

    def select_batch(
        self, model: ModelInterface, train_pool_set: TrainPoolSet, batch_size: int
    ) -> SelectionReport:
        """
        Select a batch of pool datapoints and label them.
        Does not only the selection but also the labeling, so TrainPoolSet is changed by this call.

        @params model : The model in case it is needed for labeling
        @params train_pool_set : Training and pool datasets, see class TrainPoolSet
        @params batch_size : Number of points to label

        @returns : An artifact of type SelectionReport
        """
        if self.RequiredModelType is not None and not isinstance(
            model, self.RequiredModelType
        ):
            raise TypeError(
                f"{type(self)}.select_batch: Received model of type {type(model)} "
                f"but requires type {self.RequiredModelType}"
            )
        if self.RequiredDataLoaderType is not None and not isinstance(
            train_pool_set.data_loader, self.RequiredDataLoaderType
        ):
            raise TypeError(
                f"{type(self)}.select_batch: Received train_pool_set with data loader of type "
                f"{type(train_pool_set.data_loader)} but requires data loader type {self.RequiredDataLoaderType}"
            )
        with StopWatch() as stopwatch:
            indices = self._derived_select_batch(model, train_pool_set, batch_size)
            labeled_points = train_pool_set.label_datapoints(indices)
            return SelectionReport(
                pool_size=len(train_pool_set.pool),
                selected_ids=labeled_points.ids,
                timestamp_start=stopwatch.start,
                duration=stopwatch.elapsed(),
            )
