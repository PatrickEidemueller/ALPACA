import json
from datetime import datetime, timedelta

from alpaca.system.utils.printing import to_string


def _datetime_to_str(t: datetime) -> str:
    return t.strftime("%Y%m%d_%H:%M:%S")


def _datetime_from_str(s: str) -> datetime:
    return datetime.strptime(s, "%Y%m%d_%H:%M:%S")


def _timedelta_to_str(td: timedelta) -> str:
    hours_days = td.days * 24
    hours_seconds, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours_days + hours_seconds}:{minutes}:{seconds}"


def _timedelta_from_str(s: str) -> timedelta:
    hours, minutes, seconds = [int(subs) for subs in s.split(":")]
    return timedelta(hours=hours, minutes=minutes, seconds=seconds)


class Artifact:
    """
    A general class for training / evaluation results, that are not part of the model itself.

    The interface demands that an artifact can be serialized to and parsed from string.
    Deriving classes may only contain:
        - JSON serializable members like str, int, float
        - Artifacts
        - python 'list' of such members (only use python 'list' not numpy arrays or other arrays!)
        - python 'dict' of such members

    E.g. Model accuracy and selected datapoints during an active learning iteration OR
         Loss and training time for a neural network training epoch. OR
         R^2 and parameter CIs for LinearRegression
    """

    def __init__(
        self,
        timestamp_start: datetime = datetime.now(),
        duration: timedelta = timedelta(hours=0, minutes=0, seconds=0),
    ):
        """
        An Artifact can be used to store and later load the result of some operation.

        @param timestamp_start : When the corrsponding operation was started
        @param duration : The duration of the corresponding operation
        """
        self.timestamp_start = _datetime_to_str(timestamp_start)
        self.duration = _timedelta_to_str(duration)
        self._artifact_type = type(self).__name__

    def __str__(self):
        return to_string(self)

    def get_duration(self) -> timedelta:
        return _timedelta_from_str(self.duration)

    def get_timestamp_start(self) -> datetime:
        return _datetime_from_str(self.timestamp_start)

    @staticmethod
    def typemap():
        tpmap = {c.__name__: c for c in Artifact.__subclasses__()}
        tpmap[Artifact.__name__] = Artifact
        return tpmap

    @staticmethod
    def _deparse(obj):
        if issubclass(type(obj), Artifact):
            return obj.to_dict()
        if isinstance(obj, dict):
            return {k: Artifact._deparse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Artifact._deparse(subobj) for subobj in obj]
        return obj

    @staticmethod
    def _parse(obj, parse__dict__=False):
        if not parse__dict__ and isinstance(obj, dict) and "_artifact_type" in obj:
            return Artifact.from_dict(obj)
        if isinstance(obj, dict):
            return {k: Artifact._parse(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [Artifact._parse(subobj) for subobj in obj]
        return obj

    def to_dict(self) -> "Artifact":
        return Artifact._deparse(self.__dict__)

    @staticmethod
    def from_dict(dct: dict) -> "Artifact":
        mytype = dct["_artifact_type"]
        artifact = Artifact.typemap()[mytype]()
        artifact.__dict__ = Artifact._parse(dct, parse__dict__=True)
        return artifact

    def save(self, filepath: str) -> None:
        with open(filepath, "w") as file:
            json.dump(self.to_dict(), file)

    @classmethod
    def load(cls, filepath: str) -> None:
        try:
            with open(filepath, "r") as file:
                json_dict = json.load(file)
            loaded = Artifact.from_dict(json_dict)
        except BaseException as e:
            raise RuntimeError(
                f"{cls.__name__}.load() could not load {filepath}: {type(e).__name__}: {e}"
            )
        if not isinstance(loaded, cls):
            raise RuntimeError(
                f"{cls.__name__}.load() expected a serialized artifact of class {cls.__name__} but"
                f"loaded artifact is of type {type(loaded).__name__} from {filepath} instead."
            )
        return loaded


class ArtifactSeries(Artifact):
    """
    Can represent a series of artifacts. The artifacts can be of different types.
    """

    def __init__(self):
        super().__init__()
        self._artifacts: list[Artifact] = []

    @property
    def artifacts(self) -> list["ArtifactSeries.ArtifactType"]:
        return self._artifacts

    def append(self, artifact: Artifact) -> None:
        if not issubclass(type(artifact), Artifact):
            raise ValueError(
                "ArtifactHistory.append can only append subclasses of 'Artifact'"
            )
        if len(self._artifacts) == 0:
            self.timestamp_start = artifact.timestamp_start
        self.duration = _timedelta_to_str(self.get_duration() + artifact.get_duration())
        self._artifacts.append(artifact)
