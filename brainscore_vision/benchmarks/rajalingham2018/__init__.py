from brainscore_vision import benchmark_registry
from .benchmark import DicarloRajalingham2018I2n
from .benchmark import RajalinghamMatchtosamplePublicBenchmark

benchmark_registry['dicarlo.Rajalingham2018-i2n'] = DicarloRajalingham2018I2n
benchmark_registry['dicarlo.Rajalingham2018public-i2n'] = RajalinghamMatchtosamplePublicBenchmark
