import argparse

from .evaluation import EvaluationCommand


def run_benchmark():
    EvaluationCommand().execute()
