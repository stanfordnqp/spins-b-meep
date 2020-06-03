"""This module handles Meep simulations performed in parallel via MPI."""

import argparse
import logging
import sys

import dill
import meep as mp
from mpi4py import MPI

from spins import goos
from spins.goos_sim.meep.simulate import _simulate

logging.basicConfig(format=goos.LOG_FORMAT)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def simulate_parallel_main() -> None:
    """Simulates a simulation in parallel.

    This function is the entrance for simulating Meep simulations in parallel
    with MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    logger.info("Simulate process started, rank %d", rank)

    parser = argparse.ArgumentParser()
    parser.add_argument("simfile")
    parser.add_argument("--outfile")

    args = parse_args(comm, parser)

    with open(args.simfile, "rb") as fp:
        sim_args = dill.load(fp)

    if rank == 0:
        logger.info("Simulating %s", args.simfile)
    res = _simulate(**sim_args)

    if rank == 0:
        logger.info("Writing results to %s", args.outfile)
        with open(args.outfile, "wb") as fp:
            dill.dump(res, fp)

        logger.info("Simulation complete")


def parse_args(comm, parser: argparse.ArgumentParser):
    """Parses command line arguments.

    This function handles parsing of command line arguments. Only rank 0 will
    parse the arguments so that any error messages will not be redundantly
    printed to screen. The results are then broadcasted to other processes.

    Args:
        comm: MPI communication handler.
        parser: Configured parser object.

    Returns:
        Arguments as parsed by `ArgumentParser.parse_args`.
    """
    args = None
    try:
        if comm.Get_rank() == 0:
            args = parser.parse_args()
    finally:
        args = comm.bcast(args, root=0)

    if args is None:
        sys.exit(0)
    return args


if __name__ == "__main__":
    simulate_parallel_main()
