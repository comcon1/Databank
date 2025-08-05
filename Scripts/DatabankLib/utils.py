from DatabankLib import RCODE_COMPUTED, RCODE_ERROR, RCODE_SKIPPED
from DatabankLib.core import initialize_databank


from logging import Logger
from typing import Callable


def run_analysis(
        method: Callable,
        logger: Logger,
        id_range=(None, None)
        ):
    """
    Apply analysis ``method`` to the entire databank.

    :param method: (Callable) will be called as ``fun(system, logger)``
    :param logger: (Logger) reference to Logger initialized by the top script
    :param id_range: (A,B) filter for systems to analyze, default is 
                     (None, None) which means all systems. Can be also (None, -1)
                     which means all new systems, or (0, None) which means all 
                     old systems.

    :return: None
    """
    systems = initialize_databank()

    list_ids = [s['ID'] for s in systems]
    if id_range[0] is not None:
        list_ids = [s for s in list_ids if s >= id_range[0]]
    if id_range[1] is not None:
        list_ids = [s for s in list_ids if s <= id_range[1]]
    logger.info(f"""
==> PREPARING ANALYSIS {method.__name__.upper()} <==
    ├── Filtering systems by range: {id_range}
    └── Number of systems in databank: {len(list_ids)}
    """)

    result_dict = {
        RCODE_COMPUTED: 0,
        RCODE_SKIPPED: 0,
        RCODE_ERROR: 0}

    for id in list_ids:
        system = systems.loc(id)
        # printing pre-computational information in a tree-like structure
        logger.info(f"""
....Will run analysis called: {method.__name__}
    ├── ID: {id}
    ├── System title: {system['SYSTEM']}
    └── System path: {system['path']}
        """)
        res = method(system, logger)
        logger.info(f"""
....Finished calculating {method.__name__} for system {id}
    └── Result of analysis: {res}
        """)
        result_dict[res] += 1

    logger.info(f"""
==> RESULTS OF ANALYSIS {method.__name__.upper()} FOR SYSTEMS IN RANGE {id_range} <==
    ├── COMPUTED: {result_dict[RCODE_COMPUTED]}
    ├── SKIPPED: {result_dict[RCODE_SKIPPED]}
    └── ERROR: {result_dict[RCODE_ERROR]}
    """)
