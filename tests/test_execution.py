import pytest

import substra

import substratest as sbt
from substratest.factory import Permissions

from substratest import assets
from . import settings


@pytest.mark.slow
def test_tuples_execution_on_same_node(factory, client, default_dataset, default_objective):
    """Execution of a traintuple, a following testtuple and a following traintuple."""

    spec = factory.create_algo()
    algo = client.add_algo(spec)

    # create traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    traintuple = client.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None

    # check we cannot add twice the same traintuple
    with pytest.raises(substra.exceptions.AlreadyExists):
        client.add_traintuple(spec)

    # create testtuple
    # don't create it before to avoid MVCC errors
    spec = factory.create_testtuple(objective=default_objective, traintuple=traintuple)
    testtuple = client.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done

    # add a traintuple depending on first traintuple
    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        traintuples=[traintuple],
    )
    traintuple = client.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert len(traintuple.in_models) == 1


@pytest.mark.slow
def test_federated_learning_workflow(factory, client, default_datasets):
    """Test federated learning workflow on each node."""

    # create test environment
    spec = factory.create_algo()
    algo = client.add_algo(spec)

    # create 1 traintuple per dataset and chain them
    traintuple = None
    rank = 0
    compute_plan_id = None
    for dataset in default_datasets:
        traintuples = [traintuple] if traintuple else []
        spec = factory.create_traintuple(
            algo=algo,
            dataset=dataset,
            data_samples=dataset.train_data_sample_keys,
            traintuples=traintuples,
            tag='foo',
            rank=rank,
            compute_plan_id=compute_plan_id,
        )
        traintuple = client.add_traintuple(spec).future().wait()
        assert traintuple.status == assets.Status.done
        assert traintuple.out_model is not None
        assert traintuple.tag == 'foo'
        assert traintuple.compute_plan_id   # check it is not None or ''

        rank += 1
        compute_plan_id = traintuple.compute_plan_id

    # check a compute plan has been created and its status is at done
    cp = client.get_compute_plan(compute_plan_id)
    assert cp.status == assets.Status.done


@pytest.mark.slow
def test_tuples_execution_on_different_nodes(factory, client_1, client_2, default_objective_1, default_dataset_2):
    """Execution of a traintuple on node 1 and the following testtuple on node 2."""
    # add test data samples / dataset / objective on node 1

    spec = factory.create_algo()
    algo_2 = client_2.add_algo(spec)

    # add traintuple on node 2; should execute on node 2 (dataset located on node 2)
    spec = factory.create_traintuple(
        algo=algo_2,
        dataset=default_dataset_2,
        data_samples=default_dataset_2.train_data_sample_keys,
    )
    traintuple = client_1.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.done
    assert traintuple.out_model is not None
    assert traintuple.dataset.worker == client_2.node_id

    # add testtuple; should execute on node 1 (objective dataset is located on node 1)
    spec = factory.create_testtuple(objective=default_objective_1, traintuple=traintuple)
    testtuple = client_1.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done
    assert testtuple.dataset.worker == client_1.node_id


@pytest.mark.slow
def test_traintuple_execution_failure(factory, client, default_dataset_1):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_algo(py_script=sbt.factory.INVALID_ALGO_SCRIPT)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset_1,
        data_samples=default_dataset_1.train_data_sample_keys,
    )
    traintuple = client.add_traintuple(spec).future().wait(raises=False)
    assert traintuple.status == assets.Status.failed
    assert traintuple.out_model is None


@pytest.mark.slow
def test_composite_traintuple_execution_failure(factory, client, default_dataset):
    """Invalid composite algo script is causing traintuple failure."""

    spec = factory.create_composite_algo(py_script=sbt.factory.INVALID_COMPOSITE_ALGO_SCRIPT)
    algo = client.add_composite_algo(spec)

    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    composite_traintuple = client.add_composite_traintuple(spec).future().wait(raises=False)
    assert composite_traintuple.status == assets.Status.failed
    assert composite_traintuple.out_head_model.out_model is None
    assert composite_traintuple.out_trunk_model.out_model is None


@pytest.mark.slow
def test_aggregatetuple_execution_failure(factory, client, default_dataset):
    """Invalid algo script is causing traintuple failure."""

    spec = factory.create_composite_algo()
    composite_algo = client.add_composite_algo(spec)

    spec = factory.create_aggregate_algo(py_script=sbt.factory.INVALID_AGGREGATE_ALGO_SCRIPT)
    aggregate_algo = client.add_aggregate_algo(spec)

    composite_traintuples = []
    for i in [0, 1]:
        spec = factory.create_composite_traintuple(
            algo=composite_algo,
            dataset=default_dataset,
            data_samples=[default_dataset.train_data_sample_keys[i]],
        )
        composite_traintuples.append(client.add_composite_traintuple(spec))

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        traintuples=composite_traintuples,
        worker=client.node_id,
    )
    aggregatetuple = client.add_aggregatetuple(spec).future().wait(raises=False)
    for composite_traintuple in composite_traintuples:
        composite_traintuple = client.get_composite_traintuple(composite_traintuple.key)
        assert composite_traintuple.status == assets.Status.done
    assert aggregatetuple.status == assets.Status.failed
    assert aggregatetuple.out_model is None


@pytest.mark.slow
def test_composite_traintuples_execution(factory, client, default_dataset, default_objective):
    """Execution of composite traintuples."""

    spec = factory.create_composite_algo()
    algo = client.add_composite_algo(spec)

    # first composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
    )
    composite_traintuple_1 = client.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_1.status == assets.Status.done
    assert composite_traintuple_1.out_head_model is not None
    assert composite_traintuple_1.out_head_model.out_model is not None
    assert composite_traintuple_1.out_trunk_model is not None
    assert composite_traintuple_1.out_trunk_model.out_model is not None

    # second composite traintuple
    spec = factory.create_composite_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        head_traintuple=composite_traintuple_1,
        trunk_traintuple=composite_traintuple_1,
    )
    composite_traintuple_2 = client.add_composite_traintuple(spec).future().wait()
    assert composite_traintuple_2.status == assets.Status.done
    assert composite_traintuple_2.out_head_model is not None
    assert composite_traintuple_2.out_trunk_model is not None

    # add a 'composite' testtuple
    spec = factory.create_testtuple(objective=default_objective, traintuple=composite_traintuple_2)
    testtuple = client.add_testtuple(spec).future().wait()
    assert testtuple.status == assets.Status.done

    # list composite traintuple
    composite_traintuples = client.list_composite_traintuple()
    composite_traintuple_keys = set([t.key for t in composite_traintuples])
    assert set([composite_traintuple_1.key, composite_traintuple_2.key]).issubset(
        composite_traintuple_keys
    )


@pytest.mark.slow
def test_aggregatetuple(factory, client, default_dataset):
    """Execution of aggregatetuple aggregating traintuples."""

    number_of_traintuples_to_aggregate = 3

    train_data_sample_keys = default_dataset.train_data_sample_keys[:number_of_traintuples_to_aggregate]

    spec = factory.create_algo()
    algo = client.add_algo(spec)

    # add traintuples
    traintuples = []
    for data_sample_key in train_data_sample_keys:
        spec = factory.create_traintuple(
            algo=algo,
            dataset=default_dataset,
            data_samples=[data_sample_key],
        )
        traintuple = client.add_traintuple(spec).future().wait()
        traintuples.append(traintuple)

    spec = factory.create_aggregate_algo()
    aggregate_algo = client.add_aggregate_algo(spec)

    spec = factory.create_aggregatetuple(
        algo=aggregate_algo,
        worker=client.node_id,
        traintuples=traintuples,
    )
    aggregatetuple = client.add_aggregatetuple(spec).future().wait()
    assert aggregatetuple.status == assets.Status.done
    assert len(aggregatetuple.in_models) == number_of_traintuples_to_aggregate


@pytest.mark.slow
def test_aggregate_composite_traintuples(factory, network, clients, default_datasets, default_objectives):
    """Do 2 rounds of composite traintuples aggregations on multiple nodes.

    Compute plan details:

    Round 1:
    - Create 2 composite traintuples executed on two datasets located on node 1 and
      node 2.
    - Create an aggregatetuple on node 1, aggregating the two previous composite
      traintuples (trunk models aggregation).

    Round 2:
    - Create 2 composite traintuples executed on each nodes that depend on: the
      aggregated tuple and the previous composite traintuple executed on this node. That
      is to say, the previous round aggregated trunk models from all nodes and the
      previous round head model from this node.
    - Create an aggregatetuple on node 1, aggregating the two previous composite
      traintuples (similar to round 1 aggregatetuple).
    - Create a testtuple for each previous composite traintuples and aggregate tuple
      created during this round.

    (optional) if the option "enable_intermediate_model_removal" is True:
    - Since option "enable_intermediate_model_removal" is True, the aggregate model created on round 1 should
      have been deleted from the backend after round 2 has completed.
    - Create a traintuple that depends on the aggregate tuple created on round 1. Ensure that it fails to start.

    This test refers to the model composition use case.
    """

    aggregate_worker = clients[0].node_id
    number_of_rounds = 2

    # register algos on first node
    spec = factory.create_composite_algo()
    composite_algo = clients[0].add_composite_algo(spec)
    spec = factory.create_aggregate_algo()
    aggregate_algo = clients[0].add_aggregate_algo(spec)

    # launch execution
    previous_aggregatetuple = None
    previous_composite_traintuples = []

    for round_ in range(number_of_rounds):
        # create composite traintuple on each node
        composite_traintuples = []
        for index, dataset in enumerate(default_datasets):
            kwargs = {}
            if previous_aggregatetuple:
                kwargs = {
                    'head_traintuple': previous_composite_traintuples[index],
                    'trunk_traintuple': previous_aggregatetuple,
                }
            spec = factory.create_composite_traintuple(
                algo=composite_algo,
                dataset=dataset,
                data_samples=[dataset.train_data_sample_keys[0 + round_]],
                permissions=Permissions(public=False, authorized_ids=[c.node_id for c in clients]),
                **kwargs,
            )
            t = clients[0].add_composite_traintuple(spec).future().wait()
            composite_traintuples.append(t)

        # create aggregate on its node
        spec = factory.create_aggregatetuple(
            algo=aggregate_algo,
            worker=aggregate_worker,
            traintuples=composite_traintuples,
        )
        aggregatetuple = clients[0].add_aggregatetuple(spec).future().wait()

        # save state of round
        previous_aggregatetuple = aggregatetuple
        previous_composite_traintuples = composite_traintuples

    # last round: create associated testtuple
    for traintuple, objective in zip(previous_composite_traintuples, default_objectives):
        spec = factory.create_testtuple(
            objective=objective,
            traintuple=traintuple,
        )
        clients[0].add_testtuple(spec).future().wait()

    if not network.options.enable_intermediate_model_removal:
        return

    # Optional (if "enable_intermediate_model_removal" is True): ensure the aggregatetuple of round 1 has been deleted.
    #
    # We do this by creating a new traintuple that depends on the deleted aggregatatuple, and ensuring that starting
    # the traintuple fails.
    #
    # Ideally it would be better to try to do a request "as a backend" to get the deleted model. This would be closer
    # to what we want to test and would also check that this request is correctly handled when the model has been
    # deleted. Here, we cannot know for sure the failure reason. Unfortunately this cannot be done now as the
    # username/password are not available in the settings files.

    client = clients[0]
    dataset = default_datasets.sort_by(client.node_id)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=dataset,
        data_samples=dataset.train_data_sample_keys,
    )
    traintuple = client.add_traintuple(spec).future().wait()
    assert traintuple.status == assets.Status.failed


@pytest.mark.parametrize('fail_count,status', (
    (settings.CELERY_TASK_MAX_RETRIES, 'done'),
    (settings.CELERY_TASK_MAX_RETRIES + 1, 'failed'),
))
def test_execution_retry_on_fail(fail_count, status, factory, client, default_dataset):
    """Execution of a traintuple which fails on the N first tries, and suceeds on the N+1th try"""

    # This test ensures the compute task retry mechanism works correctly.
    #
    # It executes an algorithm that `raise`s on the first N runs, and then
    # succeeds.
    #
    # /!\ This test should ideally be part of the substra-backend project,
    #     not substra-tests. For the sake of expendiency, we're keeping it
    #     as part of substra-tests for now, but we intend to re-implement
    #     it in substra-backend instead eventually.
    # /!\ This test makes use of the "local" folder to keep track of a counter.
    #     This is a hack to make the algo raise or succeed depending on the retry
    #     count. Ideally, we would use a more elegant solution.
    # /!\ This test doesn't validate that an error in the docker build phase (of
    #     the compute task execution) triggers a retry. Ideally, it's that case that
    #     would be tested, since errors in the docker build are the main use-case
    #     the retry feature was build for.

    retry_algo_snippet_toreplace = """
    tools.algo.execute(TestAlgo())"""

    retry_snippet_replacement = f"""
    counter_path = "/sandbox/local/counter"
    counter = 0
    try:
        with open(counter_path) as f:
            counter = int(f.read())
    except IOError:
        pass # file doesn't exist yet

    # Fail if the counter is below the retry count
    if counter < {fail_count}:
        counter = counter + 1
        with open(counter_path, 'w') as f:
            f.write(str(counter))
        raise Exception("Intentionally keep on failing until we have failed {fail_count} time(s). The algo has now \
            failed " + str(counter) + " time(s).")

    # The counter is greater than the retry count
    tools.algo.execute(TestAlgo())"""

    py_script = sbt.factory.DEFAULT_ALGO_SCRIPT.replace(retry_algo_snippet_toreplace, retry_snippet_replacement)
    spec = factory.create_algo(py_script)
    algo = client.add_algo(spec)

    spec = factory.create_traintuple(
        algo=algo,
        dataset=default_dataset,
        data_samples=default_dataset.train_data_sample_keys,
        rank=0,  # make sure it's part of a compute plan, so we have access to the /sandbox/local
                 # folder (that's where we store the counter)
    )
    traintuple = client.add_traintuple(spec).future().wait(raises=False)

    # Assuming that, on the backend, CELERY_TASK_MAX_RETRIES is set to 1, the algo
    # should be retried up to 1 time(s) (i.e. max 2 attempts in total)
    # - if it fails less than 2 times, it should be marked as "done"
    # - if it fails 2 times or more, it should be marked as "failed"
    assert traintuple.status == status
