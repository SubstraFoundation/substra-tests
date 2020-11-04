
import json
import os
from pathlib import Path
import pytest
import tempfile

import substra

SECRETS_PATH = "/sandbox/chainkeys/"

CHAINKEY_ALGO_SCRIPT = f"""
import json
from pathlib import Path
import substratools as tools

CHAINKEY_PATH = Path("/sandbox") / "chainkeys" / "chainkeys.json"

class TestAlgo(tools.Algo):
    def train(self, X, y, models, rank):
        with CHAINKEY_PATH.open('r') as fp:
            data = json.load(fp)
            print(f"{{data['partner']}}")
            print(f"{{data['compute_plan_tag']}}")
        return [1 for x in X]

    def predict(self, X, model):
        res = [x * model['value'] for x in X]
        with CHAINKEY_PATH.open('r') as fp:
            data = json.load(fp)
            print(f"{{data['partner']}}")
            print(f"{{data['compute_plan_tag']}}")
        return [x * model['value'] for x in X]

    def load_model(self, path):
        with open(path) as f:
            return json.load(f)

    def save_model(self, model, path):
        with open(path, 'w') as f:
            return json.dump(model, f)

if __name__ == '__main__':
    tools.algo.execute(TestAlgo())
"""

@pytest.fixture
def partners_list():
    return ["partner1", "partner2"]


@pytest.fixture
def cp_tag_list():
    return ["cp_0"]


@pytest.fixture
def debug_chainkey_dir(partners_list, cp_tag_list):
    temp_dir = tempfile.TemporaryDirectory()
    os.setenv("CHAINKEYS_ENABLED", True)
    os.setenv("CHAINKEYS_DIR", temp_dir)
    for partner in partners_list:
        for compute_plan_tag in cp_tag_list:
            chainkey_dir = Path(temp_dir.name) / partner / "computeplan" / compute_plan_tag / "chainkeys"
            chainkey_dir.mkdir(parents=True)
            data = {
                "partner": partner,
                "compute_plan_tag": compute_plan_tag
            }
            with (chainkey_dir / "chainkeys.json").open('w') as fp:
                json.dump(data, fp)
    return temp_dir


@pytest.mark.local_only
def test_debug_chainkeys(
    debug_client,
    debug_chainkey_dir,
    partners_list,
    cp_tag_list,
    factory,
):
    spec = factory.create_algo(py_script=CHAINKEY_ALGO_SCRIPT)
    algo = debug_client.add_algo(spec)

    spec = factory.create_dataset()
    dataset = debug_client.add_dataset(spec)

    spec = factory.create_objective()
    objective = debug_client.add_objective(spec)

    for partner in partners_list:
        for cp_tag in cp_tag_list:

            cp_spec = factory.create_compute_plan()
            cp_spec.tag = cp_tag

            traintuple_spec = cp_spec.add_traintuple(
                algo=algo,
                dataset=dataset,
                metadata={substra.sdk.DEBUG_OWNER: partner}
            )
            cp_spec.add_testtuple(
                objective=objective,
                traintuple_spec=traintuple_spec,
                metadata={substra.sdk.DEBUG_OWNER: partner}
            )

            cp = debug_client.add_compute_plan(cp_spec)
            traintuples = debug_client.list_compute_plan_traintuples(cp.compute_plan_id)
            testtuples = debug_client.list_compute_plan_testtuples(cp.compute_plan_id)

            tuples = traintuples + testtuples
            for t in tuples:
                assert t.status == substra.sdk.models.Status.done
