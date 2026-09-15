import argparse
from pathlib import Path

from azure.ai.ml import Input, MLClient, Output, command
from azure.ai.ml.dsl import pipeline
from azure.identity import DefaultAzureCredential

COMPONENTS_DIR = Path(__file__).resolve().parent.parent / "components"
ENVIRONMENT = "azureml:forecaster-env:4"
COMPUTE = "cpu-cluster"


def build_components(key_vault_url: str, training_identity_client_id: str, subscription_id: str,
                      resource_group: str, workspace_name: str):
    data_prep = command(
        name="data_prep",
        display_name="Assemble product order history",
        code=str(COMPONENTS_DIR / "data_prep"),
        command=(
            "python data_prep.py "
            f"--key-vault-url {key_vault_url} "
            f"--managed-identity-client-id {training_identity_client_id} "
            "--train-output ${{outputs.train_data}} "
            "--val-output ${{outputs.val_data}} "
            "--test-output ${{outputs.test_data}}"
        ),
        environment=ENVIRONMENT,
        outputs={
            "train_data": Output(type="uri_folder"),
            "val_data": Output(type="uri_folder"),
            "test_data": Output(type="uri_folder"),
        },
    )

    train = command(
        name="train",
        display_name="Train quantity regressor",
        code=str(COMPONENTS_DIR / "train"),
        command=(
            "python train.py "
            "--train-data ${{inputs.train_data}} "
            "--val-data ${{inputs.val_data}} "
            "--model-output ${{outputs.model}}"
        ),
        environment=ENVIRONMENT,
        inputs={"train_data": Input(type="uri_folder"), "val_data": Input(type="uri_folder")},
        outputs={"model": Output(type="uri_folder")},
    )

    evaluate = command(
        name="evaluate",
        display_name="Evaluate vs. baseline",
        code=str(COMPONENTS_DIR / "evaluate"),
        command=(
            "python evaluate.py "
            "--test-data ${{inputs.test_data}} "
            "--model-input ${{inputs.model}} "
            "--result-output ${{outputs.result}}"
        ),
        environment=ENVIRONMENT,
        inputs={"test_data": Input(type="uri_folder"), "model": Input(type="uri_folder")},
        outputs={"result": Output(type="uri_folder")},
    )

    register = command(
        name="register",
        display_name="Register model if it beats baseline",
        code=str(COMPONENTS_DIR / "register"),
        command=(
            "python register.py "
            "--result-input ${{inputs.result}} "
            "--model-input ${{inputs.model}} "
            f"--subscription-id {subscription_id} "
            f"--resource-group {resource_group} "
            f"--workspace-name {workspace_name} "
            f"--managed-identity-client-id {training_identity_client_id}"
        ),
        environment=ENVIRONMENT,
        inputs={"result": Input(type="uri_folder"), "model": Input(type="uri_folder")},
    )

    return data_prep, train, evaluate, register


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--key-vault-url", required=True)
    parser.add_argument("--training-identity-client-id", required=True)
    args = parser.parse_args()

    data_prep, train, evaluate, register = build_components(
        args.key_vault_url,
        args.training_identity_client_id,
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
    )

    @pipeline(compute=COMPUTE, description="Trains and conditionally registers the per-product quantity forecaster.")
    def train_forecaster_pipeline():
        prep_step = data_prep()
        train_step = train(train_data=prep_step.outputs.train_data, val_data=prep_step.outputs.val_data)
        evaluate_step = evaluate(test_data=prep_step.outputs.test_data, model=train_step.outputs.model)
        register(result=evaluate_step.outputs.result, model=train_step.outputs.model)

    ml_client = MLClient(DefaultAzureCredential(), args.subscription_id, args.resource_group, args.workspace_name)

    pipeline_job = train_forecaster_pipeline()
    submitted = ml_client.jobs.create_or_update(pipeline_job)
    print(f"Submitted pipeline job: {submitted.name}")
    print(f"Studio URL: {submitted.studio_url}")


if __name__ == "__main__":
    main()
