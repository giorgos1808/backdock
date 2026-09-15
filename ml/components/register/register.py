import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Model
from azure.identity import ManagedIdentityCredential


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-input", required=True)
    parser.add_argument("--model-input", required=True)
    parser.add_argument("--subscription-id", required=True)
    parser.add_argument("--resource-group", required=True)
    parser.add_argument("--workspace-name", required=True)
    parser.add_argument("--managed-identity-client-id", default=None)
    args = parser.parse_args()

    with open(os.path.join(args.result_input, "result.json")) as f:
        result = json.load(f)

    if not result["passed"]:
        print(
            f"register: skipped — model_mape={result['model_mape']:.2f} did not beat "
            f"baseline_mape={result['baseline_mape']:.2f}"
        )
        return

    credential = (
        ManagedIdentityCredential(client_id=args.managed_identity_client_id)
        if args.managed_identity_client_id
        else ManagedIdentityCredential()
    )
    ml_client = MLClient(credential, args.subscription_id, args.resource_group, args.workspace_name)

    model = Model(
        path=args.model_input,
        type=AssetTypes.CUSTOM_MODEL,
        name="product-quantity-forecaster",
        description=(
            f"LightGBM per-product quantity forecaster. "
            f"Test MAPE {result['model_mape']:.2f} vs. baseline {result['baseline_mape']:.2f}."
        ),
    )
    registered = ml_client.models.create_or_update(model)
    print(f"register: registered {registered.name} v{registered.version}")


if __name__ == "__main__":
    main()
