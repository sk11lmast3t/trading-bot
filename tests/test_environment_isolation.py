import os


def test_exchange_credentials_and_selection_are_absent():
    for name in (
        "API_KEY",
        "API_SECRET",
        "API_PASSPHRASE",
        "EXCHANGE_NAME",
        "TESTNET",
        "DEMO",
    ):
        assert name not in os.environ
