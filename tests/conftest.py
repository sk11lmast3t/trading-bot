import os


os.environ["PYTHON_DOTENV_DISABLED"] = "1"
for _name in (
    "API_KEY",
    "API_SECRET",
    "API_PASSPHRASE",
    "EXCHANGE_NAME",
    "TESTNET",
    "DEMO",
):
    os.environ.pop(_name, None)
