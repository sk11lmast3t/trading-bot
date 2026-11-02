from pathlib import Path
r = Path('.')
# settings
ps = r / 'config' / 'settings.py'
ts = ps.read_text()
old = '''@dataclass
class ExchangeConfig:
    name: str = \"binance\"
    api_key: str = os.getenv(\"API_KEY\", \"\")
    api_secret: str = os.getenv(\"API_SECRET\", \"\")
    testnet: bool = True
    rate_limit: int = 1200'''
new = '''@dataclass
class ExchangeConfig:
    name: str = os.getenv(\"EXCHANGE_NAME\", \"binance\")
    api_key: str = os.getenv(\"API_KEY\", \"\")
    api_secret: str = os.getenv(\"API_SECRET\", \"\")
    testnet: bool = True
    demo: bool = os.getenv(\"DEMO\", \"false\").lower() == \"true\"
    rate_limit: int = 1200'''
assert old in ts
ps.write_text(ts.replace(old, new))
# exchange
pe = r / 'core' / 'exchange.py'
te = pe.read_text()
old2 = '''    def connect(self):
        ec = self.config.exchange
        exchange_class = getattr(ccxt, ec.name, ccxt.binance)
        params = {
            'apiKey': ec.api_key,
            'secret': ec.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        }
        if ec.testnet:
            params['sandbox'] = True
        self.exchange = exchange_class(params)
        logger.info(f\"Connected to {ec.name} (sandbox={ec.testnet})\")
        return self'''
new2 = '''    def connect(self):
        ec = self.config.exchange
        exchange_class = getattr(ccxt, ec.name, ccxt.binance)
        params = {
            'apiKey': ec.api_key,
            'secret': ec.api_secret,
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},
        }
        if ec.demo:
            params['options'] = {'demo': True}
        elif ec.testnet:
            params['sandbox'] = True
        self.exchange = exchange_class(params)
        logger.info(f\"Connected to {ec.name} (sandbox={ec.testnet})\")
        return self'''
assert old2 in te
pe.write_text(te.replace(old2, new2))
# main
pm = r / 'main.py'
tm = pm.read_text()
old3 = '''if __name__ == \"__main__\":
    mode = \"paper\"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()

    cfg = BotConfig()
    cfg.mode = mode
    bot = UnifiedCryptoBot(cfg)
    bot.run()'''
new3 = '''if __name__ == \"__main__\":
    mode = \"paper\"
    demo_flag = False
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == \"--demo\":
                demo_flag = True
                continue
            if arg.lower() in (\"live\", \"paper\"):
                mode = arg.lower()
                continue
            if arg.lower().startswith(\"--mode=\"):
                mode = arg.lower().split(\"=\", 1)[1]
                continue

    cfg = BotConfig()
    if demo_flag:
        cfg.exchange.demo = True
        cfg.exchange.testnet = False
        mode = \"live\"
    cfg.mode = mode
    if cfg.mode == \"live\":
        if not cfg.exchange.api_key or not cfg.exchange.api_secret:
            logger.error(\"API key or secret missing for live mode\")
            sys.exit(1)
    bot = UnifiedCryptoBot(cfg)
    bot.run()'''
assert old3 in tm
pm.write_text(tm.replace(old3, new3))
print('done')