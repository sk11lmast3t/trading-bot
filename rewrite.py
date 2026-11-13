s=open("core/exchange.py").read()
nc="""class PaperExchange:
    def __init__(self, config):
        self.config = config
        self.exec_cfg = config.execution
        self.balance = config.scalping.initial_capital
        self.positions = {}
        self.trade_history = []
        self.order_id = 0
        self.orders = {}
        self.latency_ms = self.exec_cfg.paper_latency_ms
        self.base_slippage_pct = self.exec_cfg.paper_slippage_pct
        self.partial_fill_chance = self.exec_cfg.paper_partial_fill_chance
        self.reject_chance = self.exec_cfg.paper_reject_chance
        self._public_exchanges = []
        self._current_public = None
        self._init_public_exchanges()

    def _init_public_exchanges(self):
        candidates = []
        for name in ['kraken', 'coinbase', 'kucoin']:
            try:
                ex_class = getattr(ccxt, name)
                ex = ex_class({'enableRateLimit': True})
                try:
                    ex.fetch_ohlcv('BTC/USDT', '5m', limit=1)
                except Exception:
                    pass
                candidates.append(ex)
                if self._current_public is None:
                    self._current_public = ex
                logger.info('init %s' % getattr(ex,'id',name))
            except Exception as e:
                logger.warning('fail %s: %s' % (name,e))
        self._public_exchanges = candidates
        if self._current_public is None and candidates:
            self._current_public = candidates[0]
        if self._current_public is None:
            for name in ['kraken', 'coinbase', 'kucoin']:
                try:
                    ex_class = getattr(ccxt,name)
                    self._current_public = ex_class({'enableRateLimit':True})
                    self._public_exchanges = [self._current_public]
                    break
                except Exception:
                    continue
            if self._current_public is None:
                try:
                    self._current_public = ccxt.okx({'enableRateLimit':True})
                except Exception as e:
                    logger.error('no pub: %s' % e)

    def fetch_ohlcv(self, symbol: str, timeframe: str = '1m', limit: int = 200):
        tries = []
        if self._current_public is not None:
            tries.append(self._current_public)
        for ex in self._public_exchanges:
            if ex not in tries:
                tries.append(ex)
        if not tries:
            self._init_public_exchanges()
            if self._current_public is not None:
                tries.append(self._current_public)
            for ex in self._public_exchanges:
                if ex not in tries:
                    tries.append(ex)
        last=None
        for ex in tries:
            try:
                ohlcv = ex.fetch_ohlcv(symbol,timeframe,limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp','open','high','low','close','volume'])
                df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
                df.set_index('timestamp',inplace=True)
                self._current_public = ex
                return df
            except Exception as e:
                last=e
                logger.warning('fail %s: %s' % (getattr(ex,'id','?'),e))
                continue
        logger.error('all fail %s %s: %s' % (symbol,timeframe,last))
        return None
"""
a=s.find('class PaperExchange:')
b=s.find('    def _simulate_latency',a)
if a!=-1 and b!=-1:
    open("core/exchange.py","w").write(s[:a]+nc+"\n\n    def _simulate_latency"+s[b+len('    def _simulate_latency'):])
    print("ok")