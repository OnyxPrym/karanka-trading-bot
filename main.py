#!/usr/bin/env python3
"""
KARANKA MULTIVERSE V7 - cTrader Mobile Edition
24/5 Professional SMC Bot for IC Markets cTrader
"""

import os
import sys
import json
import time
import threading
import requests
from datetime import datetime, timedelta
from collections import defaultdict
from flask import Flask, render_template_string, jsonify, request
import pandas as pd
import numpy as np

# ============ CTRADER CONFIG ============
CTRADER_CONFIG = {
    'client_id': '19284_CKswqQmnC5403QlDqwBG8XrvLLgfn9psFXvBXWZkOdMlORJzg2',
    'client_secret': 'Tix0fEqff3Kg33qhr9DC5sKHgmlHHYkSxE1UzRsFc0fmxKhbfj',
    'token_url': 'https://openapi.ctrader.com/apps/token',
    'api_url': 'https://openapi.ctrader.com/v1',
    'demo_url': 'https://demo.ctraderapi.com/v1',
}

# ============ MARKET CONFIGS ============
MARKETS = {
    "EURUSD": {"pip": 0.0001, "digits": 5, "range": 0.007, "session": ["London", "NewYork"]},
    "GBPUSD": {"pip": 0.0001, "digits": 5, "range": 0.008, "session": ["London", "NewYork"]},
    "USDJPY": {"pip": 0.01, "digits": 3, "range": 0.80, "session": ["Asian", "London"]},
    "XAUUSD": {"pip": 0.1, "digits": 2, "range": 25, "session": ["London", "NewYork"]},
    "XAGUSD": {"pip": 0.01, "digits": 3, "range": 0.80, "session": ["London", "NewYork"]},
    "US30": {"pip": 1.0, "digits": 2, "range": 300, "session": ["NewYork"]},
    "USTEC": {"pip": 1.0, "digits": 2, "range": 250, "session": ["NewYork"]},
    "AUDUSD": {"pip": 0.0001, "digits": 5, "range": 0.0065, "session": ["Asian", "London"]},
    "BTCUSD": {"pip": 1.0, "digits": 2, "range": 1500, "session": ["All"]},
}

SESSIONS = {
    "Asian": {"hours": (0, 9), "trades_hour": 2, "risk": 0.8},
    "London": {"hours": (8, 17), "trades_hour": 4, "risk": 1.0},
    "NewYork": {"hours": (13, 22), "trades_hour": 5, "risk": 1.2},
    "Overlap": {"hours": (13, 17), "trades_hour": 6, "risk": 1.5},
}

# ============ CTRADER API CLIENT ============
class CTraderAPI:
    def __init__(self, demo_mode=True):
        self.demo = demo_mode
        self.token = None
        self.account_id = None
        self.connected = False
        self.base_url = CTRADER_CONFIG['demo_url'] if demo_mode else CTRADER_CONFIG['api_url']
        
    def connect(self):
        """Get OAuth token"""
        try:
            data = {
                'grant_type': 'client_credentials',
                'client_id': CTRADER_CONFIG['client_id'],
                'client_secret': CTRADER_CONFIG['client_secret']
            }
            
            response = requests.post(CTRADER_CONFIG['token_url'], data=data, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                self.token = result.get('access_token')
                self.connected = True
                return True, "Connected to cTrader"
            else:
                return False, f"Auth failed: {response.status_code}"
                
        except Exception as e:
            return False, f"Connection error: {str(e)}"
    
    def get_accounts(self):
        """Get trading accounts"""
        if not self.token:
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            response = requests.get(f'{self.base_url}/accounts', headers=headers, timeout=10)
            
            if response.status_code == 200:
                accounts = response.json().get('data', [])
                if accounts:
                    self.account_id = accounts[0].get('accountId')
                return accounts
        except:
            pass
        return []
    
    def get_price(self, symbol):
        """Get current price"""
        if not self.token or not self.account_id:
            return None
        
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            url = f'{self.base_url}/accounts/{self.account_id}/symbols/{symbol}/quotes'
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                bid = data.get('bid', 0)
                ask = data.get('ask', 0)
                return (bid + ask) / 2
        except:
            pass
        return None
    
    def get_historical_data(self, symbol, timeframe, bars=100):
        """Get historical candle data"""
        if not self.token or not self.account_id:
            return None
        
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            url = f'{self.base_url}/accounts/{self.account_id}/symbols/{symbol}/candles'
            params = {'period': timeframe, 'count': bars}
            
            response = requests.get(url, headers=headers, params=params, timeout=10)
            
            if response.status_code == 200:
                candles = response.json().get('data', [])
                if candles:
                    df = pd.DataFrame([{
                        'time': c.get('timestamp'),
                        'open': c.get('open'),
                        'high': c.get('high'),
                        'low': c.get('low'),
                        'close': c.get('close'),
                        'volume': c.get('volume', 0)
                    } for c in candles])
                    return df
        except:
            pass
        return None
    
    def place_order(self, symbol, direction, volume, sl, tp):
        """Place market order"""
        if not self.token or not self.account_id:
            return False, "Not connected"
        
        try:
            headers = {
                'Authorization': f'Bearer {self.token}',
                'Content-Type': 'application/json'
            }
            
            order_data = {
                'accountId': self.account_id,
                'symbolName': symbol,
                'tradeSide': direction.upper(),
                'volume': int(volume * 100000),  # Convert to micro lots
                'stopLoss': sl,
                'takeProfit': tp
            }
            
            url = f'{self.base_url}/accounts/{self.account_id}/orders'
            response = requests.post(url, headers=headers, json=order_data, timeout=10)
            
            if response.status_code in [200, 201]:
                result = response.json()
                return True, result.get('orderId', 'Success')
            else:
                return False, f"Order failed: {response.status_code}"
                
        except Exception as e:
            return False, f"Order error: {str(e)}"
    
    def get_positions(self):
        """Get open positions"""
        if not self.token or not self.account_id:
            return []
        
        try:
            headers = {'Authorization': f'Bearer {self.token}'}
            url = f'{self.base_url}/accounts/{self.account_id}/positions'
            response = requests.get(url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                return response.json().get('data', [])
        except:
            pass
        return []

# ============ SMC ANALYZER ============
class SMCAnalyzer:
    def __init__(self):
        self.cache = {}
        
    def analyze(self, symbol, config):
        """Analyze symbol with SMC"""
        api = ctrader_api
        
        # Get multi-timeframe data
        m5 = api.get_historical_data(symbol, 'M5', 80)
        m15 = api.get_historical_data(symbol, 'M15', 80)
        h1 = api.get_historical_data(symbol, 'H1', 100)
        h4 = api.get_historical_data(symbol, 'H4', 100)
        
        if m5 is None or len(m5) < 20:
            return None
        
        price = api.get_price(symbol)
        if not price:
            return None
        
        # Analyze strategies
        analyses = []
        
        # Scalp M5+M15
        if m15 is not None:
            scalp = self._analyze_scalp(m5, m15, price, config)
            if scalp['confidence'] > 0:
                analyses.append(scalp)
        
        # Intraday M15+H1
        if h1 is not None:
            intraday = self._analyze_intraday(m15, h1, price, config)
            if intraday['confidence'] > 0:
                analyses.append(intraday)
        
        # Swing H1+H4
        if h4 is not None:
            swing = self._analyze_swing(h1, h4, price, config)
            if swing['confidence'] > 0:
                analyses.append(swing)
        
        if not analyses:
            return None
        
        # Get best analysis
        best = max(analyses, key=lambda x: x['confidence'])
        
        # Get session
        session = self._get_session()
        session_config = SESSIONS[session]
        
        # Calculate final confidence
        confidence = best['confidence']
        
        # Session adjustment
        if symbol in config.get('session', []):
            confidence += 5
        
        if confidence < 65:
            return None
        
        # Calculate SL/TP
        direction = best['direction']
        pip = config['pip']
        
        sl_pips = 20 * session_config['risk']
        tp_pips = 40 * session_config['risk']
        
        if direction == 'BUY':
            sl = price - (pip * sl_pips)
            tp = price + (pip * tp_pips)
        else:
            sl = price + (pip * sl_pips)
            tp = price - (pip * tp_pips)
        
        sl = round(sl, config['digits'])
        tp = round(tp, config['digits'])
        
        return {
            'symbol': symbol,
            'direction': direction,
            'confidence': confidence,
            'price': price,
            'sl': sl,
            'tp': tp,
            'strategy': best['strategy'],
            'signals': best['signals'],
            'session': session
        }
    
    def _analyze_scalp(self, m5, m15, price, config):
        """M5+M15 scalping"""
        analysis = {
            'strategy': 'SCALP_M5_M15',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        # Check displacement
        closes = m5['close'].values[-3:]
        movement = abs(closes[-1] - closes[0])
        pip_move = movement / config['pip']
        
        if pip_move >= 8:
            analysis['confidence'] += 30
            analysis['direction'] = 'BUY' if closes[-1] > closes[0] else 'SELL'
            analysis['signals'].append(f"Displacement: {pip_move:.1f}p")
        
        # M15 alignment
        if len(m15) > 10:
            m15_sma = m15['close'].rolling(10).mean().iloc[-1]
            m15_price = m15['close'].iloc[-1]
            
            if analysis['direction'] == 'BUY' and m15_price > m15_sma:
                analysis['confidence'] += 20
                analysis['signals'].append("M15 aligned")
            elif analysis['direction'] == 'SELL' and m15_price < m15_sma:
                analysis['confidence'] += 20
                analysis['signals'].append("M15 aligned")
        
        # Order block
        for i in range(len(m5)-10, len(m5)-1):
            if i < 0:
                continue
            body = abs(m5['close'].iloc[i] - m5['open'].iloc[i])
            total = m5['high'].iloc[i] - m5['low'].iloc[i]
            if total > 0 and (body/total) > 0.7:
                analysis['confidence'] += 15
                analysis['signals'].append("Order Block")
                break
        
        return analysis
    
    def _analyze_intraday(self, m15, h1, price, config):
        """M15+H1 intraday"""
        analysis = {
            'strategy': 'INTRADAY_M15_H1',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        if m15 is None or len(m15) < 30:
            return analysis
        
        # Displacement
        closes = m15['close'].values[-3:]
        movement = abs(closes[-1] - closes[0])
        pip_move = movement / config['pip']
        
        if pip_move >= 15:
            analysis['confidence'] += 35
            analysis['direction'] = 'BUY' if closes[-1] > closes[0] else 'SELL'
            analysis['signals'].append(f"M15 Displacement: {pip_move:.1f}p")
        
        # H1 alignment
        if h1 is not None and len(h1) > 20:
            h1_sma = h1['close'].rolling(10).mean().iloc[-1]
            h1_price = h1['close'].iloc[-1]
            
            if analysis['direction'] == 'BUY' and h1_price > h1_sma:
                analysis['confidence'] += 30
                analysis['signals'].append("H1 aligned")
            elif analysis['direction'] == 'SELL' and h1_price < h1_sma:
                analysis['confidence'] += 30
                analysis['signals'].append("H1 aligned")
        
        return analysis
    
    def _analyze_swing(self, h1, h4, price, config):
        """H1+H4 swing"""
        analysis = {
            'strategy': 'SWING_H1_H4',
            'confidence': 0,
            'direction': 'NONE',
            'signals': []
        }
        
        if h1 is None or len(h1) < 40:
            return analysis
        
        # Displacement
        closes = h1['close'].values[-3:]
        movement = abs(closes[-1] - closes[0])
        pip_move = movement / config['pip']
        
        if pip_move >= 25:
            analysis['confidence'] += 40
            analysis['direction'] = 'BUY' if closes[-1] > closes[0] else 'SELL'
            analysis['signals'].append(f"H1 Displacement: {pip_move:.1f}p")
        
        # H4 alignment
        if h4 is not None and len(h4) > 30:
            h4_sma = h4['close'].rolling(20).mean().iloc[-1]
            h4_price = h4['close'].iloc[-1]
            
            if analysis['direction'] == 'BUY' and h4_price > h4_sma:
                analysis['confidence'] += 35
                analysis['signals'].append("H4 aligned")
            elif analysis['direction'] == 'SELL' and h4_price < h4_sma:
                analysis['confidence'] += 35
                analysis['signals'].append("H4 aligned")
        
        return analysis
    
    def _get_session(self):
        """Get current session"""
        hour = datetime.utcnow().hour
        
        if 13 <= hour < 17:
            return "Overlap"
        elif 0 <= hour < 9:
            return "Asian"
        elif 8 <= hour < 17:
            return "London"
        elif 13 <= hour < 22:
            return "NewYork"
        else:
            return "Asian"

# ============ TRADING ENGINE ============
class TradingEngine:
    def __init__(self):
        self.running = False
        self.settings = {
            'demo_mode': True,
            'dry_run': True,
            'enabled_symbols': list(MARKETS.keys()),
            'min_confidence': 65,
            'lot_size': 0.1,
            'max_concurrent': 5,
            'max_daily': 50,
            'max_hourly': 20
        }
        self.trades = []
        self.active_trades = []
        self.stats = {
            'daily_trades': 0,
            'hourly_trades': 0
        }
        self.analyzer = SMCAnalyzer()
        
    def start(self):
        """Start trading"""
        if not ctrader_api.connected:
            return False, "Not connected"
        
        self.running = True
        threading.Thread(target=self._trading_loop, daemon=True).start()
        return True, "Trading started"
    
    def stop(self):
        """Stop trading"""
        self.running = False
        return True, "Trading stopped"
    
    def _trading_loop(self):
        """Main trading loop"""
        while self.running:
            try:
                # Check limits
                if len(self.active_trades) >= self.settings['max_concurrent']:
                    time.sleep(5)
                    continue
                
                if self.stats['hourly_trades'] >= self.settings['max_hourly']:
                    time.sleep(60)
                    continue
                
                if self.stats['daily_trades'] >= self.settings['max_daily']:
                    time.sleep(300)
                    continue
                
                # Analyze markets
                for symbol in self.settings['enabled_symbols']:
                    if not self.running:
                        break
                    
                    config = MARKETS.get(symbol)
                    if not config:
                        continue
                    
                    analysis = self.analyzer.analyze(symbol, config)
                    
                    if analysis and analysis['confidence'] >= self.settings['min_confidence']:
                        self._execute_trade(analysis)
                        time.sleep(2)
                
                time.sleep(5)
                
            except Exception as e:
                print(f"Trading loop error: {e}")
                time.sleep(10)
    
    def _execute_trade(self, analysis):
        """Execute a trade"""
        try:
            symbol = analysis['symbol']
            direction = analysis['direction']
            
            if self.settings['dry_run']:
                # Dry run
                trade = {
                    'ticket': f"DRY_{int(time.time())}",
                    'symbol': symbol,
                    'direction': direction,
                    'entry': analysis['price'],
                    'sl': analysis['sl'],
                    'tp': analysis['tp'],
                    'confidence': analysis['confidence'],
                    'strategy': analysis['strategy'],
                    'session': analysis['session'],
                    'timestamp': datetime.now().isoformat(),
                    'dry_run': True
                }
                
                self.active_trades.append(trade)
                self.trades.append(trade)
                self.stats['daily_trades'] += 1
                self.stats['hourly_trades'] += 1
                
                print(f"✅ DRY: {symbol} {direction} @ {analysis['price']:.5f}")
                
            else:
                # Real trade
                success, result = ctrader_api.place_order(
                    symbol,
                    direction,
                    self.settings['lot_size'],
                    analysis['sl'],
                    analysis['tp']
                )
                
                if success:
                    trade = {
                        'ticket': result,
                        'symbol': symbol,
                        'direction': direction,
                        'entry': analysis['price'],
                        'sl': analysis['sl'],
                        'tp': analysis['tp'],
                        'confidence': analysis['confidence'],
                        'strategy': analysis['strategy'],
                        'session': analysis['session'],
                        'timestamp': datetime.now().isoformat(),
                        'dry_run': False
                    }
                    
                    self.active_trades.append(trade)
                    self.trades.append(trade)
                    self.stats['daily_trades'] += 1
                    self.stats['hourly_trades'] += 1
                    
                    print(f"✅ LIVE: {symbol} {direction} @ {analysis['price']:.5f} | Ticket: {result}")
                
        except Exception as e:
            print(f"Trade execution error: {e}")

# ============ FLASK WEB APP ============
app = Flask(__name__)
ctrader_api = CTraderAPI(demo_mode=True)
trading_engine = TradingEngine()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Karanka V7 Mobile</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: Arial, sans-serif;
            background: #0a0a0a;
            color: #FFD700;
            padding: 10px;
        }
        .header {
            background: linear-gradient(135deg, #1a1a1a, #2a2a2a);
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 2px solid #333;
        }
        .header h1 {
            font-size: 20px;
            margin-bottom: 5px;
        }
        .status {
            font-size: 12px;
            color: #FFED4E;
        }
        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 15px;
            overflow-x: auto;
        }
        .tab {
            background: #1a1a1a;
            color: #FFD700;
            border: 1px solid #333;
            padding: 10px 15px;
            border-radius: 5px;
            cursor: pointer;
            white-space: nowrap;
        }
        .tab.active {
            background: #D4AF37;
            color: #0a0a0a;
        }
        .content {
            background: #1a1a1a;
            padding: 15px;
            border-radius: 10px;
            border: 2px solid #333;
            display: none;
        }
        .content.active { display: block; }
        .btn {
            background: #D4AF37;
            color: #0a0a0a;
            border: none;
            padding: 12px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            width: 100%;
            margin: 5px 0;
        }
        .btn:active { background: #B8860B; }
        .input-group {
            margin: 10px 0;
        }
        .input-group label {
            display: block;
            margin-bottom: 5px;
            font-size: 12px;
        }
        .input-group input, .input-group select {
            width: 100%;
            padding: 10px;
            background: #0a0a0a;
            border: 1px solid #333;
            color: #FFD700;
            border-radius: 5px;
        }
        .trade-card {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            border-left: 3px solid #D4AF37;
        }
        .trade-card h4 {
            margin-bottom: 5px;
            color: #FFED4E;
        }
        .trade-card p {
            font-size: 11px;
            margin: 3px 0;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }
        .stat-box {
            background: #2a2a2a;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #00FF00;
        }
        .stat-label {
            font-size: 11px;
            color: #FFED4E;
        }
        .checkbox-group {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
        }
        .checkbox-item {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        .analysis-item {
            background: #2a2a2a;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            font-size: 11px;
        }
        .session-badge {
            display: inline-block;
            background: #D4AF37;
            color: #0a0a0a;
            padding: 3px 8px;
            border-radius: 3px;
            font-size: 10px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🎯 KARANKA V7</h1>
        <div class="status" id="status">Ready</div>
    </div>
    
    <div class="tabs">
        <div class="tab active" onclick="showTab('dashboard')">📊 Dashboard</div>
        <div class="tab" onclick="showTab('connection')">🔗 Connect</div>
        <div class="tab" onclick="showTab('settings')">⚙️ Settings</div>
        <div class="tab" onclick="showTab('markets')">📈 Markets</div>
        <div class="tab" onclick="showTab('trades')">💼 Trades</div>
        <div class="tab" onclick="showTab('analysis')">🔍 Analysis</div>
    </div>
    
    <div id="dashboard" class="content active">
        <h3>Trading Stats</h3>
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-value" id="active-trades">0</div>
                <div class="stat-label">Active Trades</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="daily-trades">0</div>
                <div class="stat-label">Daily Trades</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="hourly-trades">0</div>
                <div class="stat-label">Hourly Trades</div>
            </div>
            <div class="stat-box">
                <div class="stat-value" id="session">-</div>
                <div class="stat-label">Session</div>
            </div>
        </div>
        
        <div style="margin-top: 20px;">
            <button class="btn" onclick="startTrading()">🚀 Start Trading</button>
            <button class="btn" onclick="stopTrading()">🛑 Stop Trading</button>
        </div>
    </div>
    
    <div id="connection" class="content">
        <h3>cTrader Connection</h3>
        <div class="input-group">
            <label>Mode</label>
            <select id="demo-mode">
                <option value="true">Demo Account</option>
                <option value="false">Live Account</option>
            </select>
        </div>
        <button class="btn" onclick="connect()">🔗 Connect to cTrader</button>
        <div id="conn-status" style="margin-top: 15px; font-size: 12px;"></div>
    </div>
    
    <div id="settings" class="content">
        <h3>Trading Settings</h3>
        <div class="input-group">
            <label>Mode</label>
            <select id="dry-run">
                <option value="true">Dry Run</option>
                <option value="false">Live Trading</option>
            </select>
        </div>
        <div class="input-group">
            <label>Min Confidence (%)</label>
            <input type="number" id="min-confidence" value="65">
        </div>
        <div class="input-group">
            <label>Lot Size</label>
            <input type="number" step="0.01" id="lot-size" value="0.1">
        </div>
        <div class="input-group">
            <label>Max Concurrent Trades</label>
            <input type="number" id="max-concurrent" value="5">
        </div>
        <div class="input-group">
            <label>Max Daily Trades</label>
            <input type="number" id="max-daily" value="50">
        </div>
        <div class="input-group">
            <label>Max Hourly Trades</label>
            <input type="number" id="max-hourly" value="20">
        </div>
        <button class="btn" onclick="saveSettings()">💾 Save Settings</button>
    </div>
    
    <div id="markets" class="content">
        <h3>Select Markets</h3>
        <div class="checkbox-group" id="markets-list"></div>
        <button class="btn" onclick="saveMarkets()">💾 Save Selection</button>
    </div>
    
    <div id="trades" class="content">
        <h3>Active Trades</h3>
        <div id="trades-list"></div>
    </div>
    
    <div id="analysis" class="content">
        <h3>Live Market Analysis</h3>
        <div id="analysis-list"></div>
    </div>
    
    <script>
        let currentTab = 'dashboard';
        
        function showTab(tab) {
            document.querySelectorAll('.content').forEach(c => c.classList.remove('active'));
            document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            document.getElementById(tab).classList.add('active');
            event.target.classList.add('active');
            currentTab = tab;
        }
        
        function connect() {
            const demo = document.getElementById('demo-mode').value === 'true';
            fetch('/connect', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({demo_mode: demo})
            })
            .then(r => r.json())
            .then(data => {
                document.getElementById('conn-status').innerHTML = 
                    data.success ? '✅ ' + data.message : '❌ ' + data.message;
            });
        }
        
        function saveSettings() {
            const settings = {
                dry_run: document.getElementById('dry-run').value === 'true',
                min_confidence: parseInt(document.getElementById('min-confidence').value),
                lot_size: parseFloat(document.getElementById('lot-size').value),
                max_concurrent: parseInt(document.getElementById('max-concurrent').value),
                max_daily: parseInt(document.getElementById('max-daily').value),
                max_hourly: parseInt(document.getElementById('max-hourly').value)
            };
            
            fetch('/settings', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(settings)
            })
            .then(r => r.json())
            .then(data => alert(data.message));
        }
        
        function saveMarkets() {
            const selected = [];
            document.querySelectorAll('#markets-list input:checked').forEach(cb => {
                selected.push(cb.value);
            });
            
            fetch('/markets', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({symbols: selected})
            })
            .then(r => r.json())
            .then(data => alert(data.message));
        }
        
        function startTrading() {
            fetch('/start', {method: 'POST'})
            .then(r => r.json())
            .then(data => alert(data.message));
        }
        
        function stopTrading() {
            fetch('/stop', {method: 'POST'})
            .then(r => r.json())
            .then(data => alert(data.message));
        }
        
        function updateData() {
            fetch('/status')
            .then(r => r.json())
            .then(data => {
                document.getElementById('status').textContent = 
                    data.connected ? `✅ ${data.session} | Trading: ${data.running ? 'Active' : 'Stopped'}` : '❌ Disconnected';
                document.getElementById('active-trades').textContent = data.active_trades;
                document.getElementById('daily-trades').textContent = data.daily_trades;
                document.getElementById('hourly-trades').textContent = data.hourly_trades;
                document.getElementById('session').textContent = data.session;
                
                // Update trades
                const tradesList = document.getElementById('trades-list');
                if (data.trades && data.trades.length > 0) {
                    tradesList.innerHTML = data.trades.map(t => `
                        <div class="trade-card">
                            <h4>${t.symbol} ${t.direction} ${t.dry_run ? '(DRY)' : '(LIVE)'}</h4>
                            <p>Entry: ${t.entry.toFixed(5)}</p>
                            <p>SL: ${t.sl.toFixed(5)} | TP: ${t.tp.toFixed(5)}</p>
                            <p>Confidence: ${t.confidence.toFixed(1)}%</p>
                            <p>Strategy: ${t.strategy}</p>
                            <p><span class="session-badge">${t.session}</span></p>
                        </div>
                    `).join('');
                } else {
                    tradesList.innerHTML = '<p>No active trades</p>';
                }
            });
        }
        
        // Initialize markets
        fetch('/markets')
        .then(r => r.json())
        .then(data => {
            const list = document.getElementById('markets-list');
            list.innerHTML = data.all.map(symbol => `
                <div class="checkbox-item">
                    <input type="checkbox" value="${symbol}" 
                           ${data.enabled.includes(symbol) ? 'checked' : ''}>
                    <label>${symbol}</label>
                </div>
            `).join('');
        });
        
        // Auto-update every 2 seconds
        setInterval(updateData, 2000);
        updateData();
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/connect', methods=['POST'])
def connect():
    data = request.json
    demo = data.get('demo_mode', True)
    
    global ctrader_api
    ctrader_api = CTraderAPI(demo_mode=demo)
    success, message = ctrader_api.connect()
    
    if success:
        accounts = ctrader_api.get_accounts()
        if accounts:
            message += f" | Account: {accounts[0].get('accountNumber', 'Unknown')}"
    
    return jsonify({'success': success, 'message': message})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if request.method == 'POST':
        data = request.json
        trading_engine.settings.update(data)
        return jsonify({'success': True, 'message': 'Settings saved'})
    return jsonify(trading_engine.settings)

@app.route('/markets', methods=['GET', 'POST'])
def markets():
    if request.method == 'POST':
        data = request.json
        trading_engine.settings['enabled_symbols'] = data.get('symbols', [])
        return jsonify({'success': True, 'message': f"{len(data.get('symbols', []))} markets selected"})
    
    return jsonify({
        'all': list(MARKETS.keys()),
        'enabled': trading_engine.settings['enabled_symbols']
    })

@app.route('/start', methods=['POST'])
def start():
    success, message = trading_engine.start()
    return jsonify({'success': success, 'message': message})

@app.route('/stop', methods=['POST'])
def stop():
    success, message = trading_engine.stop()
    return jsonify({'success': success, 'message': message})

@app.route('/status')
def status():
    session = trading_engine.analyzer._get_session() if ctrader_api.connected else '-'
    
    return jsonify({
        'connected': ctrader_api.connected,
        'running': trading_engine.running,
        'session': session,
        'active_trades': len(trading_engine.active_trades),
        'daily_trades': trading_engine.stats['daily_trades'],
        'hourly_trades': trading_engine.stats['hourly_trades'],
        'trades': trading_engine.active_trades[-10:]
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
