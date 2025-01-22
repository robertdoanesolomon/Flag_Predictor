# Quick Start - Production Web App

## Test Locally First

### 1. Install dependencies
```bash
pip install flask schedule
```

### 2. Start the scheduler (Terminal 1)
This will update forecasts every 6 hours:
```bash
python run_scheduler.py
```

### 3. Start the web app (Terminal 2)
```bash
python app_production.py
```

### 4. Visit the website
Open: http://localhost:5001

The first time, it will take a few minutes to generate forecasts. After that, the website loads instantly from cache.

## How It Works

- **Scheduler** (`run_scheduler.py`): Runs forecasts every 6 hours using 2026_01 models
- **Web App** (`app_production.py`): Serves cached forecasts quickly
- **Cache**: Stored in `forecast_cache/` directory

## Manual Update

If you want to update forecasts manually:
1. Visit: http://localhost:5001/admin/update
2. Or run: `python forecast_scheduler.py`

## Production Deployment

See `DEPLOYMENT.md` for full deployment instructions including:
- VPS setup with systemd
- Docker deployment
- Cloud platform options (Heroku, Railway, etc.)
