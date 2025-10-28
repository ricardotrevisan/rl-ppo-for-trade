Trade Dashboard (CSV Viewer)

What it is
- A minimal React + Vite app to visualize backtester trade CSVs with an interactive timeline, quick stats, and a recent trades table.

Usage
- Install once:
  - cd web/trade-dashboard
  - npm install
- Run locally:
  - npm run dev
  - Open the printed localhost URL (e.g., http://localhost:5173)
- Upload a CSV exported by the backtester (see scripts/run_backtest.py with --save_trades_csv).
  - Expected header:
    - timestamp,symbol,side,qty,price,fees,equity_before,equity_after,position_qty_after

Notes
- No server required; this is a static app. Use npm run build to produce a dist/ folder you can serve from any static host.
- MCP integration is intentionally out-of-scope for now; this app reads local CSVs only.

Files
- src/TradingTimeline.tsx — main component with upload, timeline, and tables.
- src/App.tsx — wrapper.
- index.html, vite.config.ts — Vite scaffold.

///
MCP Inspector:
```bash
npx @modelcontextprotocol/inspector connect http://localhost:8000/mcp
```