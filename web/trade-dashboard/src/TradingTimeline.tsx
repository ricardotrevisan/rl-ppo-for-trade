import React, { useMemo, useState } from 'react'
import { ResponsiveContainer, AreaChart, Area, CartesianGrid, XAxis, YAxis, Tooltip, Legend } from 'recharts'

type TradeRow = {
  timestamp: Date
  side: 'buy' | 'sell'
  qty: number
  price: number
  fees: number
  equity: number
  position: number
  gainLoss?: number
  gainLossPct?: number
  result?: 'gain' | 'loss' | 'flat'
  cumReturnPct?: number   // ✅ acumulado até o ponto

}

const TradingTimeline: React.FC = () => {
  const [view, setView] = useState<'equity' | 'position' | 'price'>('equity')
  const [dateRange, setDateRange] = useState<'all' | '7d' | '30d' | '90d'>('all')
  const [trades, setTrades] = useState<TradeRow[]>([])
  const [windowStart, setWindowStart] = useState<number>(0)
  // Pagination for filtered table (100 per page)
  const [page, setPage] = useState<number>(1)
  const pageSize = 100

  const parseTrades = (csvText: string) => {
    const lines = csvText.trim().split(/\r?\n/)
    const out: TradeRow[] = []
    let position = 0
    let totalCost = 0

    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(',')
      if (cols.length < 9) continue
      const ts = new Date(cols[0])
      const side = cols[2] as 'buy' | 'sell'
      const qty = parseInt(cols[3] || '0', 10)
      const price = parseFloat(cols[4] || '0')
      const fees = parseFloat(cols[5] || '0')
      const equity = parseFloat(cols[7] || '0')
      const positionQty = parseInt(cols[8] || '0', 10)

      const trade: TradeRow = {
        timestamp: ts,
        side,
        qty,
        price,
        fees,
        equity,
        position: positionQty,
      }

      // === cálculo de preço médio com segurança ===
      if (side === 'buy' && qty > 0) {
        totalCost += price * qty
        position += qty
      } else if (side === 'sell' && position > 0) {
        const avgPrice = totalCost / position
        const gainLoss = price - avgPrice
        trade.gainLoss = gainLoss
        trade.gainLossPct = (gainLoss / avgPrice) * 100
        trade.result = gainLoss > 0 ? 'gain' : gainLoss < 0 ? 'loss' : 'flat'
        // zera posição total
        totalCost = 0
        position = 0
      }

      out.push(trade)
    }

      setTrades(() => {
    // --- calcula PnL acumulado ---
    let initialEquity = out[0]?.equity || 100000
    let lastEquity = initialEquity

    const withCumulative = out.map((t) => {
      if (!t.equity || isNaN(t.equity)) {
        t.cumReturnPct = ((lastEquity - initialEquity) / initialEquity) * 100
        return t
      }
      lastEquity = t.equity
      t.cumReturnPct = ((t.equity - initialEquity) / initialEquity) * 100
      return t
    })

    return withCumulative
  })

  }


  const handleFileUpload: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const f = e.target.files?.[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = (ev) => {
      const text = String(ev.target?.result || '')
      parseTrades(text)
    }
    reader.readAsText(f)
  }

  const filteredTrades = useMemo(() => {
    if (trades.length === 0) return [] as TradeRow[]
    if (dateRange === 'all') return trades
    const startIndex = Math.max(0, Math.min(windowStart, trades.length - 1))
    const startDate = trades[startIndex].timestamp
    let daysToAdd = 7
    if (dateRange === '30d') daysToAdd = 30
    else if (dateRange === '90d') daysToAdd = 90
    const endDate = new Date(startDate)
    endDate.setDate(endDate.getDate() + daysToAdd)
    const windowRows: TradeRow[] = []
    for (let i = startIndex; i < trades.length; i++) {
      if (trades[i].timestamp <= endDate) windowRows.push(trades[i])
      else break
    }
    return windowRows.length > 0 ? windowRows : [trades[startIndex]]
  }, [trades, dateRange, windowStart])

  // Reset pagination when filter changes
  React.useEffect(() => { setPage(1) }, [dateRange, windowStart, trades])

  // Descending list within the filtered window (most recent first)
  const filteredDesc = useMemo(() => {
    return [...filteredTrades].reverse()
  }, [filteredTrades])

  // Current page slice
  const totalItems = filteredDesc.length
  const totalPages = Math.max(1, Math.ceil(totalItems / pageSize))
  const startIdx = (page - 1) * pageSize
  const pagedTrades = filteredDesc.slice(startIdx, startIdx + pageSize)

  const stats = useMemo(() => {
    if (trades.length === 0) return null as null | Record<string, any>
    const initial = 100000
    const final = trades[trades.length - 1].equity
    const totalTrades = trades.length
    const buyTrades = trades.filter(t => t.side === 'buy').length
    const sellTrades = trades.filter(t => t.side === 'sell').length
    const maxPosition = Math.max(...trades.map(t => t.position))
    const completeCycles = trades.filter(t => t.position === 0).length
    const totalFees = trades.reduce((s, t) => s + (t.fees || 0), 0)
    const startDate = trades[0].timestamp
    const endDate = trades[trades.length - 1].timestamp
    const daysDiff = Math.max(0, Math.floor((+endDate - +startDate) / (1000 * 60 * 60 * 24)))
    return {
      initial,
      final,
      returnPct: ((final - initial) / initial * 100),
      totalTrades,
      buyTrades,
      sellTrades,
      maxPosition,
      completeCycles,
      totalFees,
      daysDiff,
      startDate,
      endDate,
    }
  }, [trades])

 const CustomTooltip = ({ active, payload }: any) => {
  if (active && payload && payload.length) {
    const d: TradeRow = payload[0].payload
    return (
      <div
        style={{
          background: '#fff',
          padding: 12,
          border: '1px solid #e5e7eb',
          borderRadius: 6,
          boxShadow: '0 1px 4px rgba(0,0,0,0.1)',
        }}
      >
        <p style={{ fontWeight: 600 }}>{d.timestamp.toLocaleString('pt-BR')}</p>

        <p
          style={{
            color: d.side === 'buy' ? '#059669' : '#dc2626',
            fontWeight: 600,
          }}
        >
          {d.side === 'buy' ? '🟢 COMPRA' : '🔴 VENDA'}: {d.qty} ações
        </p>

        <p>Preço: R$ {d.price.toFixed(2)}</p>
        <p>Taxa: R$ {d.fees.toFixed(2)}</p>
        <p style={{ fontWeight: 600 }}>
          Equity: R$ {d.equity.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}
        </p>
        <p>Posição: {d.position} ações</p>

        {/* Resultado de venda (gain/loss) */}
        {d.side === 'sell' && d.result && (
          <p
            style={{
              fontWeight: 700,
              color:
                d.result === 'gain'
                  ? '#16a34a' // verde
                  : d.result === 'loss'
                  ? '#dc2626' // vermelho
                  : '#6b7280', // neutro
            }}
          >
            Resultado:{' '}
            {d.result === 'gain'
              ? `💰 GAIN (+${d.gainLossPct?.toFixed(2)}%)`
              : d.result === 'loss'
              ? `🔻 LOSS (${d.gainLossPct?.toFixed(2)}%)`
              : '⚪ FLAT'}
          </p>
        )}
      </div>
    )
  }
  return null
}

  if (trades.length === 0) {
    return (
      <div style={{ width: '100%', height: '100%', background: '#f9fafb', padding: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
        <div style={{ background: '#fff', padding: 24, borderRadius: 12, boxShadow: '0 1px 10px rgba(0,0,0,0.06)', maxWidth: 560, width: '100%' }}>
          <h2 style={{ fontSize: 22, fontWeight: 700, marginBottom: 12, color: '#111827' }}>Carregar Arquivo de Trades</h2>
          <p style={{ color: '#4b5563', marginBottom: 16 }}>
            Faça upload do arquivo CSV com o histórico de trades exportado pelo backtester.
          </p>
          <input type="file" accept=".txt,.csv" onChange={handleFileUpload} />
          <p style={{ color: '#6b7280', fontSize: 12, marginTop: 8 }}>
            Formato: timestamp,symbol,side,qty,price,fees,equity_before,equity_after,position_qty_after
          </p>
        </div>
      </div>
    )
  }

  return (
    <div style={{ width: '100%', height: '100%', background: '#f9fafb', padding: 24, overflowY: 'auto' }}>
      <div style={{ maxWidth: 1200, margin: '0 auto' }}>
        <h1 style={{ fontSize: 28, fontWeight: 800, marginBottom: 6, color: '#111827' }}>Timeline Completa - Trading</h1>
        <p style={{ color: '#6b7280', marginBottom: 24 }}>
          Período: {stats!.startDate.toLocaleDateString('pt-BR')} a {stats!.endDate.toLocaleDateString('pt-BR')} ({stats!.daysDiff} dias)
        </p>

        {/* Stats */}
        <div style={{ display: 'grid', gap: 16, gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', marginBottom: 24 }}>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Retorno Total</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: stats!.returnPct >= 0 ? '#059669' : '#dc2626' }}>
              {stats!.returnPct.toFixed(2)}%
            </div>
          </div>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Equity Final</div>
            <div style={{ fontSize: 20, fontWeight: 800, color: '#111827' }}>
              R$ {stats!.final.toLocaleString('pt-BR', { maximumFractionDigits: 0 })}
            </div>
          </div>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Total Trades</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#2563eb' }}>{stats!.totalTrades}</div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>
              {stats!.buyTrades} compras / {stats!.sellTrades} vendas
            </div>
          </div>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Ciclos Completos</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#7c3aed' }}>{stats!.completeCycles}</div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>zerou posição</div>
          </div>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Posição Máxima</div>
            <div style={{ fontSize: 24, fontWeight: 800, color: '#ea580c' }}>{stats!.maxPosition}</div>
            <div style={{ fontSize: 12, color: '#6b7280', marginTop: 4 }}>ações</div>
          </div>
          <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
            <div style={{ fontSize: 12, color: '#6b7280' }}>Taxas Totais</div>
            <div style={{ fontSize: 20, fontWeight: 800, color: '#dc2626' }}>R$ {stats!.totalFees.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}</div>
          </div>
        </div>

        Controls
        <div style={{ background: '#fff', padding: 16, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)', marginBottom: 24 }}>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            <div>
              <div style={{ fontSize: 12, color: '#374151', marginBottom: 8 }}>Visualização</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => setView('equity')} style={{ padding: '8px 12px', borderRadius: 8, background: view === 'equity' ? '#2563eb' : '#e5e7eb', color: view === 'equity' ? '#fff' : '#111827' }}>Equity</button>
                <button onClick={() => setView('position')} style={{ padding: '8px 12px', borderRadius: 8, background: view === 'position' ? '#2563eb' : '#e5e7eb', color: view === 'position' ? '#fff' : '#111827' }}>Posição</button>
                <button onClick={() => setView('price')} style={{ padding: '8px 12px', borderRadius: 8, background: view === 'price' ? '#2563eb' : '#e5e7eb', color: view === 'price' ? '#fff' : '#111827' }}>Preço</button>
              </div>
            </div>
            <div>
              <div style={{ fontSize: 12, color: '#374151', marginBottom: 8 }}>Janela de Tempo</div>
              <div style={{ display: 'flex', gap: 8 }}>
                <button onClick={() => { setDateRange('7d'); setWindowStart(0) }} style={{ padding: '6px 10px', borderRadius: 8, background: dateRange === '7d' ? '#059669' : '#e5e7eb', color: dateRange === '7d' ? '#fff' : '#111827' }}>7 dias</button>
                <button onClick={() => { setDateRange('30d'); setWindowStart(0) }} style={{ padding: '6px 10px', borderRadius: 8, background: dateRange === '30d' ? '#059669' : '#e5e7eb', color: dateRange === '30d' ? '#fff' : '#111827' }}>30 dias</button>
                <button onClick={() => { setDateRange('90d'); setWindowStart(0) }} style={{ padding: '6px 10px', borderRadius: 8, background: dateRange === '90d' ? '#059669' : '#e5e7eb', color: dateRange === '90d' ? '#fff' : '#111827' }}>90 dias</button>
                <button onClick={() => { setDateRange('all'); setWindowStart(0) }} style={{ padding: '6px 10px', borderRadius: 8, background: dateRange === 'all' ? '#059669' : '#e5e7eb', color: dateRange === 'all' ? '#fff' : '#111827' }}>Tudo ({stats!.totalTrades} trades)</button>
              </div>
            </div>
          </div>
          {dateRange !== 'all' && (
            <div style={{ marginTop: 16, paddingTop: 16, borderTop: '1px solid #e5e7eb' }}>
              <div style={{ fontSize: 12, color: '#374151', marginBottom: 8 }}>
                Navegar no Período:
                <span style={{ marginLeft: 8, color: '#2563eb', fontWeight: 600 }}>
                  {filteredTrades.length > 0 && filteredTrades[0].timestamp.toLocaleDateString('pt-BR')} → {filteredTrades.length > 0 && filteredTrades[filteredTrades.length - 1].timestamp.toLocaleDateString('pt-BR')}
                </span>
              </div>
              <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                <button onClick={() => setWindowStart(Math.max(0, windowStart - 20))} disabled={windowStart === 0} style={{ padding: '6px 10px', borderRadius: 8, background: '#2563eb', color: '#fff', opacity: windowStart === 0 ? 0.5 : 1 }}>← Anterior</button>
                <input type="range" min={0} max={Math.max(0, trades.length - 1)} value={windowStart} onChange={(e) => setWindowStart(parseInt(e.target.value))} style={{ flex: 1 }} />
                <button onClick={() => setWindowStart(Math.min(trades.length - 1, windowStart + 20))} disabled={windowStart >= trades.length - 1} style={{ padding: '6px 10px', borderRadius: 8, background: '#2563eb', color: '#fff', opacity: windowStart >= trades.length - 1 ? 0.5 : 1 }}>Próximo →</button>
              </div>
              <div style={{ fontSize: 12, color: '#6b7280', marginTop: 8, textAlign: 'center' }}>
                Janela de {dateRange === '7d' ? '7' : dateRange === '30d' ? '30' : '90'} dias | Trade inicial: {windowStart + 1} de {trades.length} | Mostrando {filteredTrades.length} trades
              </div>
            </div>
          )}
        </div>

        {/* Chart */}
        <div style={{ background: '#fff', padding: 24, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)' }}>
          <ResponsiveContainer width="100%" height={400}>
            <AreaChart data={filteredTrades}>
              <defs>
                <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPosition" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#8b5cf6" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="colorPrice" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.8} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" tickFormatter={(ts: Date) => new Date(ts).toLocaleDateString('pt-BR', { day: '2-digit', month: '2-digit' })} />
              <YAxis />
              <Tooltip content={<CustomTooltip />} />
              <Legend />
              {view === 'equity' && (
                <Area type="monotone" dataKey="equity" stroke="#3b82f6" fillOpacity={1} fill="url(#colorEquity)" name="Equity (R$)" />
              )}
              {view === 'position' && (
                <Area type="stepAfter" dataKey="position" stroke="#8b5cf6" fillOpacity={1} fill="url(#colorPosition)" name="Posição (ações)" />
              )}
              {view === 'price' && (
                <Area type="monotone" dataKey="price" stroke="#10b981" fillOpacity={1} fill="url(#colorPrice)" name="Preço (R$)" />
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Recent Trades (filtered + paginated) */}
        <div style={{ background: '#fff', padding: 24, borderRadius: 12, boxShadow: '0 1px 6px rgba(0,0,0,0.05)', marginTop: 24 }}>
          <h2 style={{ fontSize: 20, fontWeight: 800, marginBottom: 8 }}>Trades do Período</h2>
          <div style={{ fontSize: 12, color: '#6b7280', marginBottom: 12 }}>
            Mostrando {pagedTrades.length} de {totalItems} trades filtrados • Página {page} de {totalPages} • Até {pageSize} por página
          </div>
          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', fontSize: 14 }}>
              <thead>
                <tr style={{ borderBottom: '1px solid #e5e7eb' }}>
                  <th style={{ textAlign: 'left', padding: '8px 12px' }}>Data/Hora</th>
                  <th style={{ textAlign: 'left', padding: '8px 12px' }}>Tipo</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Qtd</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Preço</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Taxa</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Equity</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Posição</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Resultado</th>
                  <th style={{ textAlign: 'right', padding: '8px 12px' }}>Acum.</th>
                </tr>
              </thead>
              <tbody>
                {pagedTrades.map((t, i) => (
                  <tr key={i} style={{ borderBottom: '1px solid #f3f4f6' }}>
                    <td style={{ padding: '8px 12px' }}>
                      {t.timestamp.toLocaleString('pt-BR', {
                        day: '2-digit',
                        month: '2-digit',
                        year: 'numeric',
                        hour: '2-digit',
                        minute: '2-digit',
                      })}
                    </td>

                    <td style={{ padding: '8px 12px' }}>
                      <span
                        style={{
                          padding: '2px 6px',
                          borderRadius: 6,
                          fontSize: 12,
                          fontWeight: 700,
                          background: t.side === 'buy' ? '#dcfce7' : '#fee2e2',
                          color: t.side === 'buy' ? '#166534' : '#b91c1c',
                        }}
                      >
                        {t.side === 'buy' ? '🟢 COMPRA' : '🔴 VENDA'}
                      </span>
                    </td>

                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>{t.qty}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right' }}>R$ {t.price.toFixed(2)}</td>
                    <td style={{ padding: '8px 12px', textAlign: 'right', color: '#dc2626' }}>
                      R$ {t.fees.toFixed(2)}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 700 }}>
                      R$ {t.equity.toLocaleString('pt-BR', { maximumFractionDigits: 2 })}
                    </td>
                    <td style={{ padding: '8px 12px', textAlign: 'right', fontWeight: 700 }}>
                      {t.position}
                    </td>

                    {/* Coluna de resultado (só aparece para VENDA) */}
                    <td
                      style={{
                        padding: '8px 12px',
                        textAlign: 'right',
                        fontWeight: 700,
                        color:
                          t.side === 'sell'
                            ? t.result === 'gain'
                              ? '#16a34a'
                              : t.result === 'loss'
                              ? '#dc2626'
                              : '#6b7280'
                            : '#9ca3af',
                      }}
                    >
                      {t.side === 'sell'
                        ? t.result === 'gain'
                          ? `💰 +${t.gainLossPct?.toFixed(2)}%`
                          : t.result === 'loss'
                          ? `🔻 ${t.gainLossPct?.toFixed(2)}%`
                          : '⚪ 0.00%'
                        : '—'}
                    </td>

                    {/* Coluna de Retorno Acumulado */}
                    <td style={{
                      padding: '8px 12px',
                      textAlign: 'right',
                      fontWeight: 700,
                      color: t.cumReturnPct && t.cumReturnPct >= 0 ? '#059669' : '#dc2626'
                    }}>
                      {t.cumReturnPct?.toFixed(2)}%
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Pagination controls */}
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', justifyContent: 'center', marginTop: 12 }}>
            <button onClick={() => setPage(1)} disabled={page <= 1} style={{ padding: '6px 10px', borderRadius: 8, background: '#e5e7eb', color: '#111827', opacity: page <= 1 ? 0.5 : 1 }}>« Primeiro</button>
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1} style={{ padding: '6px 10px', borderRadius: 8, background: '#2563eb', color: '#fff', opacity: page <= 1 ? 0.5 : 1 }}>← Anterior</button>
            <span style={{ fontSize: 12, color: '#6b7280' }}>Página {page} / {totalPages}</span>
            <button onClick={() => setPage(p => Math.min(totalPages, p + 1))} disabled={page >= totalPages} style={{ padding: '6px 10px', borderRadius: 8, background: '#2563eb', color: '#fff', opacity: page >= totalPages ? 0.5 : 1 }}>Próximo →</button>
            <button onClick={() => setPage(totalPages)} disabled={page >= totalPages} style={{ padding: '6px 10px', borderRadius: 8, background: '#e5e7eb', color: '#111827', opacity: page >= totalPages ? 0.5 : 1 }}>Último »</button>
          </div>
        </div>
      </div>
    </div>
  )
}

export default TradingTimeline
