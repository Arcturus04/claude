"""
+=====================================================================+
|  NEXUS*AI - Igon Value Screener                                     |
|  William Igon's value investment methodology                        |
|  Extracted from interview: PE, ROE, revenue growth screening        |
|  with position sizing, profit-taking, and loss-cutting rules        |
+=====================================================================+

William Igon - nicknamed "the French Warren Buffett" - has delivered
4000%+ returns (12.8% annualized net of fees) since 1993 on European
small and mid caps using a rigorous, repeatable value system.

Core rules:
  BUY:   PE < 12, ROE > 10%, revenue growing YoY
  HOLD:  PE 12-15
  SELL:  Start reducing at PE 15 (sell 20% per 1pt PE increase)
  EXIT:  Full exit at PE >= 20
  CLEAN: Every 6 months, cut the biggest portfolio losers
"""

import logging
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


# ---- Igon Valuation Thresholds ----

PE_BUY_MAX = 12          # Only buy below this PE
PE_SELL_START = 15        # Start taking profits here
PE_FULL_EXIT = 20         # Full exit above this
PE_SELL_STEP_PCT = 0.20   # Sell 20% of position per 1pt PE above 15

ROE_MIN = 10.0            # Minimum return on equity (%)

WCR_EXCELLENT = 20.0      # WCR/turnover below this = excellent positioning
WCR_NORMAL = 25.0         # WCR/turnover below this = acceptable
# Above 25% = warning, company has weak negotiating power

TARGET_LINES = 33         # Target number of portfolio lines
WEIGHT_PER_LINE = 3.0     # Target weight per line (%)
BUILD_TRANCHES = 10       # Build position in 10 tranches
MAX_VOLUME_PCT = 0.20     # Max 20% of daily volume to avoid price impact

LOSS_REVIEW_MONTHS = 6    # Review and cut losers every 6 months


def score_stock(pe: float, roe: float, revenue_growth: float,
                wcr_pct: Optional[float] = None) -> Dict:
    """
    Score a stock using Igon's value criteria.

    Args:
        pe: Price-to-earnings ratio (trailing)
        roe: Return on equity (%)
        revenue_growth: Year-over-year revenue growth (%)
        wcr_pct: Working capital requirement as % of turnover (optional)

    Returns:
        Dict with signal, score, breakdown, and reasoning
    """
    score = 0.0
    reasons = []
    warnings = []

    # ---- PE Analysis (most important) ----
    if pe <= 0:
        score -= 30
        warnings.append(f"Negative or zero PE ({pe:.1f}) - company may be in loss")
    elif pe <= 7:
        score += 40
        reasons.append(f"PE {pe:.1f} - extremely attractive (Igon sweet spot)")
    elif pe <= 8:
        score += 35
        reasons.append(f"PE {pe:.1f} - very attractive valuation")
    elif pe <= 12:
        score += 25
        reasons.append(f"PE {pe:.1f} - within Igon buy zone (< 12)")
    elif pe <= 15:
        score += 5
        reasons.append(f"PE {pe:.1f} - hold zone, no new buying")
    elif pe <= 20:
        score -= 15
        # Calculate how much to reduce
        points_above_15 = pe - PE_SELL_START
        reduce_pct = min(points_above_15 * PE_SELL_STEP_PCT * 100, 100)
        reasons.append(
            f"PE {pe:.1f} - profit-taking zone, reduce by ~{reduce_pct:.0f}%"
        )
    else:
        score -= 30
        reasons.append(f"PE {pe:.1f} - full exit zone (> 20)")

    # ---- ROE Analysis ----
    if roe >= 17:
        score += 25
        reasons.append(f"ROE {roe:.1f}% - exceptional profitability")
    elif roe >= 13:
        score += 20
        reasons.append(f"ROE {roe:.1f}% - strong profitability")
    elif roe >= ROE_MIN:
        score += 15
        reasons.append(f"ROE {roe:.1f}% - meets minimum threshold (> 10%)")
    else:
        score -= 20
        warnings.append(f"ROE {roe:.1f}% - below Igon minimum of 10%")

    # ---- Revenue Growth ----
    if revenue_growth > 10:
        score += 15
        reasons.append(f"Revenue growth {revenue_growth:.1f}% - strong expansion")
    elif revenue_growth > 0:
        score += 10
        reasons.append(f"Revenue growth {revenue_growth:.1f}% - positive, meets criteria")
    elif revenue_growth > -5:
        score += 0
        warnings.append(f"Revenue growth {revenue_growth:.1f}% - stagnating")
    else:
        score -= 15
        warnings.append(f"Revenue growth {revenue_growth:.1f}% - declining, fails criteria")

    # ---- WCR Analysis (optional but valuable) ----
    wcr_analysis = None
    if wcr_pct is not None:
        if wcr_pct < WCR_EXCELLENT:
            score += 10
            wcr_analysis = f"WCR {wcr_pct:.1f}% of turnover - excellent positioning"
            reasons.append(wcr_analysis)
        elif wcr_pct < WCR_NORMAL:
            score += 5
            wcr_analysis = f"WCR {wcr_pct:.1f}% of turnover - normal range"
            reasons.append(wcr_analysis)
        else:
            score -= 5
            wcr_analysis = f"WCR {wcr_pct:.1f}% of turnover - weak negotiating power"
            warnings.append(wcr_analysis)

    # ---- Determine Signal ----
    passes_screen = pe > 0 and pe <= PE_BUY_MAX and roe >= ROE_MIN and revenue_growth > 0

    if passes_screen and score >= 40:
        signal = "STRONG_BUY"
    elif passes_screen:
        signal = "BUY"
    elif pe > 0 and pe <= PE_SELL_START:
        signal = "HOLD"
    elif pe > PE_FULL_EXIT or pe <= 0:
        signal = "SELL"
    else:
        signal = "REDUCE"

    # Normalize score to 0-100
    normalized_score = max(0, min(100, score + 50))

    return {
        "signal": signal,
        "score": round(normalized_score, 1),
        "pe": round(pe, 2),
        "roe": round(roe, 2),
        "revenue_growth": round(revenue_growth, 2),
        "wcr_pct": round(wcr_pct, 2) if wcr_pct is not None else None,
        "passes_igon_screen": passes_screen,
        "reasons": reasons,
        "warnings": warnings,
        "criteria": {
            "pe_ok": 0 < pe <= PE_BUY_MAX,
            "roe_ok": roe >= ROE_MIN,
            "growth_ok": revenue_growth > 0,
        },
    }


def calculate_position_size(
    portfolio_value: float,
    stock_price: float,
    avg_daily_volume: float,
    current_lines: int = 0,
) -> Dict:
    """
    Calculate position sizing using Igon's rules:
    - Target 33 equal-weight lines at ~3% each
    - Build in 10 tranches
    - Max 20% of daily volume per day

    Args:
        portfolio_value: Total portfolio value
        stock_price: Current stock price
        avg_daily_volume: Average daily trading volume (in currency)
        current_lines: Number of existing portfolio lines

    Returns:
        Dict with sizing recommendations
    """
    target_weight = WEIGHT_PER_LINE / 100
    full_position_value = portfolio_value * target_weight
    tranche_value = full_position_value / BUILD_TRANCHES
    tranche_shares = int(tranche_value / stock_price) if stock_price > 0 else 0

    # Liquidity constraint: max 20% of daily volume
    max_daily_value = avg_daily_volume * MAX_VOLUME_PCT
    days_to_fill_tranche = max(1, int(np.ceil(tranche_value / max_daily_value))) if max_daily_value > 0 else 999
    days_to_full_position = days_to_fill_tranche * BUILD_TRANCHES

    return {
        "target_weight_pct": WEIGHT_PER_LINE,
        "full_position_value": round(full_position_value, 2),
        "tranche_value": round(tranche_value, 2),
        "tranche_shares": tranche_shares,
        "total_tranches": BUILD_TRANCHES,
        "max_daily_purchase": round(max_daily_value, 2),
        "days_per_tranche": days_to_fill_tranche,
        "days_to_full_position": days_to_full_position,
        "current_lines": current_lines,
        "target_lines": TARGET_LINES,
        "room_for_new_lines": max(0, TARGET_LINES - current_lines),
    }


def calculate_exit_schedule(pe: float, position_pct: float) -> Dict:
    """
    Calculate the profit-taking schedule based on PE expansion.

    Igon's rule: from PE 15, sell 20% of position for each 1-point
    PE increase. Full exit at PE 20.

    Args:
        pe: Current price-to-earnings ratio
        position_pct: Current position size as % of portfolio

    Returns:
        Dict with exit schedule and recommended action
    """
    if pe < PE_SELL_START:
        return {
            "action": "HOLD",
            "current_pe": round(pe, 2),
            "distance_to_sell_zone": round(PE_SELL_START - pe, 2),
            "position_to_keep": 100.0,
            "position_to_sell": 0.0,
            "schedule": [],
        }

    if pe >= PE_FULL_EXIT:
        return {
            "action": "FULL_EXIT",
            "current_pe": round(pe, 2),
            "distance_to_sell_zone": 0,
            "position_to_keep": 0.0,
            "position_to_sell": 100.0,
            "schedule": [{"pe": round(pe, 1), "sell_pct": 100.0, "remaining_pct": 0.0}],
        }

    # Gradual exit between PE 15 and 20
    points_above = pe - PE_SELL_START
    cumulative_sell = min(points_above * PE_SELL_STEP_PCT * 100, 100)
    remaining = max(0, 100 - cumulative_sell)

    schedule = []
    for step_pe in range(int(PE_SELL_START), int(PE_FULL_EXIT) + 1):
        if step_pe <= pe:
            pts = step_pe - PE_SELL_START
            sold = min(pts * PE_SELL_STEP_PCT * 100, 100)
            schedule.append({
                "pe": step_pe,
                "sell_pct": round(PE_SELL_STEP_PCT * 100, 1),
                "cumulative_sold_pct": round(sold, 1),
                "remaining_pct": round(max(0, 100 - sold), 1),
                "status": "executed" if step_pe < pe else "current",
            })
        else:
            pts = step_pe - PE_SELL_START
            sold = min(pts * PE_SELL_STEP_PCT * 100, 100)
            schedule.append({
                "pe": step_pe,
                "sell_pct": round(PE_SELL_STEP_PCT * 100, 1),
                "cumulative_sold_pct": round(sold, 1),
                "remaining_pct": round(max(0, 100 - sold), 1),
                "status": "pending",
            })

    return {
        "action": "REDUCE",
        "current_pe": round(pe, 2),
        "distance_to_sell_zone": 0,
        "position_to_keep": round(remaining, 1),
        "position_to_sell": round(cumulative_sell, 1),
        "recommended_sell_value": round(position_pct * cumulative_sell / 100, 2),
        "schedule": schedule,
    }


def portfolio_cleanup_check(holdings: List[Dict]) -> Dict:
    """
    Igon's 6-month portfolio cleanup: identify the biggest losers
    over the trailing 6 months and flag them for removal.

    Igon's philosophy: "sell the losers" - studies show that stocks
    people sell tend to outperform, while stocks they hold (losers)
    continue to underperform.

    Args:
        holdings: List of dicts with keys:
            - symbol: ticker
            - return_6m: 6-month return (%)
            - pe: current PE
            - weight_pct: current portfolio weight

    Returns:
        Dict with cleanup recommendations
    """
    if not holdings:
        return {"candidates": [], "message": "No holdings to review"}

    sorted_holdings = sorted(holdings, key=lambda h: h.get("return_6m", 0))

    # Flag bottom quartile as cleanup candidates
    cutoff = max(1, len(sorted_holdings) // 4)
    candidates = []

    for h in sorted_holdings[:cutoff]:
        ret = h.get("return_6m", 0)
        pe = h.get("pe", 0)

        # If it's cheap (low PE) and the loss is moderate, maybe keep it
        if pe > 0 and pe < 8 and ret > -20:
            action = "REVIEW"
            reason = f"Losing {ret:.1f}% but PE {pe:.1f} is very cheap - may be temporary"
        else:
            action = "CUT"
            reason = f"Losing {ret:.1f}% over 6 months - Igon rule: sell the losers"

        candidates.append({
            "symbol": h.get("symbol", "?"),
            "return_6m": round(ret, 2),
            "pe": round(pe, 2) if pe else None,
            "weight_pct": round(h.get("weight_pct", 0), 2),
            "action": action,
            "reason": reason,
        })

    return {
        "review_date": datetime.now().isoformat(),
        "next_review": "in 6 months",
        "total_holdings": len(holdings),
        "candidates_flagged": len(candidates),
        "candidates": candidates,
        "principle": "Sell the losers - studies show held losers continue to underperform",
    }


def screen_from_fundamentals(fundamentals: Dict) -> Dict:
    """
    Run the full Igon screen on a stock given its fundamental data
    from Yahoo Finance or similar.

    Args:
        fundamentals: Dict with keys like trailingPE, returnOnEquity,
                      revenueGrowth, etc. (yfinance .info format)

    Returns:
        Full Igon analysis dict
    """
    pe = fundamentals.get("trailingPE") or fundamentals.get("forwardPE") or 0
    roe = (fundamentals.get("returnOnEquity") or 0) * 100  # yfinance gives decimal
    rev_growth = (fundamentals.get("revenueGrowth") or 0) * 100

    # Try to compute WCR if balance sheet data available
    wcr_pct = None
    receivables = fundamentals.get("totalReceivables", 0) or 0
    inventory = fundamentals.get("inventory", 0) or 0
    payables = fundamentals.get("totalPayables", 0) or 0
    revenue = fundamentals.get("totalRevenue", 0) or 0
    if revenue > 0 and (receivables > 0 or inventory > 0):
        wcr = receivables + inventory - payables
        wcr_pct = (wcr / revenue) * 100

    stock_score = score_stock(pe, roe, rev_growth, wcr_pct)

    # Add extra context
    stock_score["name"] = fundamentals.get("shortName", "")
    stock_score["sector"] = fundamentals.get("sector", "")
    stock_score["market_cap"] = fundamentals.get("marketCap", 0)
    stock_score["dividend_yield"] = fundamentals.get("dividendYield", 0)

    # Igon note: dividends are the worst selection criterion
    if stock_score.get("dividend_yield"):
        stock_score["warnings"].append(
            "Note: Igon considers dividends the worst stock selection criterion - "
            "PE, P/B, and P/CF all outperform dividend yield as factors"
        )

    return stock_score


def generate_igon_signals(pe_series: np.ndarray, roe: float,
                          revenue_growth: float) -> np.ndarray:
    """
    Generate trading signals from a time series of PE ratios
    using Igon's rules. For integration with the backtester.

    Args:
        pe_series: Array of trailing PE ratios over time
        roe: Return on equity (assumed constant or latest)
        revenue_growth: Revenue growth rate (assumed constant or latest)

    Returns:
        Array of signals: 1=long, 0=flat, -1=short(exit)
    """
    signals = np.zeros(len(pe_series))

    for i, pe in enumerate(pe_series):
        if pe <= 0 or np.isnan(pe):
            signals[i] = 0
            continue

        passes = pe <= PE_BUY_MAX and roe >= ROE_MIN and revenue_growth > 0

        if passes:
            signals[i] = 1  # Long
        elif pe >= PE_FULL_EXIT:
            signals[i] = -1  # Exit
        elif pe >= PE_SELL_START:
            # Partial exit - reduce signal strength proportionally
            reduction = (pe - PE_SELL_START) / (PE_FULL_EXIT - PE_SELL_START)
            signals[i] = max(0, 1 - reduction)
        else:
            signals[i] = 0  # Neutral / hold

    return signals


# ---- Convenience: method summary for display ----

IGON_METHOD_SUMMARY = {
    "name": "Igon Value Screen",
    "author": "William Igon",
    "track_record": "4000%+ returns since 1993 (12.8% annualized net of fees)",
    "fund": "Independence AM - France Small & Mid",
    "philosophy": "Buy undervalued, extremely profitable stocks. Stay disciplined.",
    "entry_rules": {
        "pe": f"< {PE_BUY_MAX}",
        "roe": f"> {ROE_MIN}%",
        "revenue": "Must be growing YoY",
    },
    "exit_rules": {
        "profit_taking_start": f"PE >= {PE_SELL_START}",
        "sell_per_pe_point": f"{PE_SELL_STEP_PCT * 100:.0f}% of position",
        "full_exit": f"PE >= {PE_FULL_EXIT}",
        "loss_cleanup": f"Every {LOSS_REVIEW_MONTHS} months, cut biggest losers",
    },
    "position_sizing": {
        "target_lines": TARGET_LINES,
        "weight_per_line": f"{WEIGHT_PER_LINE}%",
        "build_method": f"{BUILD_TRANCHES} tranches",
        "volume_limit": f"{MAX_VOLUME_PCT * 100:.0f}% of daily volume",
    },
    "additional_metrics": {
        "wcr_turnover": "< 20% excellent, 20-25% normal, > 25% warning",
        "revenue_per_employee": "Efficiency check",
        "financial_costs_vs_debt": "Detect balance sheet dressing",
    },
    "avoid": [
        "Dividends as selection criterion (statistically worst factor)",
        "Market timing (always 100% invested)",
        "Luxury sector (persistently overvalued)",
        "Pharma sector (too expensive for value approach)",
    ],
    "key_quotes": [
        "Buy things cheap and very profitable, then let yourself be carried by the wave",
        "To gain a lot, you must lose very little",
        "Sell the losers - studies prove held losers continue to underperform",
        "We must always remember that we don't know companies very well",
        "The danger of knowing a company too well is thinking you can predict tomorrow",
    ],
}
