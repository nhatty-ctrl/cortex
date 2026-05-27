"""
Cortex supervisor: routes, dispatches, and summarizes multi-agent runs.
"""

import asyncio
import inspect
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Optional

from agents.orchestrator import (
    AlertEngine,
    BullBearSynthesizer,
    CompetitiveIntelligence,
    DCFValuator,
    EarningsAnalyst,
    EventMapper,
    FactChecker,
    FundamentalAnalyst,
    InsiderTracker,
    MacroAnalyst,
    PriceTargetCalculator,
    ReportWriter,
    RiskScorer,
    SentimentAnalyst,
    SmartMoneyTracker,
    TechnicalAnalyst,
)
from agents.specialists.goldman_suite import (
    ASSET_UNIVERSE,
    AgentResult,
    AlphaCalculator,
    CrashPredictor,
    CurrencyCommodityAnalyst,
    GeopoliticalRiskRadar,
    HistoricalPlaybook,
    HoldWindowEngine,
    MultiAssetRiskCalculator,
    PeerRelativeValue,
    SocialIntelligenceAgent,
)

TIMEOUTS = {
    "urgent": 8,
    "fast": 15,
    "standard": 30,
    "deep": 60,
}

SQUADS = {
    "urgent": ["crash_predictor", "alert_engine", "hold_window_engine"],
    "quick": [
        "fundamental_analyst",
        "crash_predictor",
        "sentiment_aggregator",
        "bull_bear_synthesizer",
    ],
    "full": [
        "fundamental_analyst",
        "technical_analyst",
        "sentiment_aggregator",
        "event_mapper",
        "crash_predictor",
        "alpha_calculator",
        "risk_calculator",
        "geo_risk_radar",
        "social_intelligence",
        "smart_money_tracker",
        "insider_tracker",
        "earnings_analyst",
        "competitive_intelligence",
        "peer_relative_value",
        "currency_commodity_analyst",
        "dcf_valuator",
        "historical_playbook",
        "hold_window_engine",
        "bull_bear_synthesizer",
        "price_target_calculator",
        "report_writer",
        "fact_checker",
        "alert_engine",
    ],
    "macro": [
        "geo_risk_radar",
        "macro_analyst",
        "sentiment_aggregator",
        "crash_predictor",
    ],
    "valuation": [
        "fundamental_analyst",
        "dcf_valuator",
        "peer_relative_value",
        "price_target_calculator",
    ],
    "insider": ["insider_tracker", "smart_money_tracker", "alpha_calculator"],
}

PHASE_1 = {
    "fundamental_analyst",
    "technical_analyst",
    "sentiment_aggregator",
    "event_mapper",
    "crash_predictor",
    "alpha_calculator",
    "risk_calculator",
    "geo_risk_radar",
    "social_intelligence",
    "smart_money_tracker",
    "insider_tracker",
    "earnings_analyst",
    "competitive_intelligence",
    "peer_relative_value",
    "currency_commodity_analyst",
    "macro_analyst",
}
PHASE_2 = {
    "dcf_valuator",
    "historical_playbook",
    "hold_window_engine",
    "bull_bear_synthesizer",
    "price_target_calculator",
}
PHASE_3 = {"report_writer", "fact_checker", "alert_engine"}


@dataclass
class SupervisorRun:
    asset: str
    company: str
    mode: str
    started_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    completed_at: Optional[str] = None
    agents_run: int = 0
    agents_failed: int = 0
    total_ms: int = 0
    crash_risk: str = "unknown"
    final_rating: str = "NEUTRAL"
    signal: str = "neutral"
    confidence: float = 0.0


class CortexSupervisor:
    REGISTRY = {
        "fundamental_analyst": FundamentalAnalyst,
        "technical_analyst": TechnicalAnalyst,
        "sentiment_aggregator": SentimentAnalyst,
        "event_mapper": EventMapper,
        "crash_predictor": CrashPredictor,
        "alpha_calculator": AlphaCalculator,
        "risk_calculator": MultiAssetRiskCalculator,
        "geo_risk_radar": GeopoliticalRiskRadar,
        "social_intelligence": SocialIntelligenceAgent,
        "smart_money_tracker": SmartMoneyTracker,
        "insider_tracker": InsiderTracker,
        "earnings_analyst": EarningsAnalyst,
        "competitive_intelligence": CompetitiveIntelligence,
        "peer_relative_value": PeerRelativeValue,
        "currency_commodity_analyst": CurrencyCommodityAnalyst,
        "dcf_valuator": DCFValuator,
        "historical_playbook": HistoricalPlaybook,
        "hold_window_engine": HoldWindowEngine,
        "bull_bear_synthesizer": BullBearSynthesizer,
        "price_target_calculator": PriceTargetCalculator,
        "report_writer": ReportWriter,
        "fact_checker": FactChecker,
        "alert_engine": AlertEngine,
        "macro_analyst": MacroAnalyst,
    }

    def __init__(self, signal_queue=None, scrape_queue=None):
        self._signal_q = signal_queue
        self._scrape_q = scrape_queue

    def bind_queues(self, signal_queue=None, scrape_queue=None):
        if signal_queue is not None:
            self._signal_q = signal_queue
        if scrape_queue is not None:
            self._scrape_q = scrape_queue

    async def _emit(self, queue, event: dict):
        if queue:
            try:
                await queue.put(event)
            except Exception:
                pass

    async def _emit_progress(self, asset: str, message: str, agent: str = "", status: str = "running"):
        await self._emit(
            self._scrape_q,
            {
                "type": "agent_progress",
                "asset": asset,
                "agent": agent,
                "message": message,
                "status": status,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    def _build_run_kwargs(
        self,
        agent,
        asset: str,
        company: str,
        context: str,
        prior: dict | None,
        **kwargs,
    ) -> dict:
        params = inspect.signature(agent.run).parameters
        run_kwargs = {**kwargs}

        if prior:
            if "agent_outputs" in params:
                run_kwargs["agent_outputs"] = prior
            if "all_outputs" in params:
                run_kwargs["all_outputs"] = prior
            if "all_agent_outputs" in params:
                run_kwargs["all_agent_outputs"] = prior

        if "ticker" in params:
            run_kwargs["ticker"] = asset
        if "asset" in params:
            run_kwargs["asset"] = asset
        if "company" in params:
            run_kwargs["company"] = company
        if "context" in params:
            run_kwargs["context"] = context

        return run_kwargs

    async def _run_one(
        self,
        name: str,
        asset: str,
        company: str,
        context: str,
        timeout: int,
        prior: dict | None = None,
        **kwargs,
    ) -> tuple[str, Optional[AgentResult]]:
        cls = self.REGISTRY.get(name)
        if not cls:
            return name, None

        agent = cls()
        run_kwargs = self._build_run_kwargs(agent, asset, company, context, prior, **kwargs)
        await self._emit_progress(asset, f"Running {name}...", agent=name, status="running")

        try:
            result = await asyncio.wait_for(agent.run(**run_kwargs), timeout=timeout)
            await self._emit_progress(asset, f"{name} done ({result.duration_ms}ms)", agent=name, status="done")
            return name, result
        except asyncio.TimeoutError:
            await self._emit_progress(asset, f"{name} timed out after {timeout}s", agent=name, status="timeout")
            return (
                name,
                AgentResult(
                    agent=name,
                    asset=asset,
                    asset_class="unknown",
                    output={"error": f"timeout after {timeout}s"},
                    confidence=0.0,
                    model_used="timeout",
                    duration_ms=timeout * 1000,
                    error=f"timeout after {timeout}s",
                ),
            )
        except Exception as exc:
            await self._emit_progress(asset, f"{name} error: {str(exc)[:60]}", agent=name, status="error")
            return (
                name,
                AgentResult(
                    agent=name,
                    asset=asset,
                    asset_class="unknown",
                    output={"error": str(exc)},
                    confidence=0.0,
                    model_used="error",
                    duration_ms=0,
                    error=str(exc),
                ),
            )

    async def _run_phase(
        self,
        names: list[str],
        asset: str,
        company: str,
        context: str,
        timeout: int,
        prior: dict | None = None,
        **kwargs,
    ) -> dict:
        tasks = [self._run_one(name, asset, company, context, timeout, prior, **kwargs) for name in names]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        output = {}
        for item in results:
            if isinstance(item, tuple):
                result_name, result = item
                output[result_name] = result
        return output

    async def run(
        self,
        asset: str,
        company: str,
        context: str,
        mode: str = "quick",
        portfolio_size: float = 100_000,
        entry_price: float = 0.0,
        headlines: list | None = None,
        social_data: dict | None = None,
    ) -> dict:
        t0 = time.monotonic()
        squad = SQUADS.get(mode, SQUADS["quick"])
        run = SupervisorRun(asset=asset, company=company, mode=mode)
        asset_info = ASSET_UNIVERSE.get(asset, {"name": company, "class": "stock"})

        await self._emit_progress(
            asset,
            f"CortexSupervisor: launching {mode} run for {company} ({asset}) - {len(squad)} agents",
            agent="supervisor",
            status="running",
        )

        kwargs = {
            "portfolio_size_usd": portfolio_size,
            "entry_price": entry_price,
            "headlines": headlines or [],
            "social_data": social_data or {},
        }

        p1_results: dict = {}
        p1_names = [name for name in squad if name in PHASE_1]
        if p1_names:
            await self._emit_progress(asset, f"Phase 1: {len(p1_names)} agents in parallel...", agent="supervisor")
            p1_results = await self._run_phase(p1_names, asset, company, context, TIMEOUTS["standard"], **kwargs)

        p2_results: dict = {}
        p2_names = [name for name in squad if name in PHASE_2]
        if p2_names:
            await self._emit_progress(asset, f"Phase 2: {len(p2_names)} synthesis agents...", agent="supervisor")
            p2_results = await self._run_phase(
                p2_names,
                asset,
                company,
                context,
                TIMEOUTS["deep"],
                prior=p1_results,
                **kwargs,
            )

        p3_results: dict = {}
        p3_names = [name for name in squad if name in PHASE_3]
        all_prior = {**p1_results, **p2_results}
        if p3_names:
            await self._emit_progress(asset, "Phase 3: report + audit...", agent="supervisor")
            p3_results = await self._run_phase(
                p3_names,
                asset,
                company,
                context,
                TIMEOUTS["deep"],
                prior=all_prior,
                **kwargs,
            )

        all_results = {**p1_results, **p2_results, **p3_results}

        run.agents_run = sum(1 for result in all_results.values() if result and not result.error)
        run.agents_failed = sum(1 for result in all_results.values() if result and result.error)
        run.total_ms = int((time.monotonic() - t0) * 1000)
        run.completed_at = datetime.utcnow().isoformat()

        crash = all_results.get("crash_predictor")
        if crash and not crash.error:
            run.crash_risk = crash.output.get("crash_risk_level", "unknown")

        report_agent = all_results.get("report_writer")
        if report_agent and not report_agent.error:
            report_payload = report_agent.output.get("report", {})
            run.final_rating = report_payload.get("rating", "NEUTRAL").upper()
            run.confidence = report_agent.confidence

        bull_bear = all_results.get("bull_bear_synthesizer")
        if bull_bear and not bull_bear.error:
            run.signal = bull_bear.output.get("final_verdict", "neutral")
            run.confidence = max(run.confidence, bull_bear.confidence)

        summary = {
            "asset": asset,
            "company": company,
            "asset_class": asset_info.get("class", "stock"),
            "mode": mode,
            "rating": run.final_rating,
            "signal": run.signal,
            "confidence": round(run.confidence, 3),
            "crash_risk": run.crash_risk,
            "agents_run": run.agents_run,
            "agents_failed": run.agents_failed,
            "total_ms": run.total_ms,
            "timestamp": run.completed_at,
        }

        await self._emit(
            self._signal_q,
            {
                "type": "new_signal",
                "asset": asset,
                "summary": summary,
                "report": report_agent.output if report_agent and not report_agent.error else {},
                "timestamp": run.completed_at,
            },
        )
        await self._emit_progress(
            asset,
            f"{company} complete - {run.agents_run} agents, {run.total_ms}ms, rating: {run.final_rating}",
            agent="supervisor",
            status="done",
        )

        return {
            "run": asdict(run),
            "summary": summary,
            "results": {
                name: {
                    "output": result.output,
                    "confidence": result.confidence,
                    "duration_ms": result.duration_ms,
                    "error": result.error,
                }
                for name, result in all_results.items()
                if result
            },
        }


_supervisor_instance: Optional[CortexSupervisor] = None


def get_supervisor(signal_queue=None, scrape_queue=None) -> CortexSupervisor:
    global _supervisor_instance
    if _supervisor_instance is None:
        _supervisor_instance = CortexSupervisor(signal_queue=signal_queue, scrape_queue=scrape_queue)
    else:
        _supervisor_instance.bind_queues(signal_queue=signal_queue, scrape_queue=scrape_queue)
    return _supervisor_instance


async def cortex_run(
    asset: str,
    company: str,
    context: str,
    mode: str = "quick",
    signal_q=None,
    scrape_q=None,
    **kwargs,
) -> dict:
    supervisor = get_supervisor(signal_q, scrape_q)
    return await supervisor.run(asset, company, context, mode=mode, **kwargs)
