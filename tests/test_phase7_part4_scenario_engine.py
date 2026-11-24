"""Phase 7 Part 4 — Domain-Agnostic Scenario Engine tests.

Covers all 20 required test categories:
1. Scenario contract validation
2. Valid supported numerical simulations
3. Missing assumptions
4. Insufficient evidence
5. Unsupported calculations
6. Baseline-versus-alternative comparisons
7. Multiple scenarios
8. Deterministic results
9. Invalid and non-finite values
10. Unit compatibility
11. Out-of-range changes
12. Contradictory evidence
13. Dataset isolation
14. Serialization
15. Forecast and model-estimate labeling
16. Scenario-versus-fact safeguards
17. No guaranteed-outcome claims
18. No invented financial estimates
19. Generic non-marketing examples
20. Regression-safe imports
"""

import math
import pytest

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from src.decision.contracts import (
    ConfidenceLevel,
    ConfidenceScore,
    DecisionEvidence,
    DecisionLimitation,
    DecisionSignal,
    EvidenceLevel,
    LimitationCategory,
    _make_id,
    _safe_float,
)
from src.decision.evidence import (
    create_evidence,
    create_limitation,
    validate_dataset_scope,
)
from src.decision.confidence import assess_confidence
from src.decision.signal_synthesizer import (
    SignalPriority,
    SynthesizedSignal,
    SynthesisResult,
)
from src.decision.impact_quantifier import (
    ImpactType,
    ImpactCategory,
    ImpactAssessment,
    ImpactReport,
    ImpactQuantifier,
)
from src.decision.scenario_engine import (
    ScenarioStatus,
    ScenarioParameter,
    ScenarioDefinition,
    ScenarioResult,
    ScenarioComparison,
    ScenarioEngine,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DS_ID = "ds_test_004"


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_evidence(
    level=EvidenceLevel.FACT.value,
    description="Test evidence",
    source="test_source",
    metric_name="metric",
    metric_value=None,
    details=None,
    dataset_id=DS_ID,
    limitations=None,
) -> DecisionEvidence:
    return DecisionEvidence(
        level=level,
        description=description,
        source=source,
        metric_name=metric_name,
        metric_value=metric_value,
        details=details or {},
        dataset_id=dataset_id,
        limitations=limitations or [],
    )


def _make_param(
    name="param_a",
    baseline_value=100.0,
    alternative_value=120.0,
    unit="units",
    supported=True,
    unsupported_reason="",
) -> ScenarioParameter:
    return ScenarioParameter(
        name=name,
        baseline_value=baseline_value,
        alternative_value=alternative_value,
        unit=unit,
        supported=supported,
        unsupported_reason=unsupported_reason,
    )


def _make_model_info(
    coefficients=None,
    intercept=0.0,
    r_squared=0.85,
    model_type="linear regression",
    unit="target_units",
    supported_range=None,
) -> dict:
    return {
        "coefficients": coefficients if coefficients is not None else {"param_a": 2.0},
        "intercept": intercept,
        "r_squared": r_squared,
        "model_type": model_type,
        "unit": unit,
        "supported_range": supported_range or {},
    }


# ===================================================================
# 20. Regression-safe imports
# ===================================================================

class TestRegressionSafeImports:
    """Verify all Part 1–4 modules and public names import cleanly."""

    def test_import_contracts(self):
        from src.decision import contracts
        assert hasattr(contracts, "EvidenceLevel")
        assert hasattr(contracts, "ConfidenceLevel")

    def test_import_evidence(self):
        from src.decision import evidence
        assert hasattr(evidence, "create_evidence")

    def test_import_confidence(self):
        from src.decision import confidence
        assert hasattr(confidence, "assess_confidence")

    def test_import_signal_synthesizer(self):
        from src.decision import signal_synthesizer
        assert hasattr(signal_synthesizer, "SynthesizedSignal")

    def test_import_impact_quantifier(self):
        from src.decision import impact_quantifier
        assert hasattr(impact_quantifier, "ImpactQuantifier")

    def test_import_scenario_engine(self):
        from src.decision import scenario_engine
        assert hasattr(scenario_engine, "ScenarioEngine")
        assert hasattr(scenario_engine, "ScenarioStatus")
        assert hasattr(scenario_engine, "ScenarioParameter")
        assert hasattr(scenario_engine, "ScenarioDefinition")
        assert hasattr(scenario_engine, "ScenarioResult")
        assert hasattr(scenario_engine, "ScenarioComparison")

    def test_scenario_status_values(self):
        assert ScenarioStatus.SIMULATED.value == "simulated"
        assert ScenarioStatus.PARTIALLY_SIMULATED.value == "partially_simulated"
        assert ScenarioStatus.NOT_SIMULATABLE.value == "not_simulatable"
        assert ScenarioStatus.INSUFFICIENT_EVIDENCE.value == "insufficient_evidence"

    def test_scenario_status_is_str_enum(self):
        assert isinstance(ScenarioStatus.SIMULATED, str)
        assert isinstance(ScenarioStatus.SIMULATED, ScenarioStatus)


# ===================================================================
# 1. Scenario contract validation
# ===================================================================

class TestScenarioContractValidation:
    """ScenarioParameter, ScenarioDefinition, ScenarioResult validation."""

    def test_parameter_requires_name(self):
        with pytest.raises(ValueError, match="non-empty name"):
            ScenarioParameter(name="")

    def test_definition_requires_name(self):
        with pytest.raises(ValueError, match="non-empty name"):
            ScenarioDefinition(name="", description="d", dataset_id=DS_ID)

    def test_definition_requires_description(self):
        with pytest.raises(ValueError, match="non-empty description"):
            ScenarioDefinition(name="n", description="", dataset_id=DS_ID)

    def test_definition_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id"):
            ScenarioDefinition(name="n", description="d", dataset_id="")

    def test_result_requires_valid_status(self):
        with pytest.raises(ValueError, match="Invalid scenario status"):
            ScenarioResult(status="bogus", description="d")

    def test_result_requires_description(self):
        with pytest.raises(ValueError, match="non-empty description"):
            ScenarioResult(
                status=ScenarioStatus.SIMULATED.value, description=""
            )

    def test_comparison_requires_two_results(self):
        r1 = ScenarioResult(
            status=ScenarioStatus.SIMULATED.value,
            description="result 1",
        )
        with pytest.raises(ValueError, match="at least two"):
            ScenarioComparison(results=[r1])

    def test_valid_definition_creation(self):
        defn = ScenarioDefinition(
            name="Test scenario",
            description="Increase parameter A",
            dataset_id=DS_ID,
            parameters=[_make_param()],
            assumptions=["Relationship is linear"],
        )
        assert defn.name == "Test scenario"
        assert defn.dataset_id == DS_ID
        assert len(defn.parameters) == 1

    def test_engine_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id"):
            ScenarioEngine(dataset_id="")


# ===================================================================
# 2. Valid supported numerical simulations
# ===================================================================

class TestValidNumericalSimulations:
    """Test scenarios that can be fully numerically evaluated."""

    def test_single_param_linear_model(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Observed relationship")
        defn = engine.define_scenario(
            name="Increase A",
            description="What if param A increases by 20%?",
            parameters=[_make_param(baseline_value=100.0, alternative_value=120.0)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 2.0},
                intercept=10.0,
            ),
        )
        assert result.status == ScenarioStatus.SIMULATED.value
        assert result.estimated_outcome is not None
        assert result.baseline_outcome is not None
        assert result.delta is not None
        # baseline = 10 + 2*100 = 210, alternative = 10 + 2*120 = 250
        assert abs(result.baseline_outcome - 210.0) < 0.01
        assert abs(result.estimated_outcome - 250.0) < 0.01
        assert abs(result.delta - 40.0) < 0.01

    def test_multi_param_model(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Model evidence")
        params = [
            _make_param(name="x1", baseline_value=10, alternative_value=15),
            _make_param(name="x2", baseline_value=5, alternative_value=8),
        ]
        defn = engine.define_scenario(
            name="Multi-param",
            description="Change two params",
            parameters=params,
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"x1": 3.0, "x2": -1.0},
                intercept=5.0,
            ),
        )
        assert result.status == ScenarioStatus.SIMULATED.value
        # baseline = 5 + 3*10 + (-1)*5 = 30
        # alt = 5 + 3*15 + (-1)*8 = 42
        assert abs(result.baseline_outcome - 30.0) < 0.01
        assert abs(result.estimated_outcome - 42.0) < 0.01

    def test_delta_pct_calculated(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Delta pct",
            description="Compute percentage delta",
            parameters=[_make_param(baseline_value=100, alternative_value=150)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 1.0}, intercept=0.0,
            ),
        )
        assert result.delta_pct is not None
        # delta = 50, baseline = 100, pct = 50%
        assert abs(result.delta_pct - 50.0) < 0.01


# ===================================================================
# 3. Missing assumptions
# ===================================================================

class TestMissingAssumptions:
    """Scenarios must surface their assumptions."""

    def test_assumptions_carried_from_definition(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="Assumed",
            description="Scenario with assumptions",
            parameters=[_make_param()],
            assumptions=["Market conditions remain stable"],
        )
        ev = _make_evidence(description="Evidence")
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        assert "Market conditions remain stable" in result.assumptions

    def test_model_assumptions_added(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Model",
            description="Model scenario",
            parameters=[_make_param()],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(model_type="ridge regression"),
        )
        assert any("ridge regression" in a for a in result.assumptions)


# ===================================================================
# 4. Insufficient evidence
# ===================================================================

class TestInsufficientEvidence:
    """Scenarios with no or insufficient evidence."""

    def test_no_evidence_no_model(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="No data",
            description="Nothing to evaluate",
            parameters=[_make_param()],
        )
        result = engine.evaluate_scenario(defn)
        assert result.status == ScenarioStatus.INSUFFICIENT_EVIDENCE.value
        assert result.confidence.level == ConfidenceLevel.INSUFFICIENT.value

    def test_no_parameters(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = ScenarioDefinition(
            name="Empty",
            description="No params",
            dataset_id=DS_ID,
        )
        ev = _make_evidence(description="Evidence")
        result = engine.evaluate_scenario(defn, evidence_items=[ev])
        assert result.status == ScenarioStatus.INSUFFICIENT_EVIDENCE.value

    def test_all_evidence_filtered_out(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        other_ev = _make_evidence(
            description="Other dataset",
            dataset_id="ds_other",
        )
        defn = engine.define_scenario(
            name="Wrong data",
            description="Evidence from different dataset",
            parameters=[_make_param()],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[other_ev],
        )
        assert result.status in {
            ScenarioStatus.INSUFFICIENT_EVIDENCE.value,
            ScenarioStatus.NOT_SIMULATABLE.value,
        }


# ===================================================================
# 5. Unsupported calculations
# ===================================================================

class TestUnsupportedCalculations:
    """Scenarios where the engine cannot compute a result."""

    def test_evidence_but_no_model(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Data exists")
        defn = engine.define_scenario(
            name="No model",
            description="Evidence but no model",
            parameters=[_make_param()],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(defn, evidence_items=[ev])
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value
        assert len(result.missing_requirements) > 0
        assert "model" in result.explanation.lower() or "formula" in result.explanation.lower()

    def test_model_without_matching_coefficients(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="No coefficients",
            description="Model has no matching params",
            parameters=[_make_param(name="unmatched_param")],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(coefficients={"other_param": 1.0}),
        )
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value

    def test_empty_coefficients(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Empty model",
            description="Model with empty coefficients",
            parameters=[_make_param()],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(coefficients={}),
        )
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value

    def test_all_params_unsupported(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        param = _make_param(supported=False, unsupported_reason="No data")
        defn = engine.define_scenario(
            name="All unsupported",
            description="All params unsupported",
            parameters=[param],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value


# ===================================================================
# 6. Baseline-versus-alternative comparisons
# ===================================================================

class TestBaselineAlternativeComparisons:
    """Verify baseline vs alternative outcome separation."""

    def test_baseline_and_alternative_separate(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Compare",
            description="Baseline vs alternative",
            parameters=[_make_param(baseline_value=50, alternative_value=75)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 2.0}, intercept=0.0
            ),
        )
        assert result.baseline_outcome != result.estimated_outcome
        assert result.baseline_outcome == 100.0  # 0 + 2*50
        assert result.estimated_outcome == 150.0  # 0 + 2*75

    def test_change_pct_computed_for_parameter(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="Pct change",
            description="Param change percentage",
            parameters=[_make_param(baseline_value=200, alternative_value=250)],
        )
        param = defn.parameters[0]
        assert param.change_pct is not None
        assert abs(param.change_pct - 25.0) < 0.01


# ===================================================================
# 7. Multiple scenarios
# ===================================================================

class TestMultipleScenarios:
    """Multiple scenarios evaluated and compared."""

    def test_compare_two_scenarios(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        model = _make_model_info(coefficients={"param_a": 2.0}, intercept=0.0)

        defn_a = engine.define_scenario(
            name="Low", description="Low change",
            parameters=[_make_param(baseline_value=100, alternative_value=110)],
            evidence_items=[ev],
        )
        defn_b = engine.define_scenario(
            name="High", description="High change",
            parameters=[_make_param(baseline_value=100, alternative_value=200)],
            evidence_items=[ev],
        )
        r_a = engine.evaluate_scenario(defn_a, evidence_items=[ev], model_info=model)
        r_b = engine.evaluate_scenario(defn_b, evidence_items=[ev], model_info=model)

        comparison = engine.compare_scenarios([r_a, r_b])
        assert len(comparison.results) == 2
        assert comparison.best_outcome is not None
        assert comparison.best_outcome.estimated_outcome == r_b.estimated_outcome

    def test_compare_mixed_status(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")

        r_sim = ScenarioResult(
            dataset_id=DS_ID,
            status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=100.0,
            description="Simulated result",
        )
        r_nosim = ScenarioResult(
            dataset_id=DS_ID,
            status=ScenarioStatus.NOT_SIMULATABLE.value,
            description="Not simulatable result",
        )
        comparison = engine.compare_scenarios([r_sim, r_nosim])
        assert not comparison.all_simulated
        assert any("could not be numerically" in n for n in comparison.comparison_notes)


# ===================================================================
# 8. Deterministic results
# ===================================================================

class TestDeterministicResults:
    """Same inputs produce identical outputs."""

    def test_same_inputs_same_output(self):
        ev = _make_evidence(description="Evidence", metric_value=42.0)
        model = _make_model_info(coefficients={"param_a": 2.0}, intercept=5.0)
        params = [_make_param(baseline_value=10, alternative_value=20)]

        results = []
        for _ in range(3):
            engine = ScenarioEngine(dataset_id=DS_ID)
            defn = engine.define_scenario(
                name="Det test", description="Determinism",
                parameters=params, evidence_items=[ev],
            )
            r = engine.evaluate_scenario(defn, evidence_items=[ev], model_info=model)
            results.append(r)

        assert all(r.status == results[0].status for r in results)
        assert all(r.estimated_outcome == results[0].estimated_outcome for r in results)
        assert all(r.baseline_outcome == results[0].baseline_outcome for r in results)
        assert all(r.delta == results[0].delta for r in results)

    def test_comparison_deterministic(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=100.0, description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=200.0, description="R2",
        )
        engine = ScenarioEngine(dataset_id=DS_ID)
        comp_a = engine.compare_scenarios([r1, r2])
        comp_b = engine.compare_scenarios([r1, r2])
        assert comp_a.comparison_notes == comp_b.comparison_notes


# ===================================================================
# 9. Invalid and non-finite values
# ===================================================================

class TestInvalidNonFiniteValues:
    """NaN, Inf, and other invalid values are handled safely."""

    def test_nan_baseline_sanitised(self):
        param = _make_param(baseline_value=float("nan"), alternative_value=10.0)
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="NaN base", description="NaN baseline",
            parameters=[param],
        )
        validated = defn.parameters[0]
        assert validated.baseline_value is None  # NaN → None via _safe_float

    def test_inf_alternative_sanitised(self):
        param = _make_param(baseline_value=10.0, alternative_value=float("inf"))
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="Inf alt", description="Inf alternative",
            parameters=[param],
        )
        validated = defn.parameters[0]
        assert validated.alternative_value is None
        assert not validated.supported

    def test_nan_coefficient_handled(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="NaN coeff", description="NaN coefficient",
            parameters=[_make_param()],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": float("nan")}
            ),
        )
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value

    def test_serialization_sanitises_nan(self):
        result = ScenarioResult(
            status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=float("nan"),
            description="NaN result",
        )
        d = result.to_dict()
        assert d["estimated_outcome"] is None

    def test_serialization_sanitises_inf(self):
        result = ScenarioResult(
            status=ScenarioStatus.SIMULATED.value,
            delta=float("inf"),
            description="Inf delta",
        )
        d = result.to_dict()
        assert d["delta"] is None


# ===================================================================
# 10. Unit compatibility
# ===================================================================

class TestUnitCompatibility:
    """Units are preserved and incompatible units are flagged."""

    def test_unit_preserved_in_result(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Units", description="Unit test",
            parameters=[_make_param(unit="kg")],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(unit="tonnes"),
        )
        assert result.unit == "tonnes"

    def test_incompatible_units_in_comparison(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=100.0, unit="kg",
            description="Result in kg",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=50.0, unit="liters",
            description="Result in liters",
        )
        comparison = engine.compare_scenarios([r1, r2])
        assert any("Incompatible units" in lim.description
                    for lim in comparison.limitations
                    if hasattr(lim, "description"))
        assert any("different units" in n for n in comparison.comparison_notes)

    def test_parameter_unit_preserved(self):
        param = _make_param(unit="megawatts")
        assert param.unit == "megawatts"
        d = param.to_dict()
        assert d["unit"] == "megawatts"


# ===================================================================
# 11. Out-of-range changes
# ===================================================================

class TestOutOfRangeChanges:
    """Extrapolation outside model's supported range is flagged."""

    def test_out_of_range_warning(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Extrapolation",
            description="Value outside model range",
            parameters=[_make_param(baseline_value=50, alternative_value=500)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 1.0},
                supported_range={"param_a": (0, 200)},
            ),
        )
        # Should still evaluate but with warnings
        assert any("outside supported range" in lim.description
                    for lim in result.limitations
                    if hasattr(lim, "description"))
        assert "outside" in result.explanation.lower() or "range" in result.explanation.lower()

    def test_within_range_no_warning(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="In range",
            description="Value within model range",
            parameters=[_make_param(baseline_value=50, alternative_value=100)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 1.0},
                supported_range={"param_a": (0, 200)},
            ),
        )
        scope_lims = [
            lim for lim in result.limitations
            if hasattr(lim, "category") and lim.category == LimitationCategory.SCOPE.value
        ]
        assert len(scope_lims) == 0


# ===================================================================
# 12. Contradictory evidence
# ===================================================================

class TestContradictoryEvidence:
    """Contradictory evidence is preserved, not discarded."""

    def test_contradictory_evidence_preserved(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev_up = _make_evidence(
            description="Metric trending up",
            details={"direction": "increasing"},
        )
        ev_down = _make_evidence(
            description="Metric trending down",
            details={"direction": "decreasing"},
        )
        defn = engine.define_scenario(
            name="Contradictory",
            description="Conflicting evidence",
            parameters=[_make_param()],
            evidence_items=[ev_up, ev_down],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev_up, ev_down],
            model_info=_make_model_info(),
        )
        # Both evidence items should be referenced
        assert ev_up.id in result.evidence_ids
        assert ev_down.id in result.evidence_ids


# ===================================================================
# 13. Dataset isolation
# ===================================================================

class TestDatasetIsolation:
    """Scenarios enforce dataset isolation."""

    def test_engine_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id"):
            ScenarioEngine(dataset_id="")

    def test_definition_requires_dataset_id(self):
        with pytest.raises(ValueError, match="dataset_id"):
            ScenarioDefinition(name="n", description="d", dataset_id="")

    def test_mismatched_scenario_dataset(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = ScenarioDefinition(
            name="Wrong DS", description="Wrong dataset",
            dataset_id="ds_other",
        )
        result = engine.evaluate_scenario(defn)
        assert result.status == ScenarioStatus.INSUFFICIENT_EVIDENCE.value

    def test_cross_dataset_evidence_filtered(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev_same = _make_evidence(description="Same dataset", dataset_id=DS_ID)
        ev_other = _make_evidence(description="Other dataset", dataset_id="ds_other")
        defn = engine.define_scenario(
            name="Mixed", description="Mixed dataset evidence",
            parameters=[_make_param()],
            evidence_items=[ev_same, ev_other],
        )
        # Only same-dataset evidence should be kept
        assert ev_same.id in defn.evidence_ids
        assert ev_other.id not in defn.evidence_ids

    def test_result_carries_dataset_id(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="DS check", description="Dataset check",
            parameters=[_make_param()],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        assert result.dataset_id == DS_ID

    def test_comparison_rejects_cross_dataset(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            description="R1",
        )
        r2 = ScenarioResult(
            dataset_id="ds_other", status=ScenarioStatus.SIMULATED.value,
            description="R2",
        )
        with pytest.raises(ValueError, match="different datasets"):
            engine.compare_scenarios([r1, r2])

    def test_unscoped_evidence_accepted(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev_unscoped = _make_evidence(description="Unscoped", dataset_id="")
        defn = engine.define_scenario(
            name="Unscoped", description="Unscoped evidence",
            parameters=[_make_param()],
            evidence_items=[ev_unscoped],
        )
        assert ev_unscoped.id in defn.evidence_ids


# ===================================================================
# 14. Serialization
# ===================================================================

class TestSerialization:
    """to_dict / from_dict roundtrips."""

    def test_parameter_roundtrip(self):
        param = ScenarioParameter(
            name="p", baseline_value=100.0, alternative_value=120.0,
            unit="u", supported=True, change_pct=20.0,
        )
        d = param.to_dict()
        restored = ScenarioParameter.from_dict(d)
        assert restored.name == param.name
        assert restored.baseline_value == param.baseline_value
        assert restored.alternative_value == param.alternative_value
        assert restored.change_pct == 20.0

    def test_definition_roundtrip(self):
        defn = ScenarioDefinition(
            name="Roundtrip", description="Test",
            dataset_id=DS_ID,
            parameters=[_make_param()],
            assumptions=["Assume A"],
        )
        d = defn.to_dict()
        restored = ScenarioDefinition.from_dict(d)
        assert restored.name == defn.name
        assert restored.dataset_id == defn.dataset_id
        assert len(restored.parameters) == 1

    def test_result_roundtrip(self):
        result = ScenarioResult(
            scenario_id="scen_001",
            dataset_id=DS_ID,
            status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=42.5,
            baseline_outcome=40.0,
            delta=2.5,
            delta_pct=6.25,
            unit="target_units",
            description="Test result",
            explanation="Explanation",
            confidence=ConfidenceScore(
                level=ConfidenceLevel.LOW.value,
                explanation="Low confidence",
                dataset_id=DS_ID,
            ),
            assumptions=["A1"],
        )
        d = result.to_dict()
        assert d["status"] == "simulated"
        assert d["estimated_outcome"] == 42.5
        restored = ScenarioResult.from_dict(d)
        assert restored.status == result.status
        assert restored.estimated_outcome == result.estimated_outcome

    def test_comparison_roundtrip(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=100.0, description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=200.0, description="R2",
        )
        comp = ScenarioComparison(
            dataset_id=DS_ID, results=[r1, r2],
            comparison_notes=["Note 1"],
        )
        d = comp.to_dict()
        assert d["result_count"] == 2
        assert d["all_simulated"] is True
        restored = ScenarioComparison.from_dict(d)
        assert len(restored.results) == 2

    def test_extra_keys_ignored_in_from_dict(self):
        d = {
            "name": "Test",
            "baseline_value": 10.0,
            "alternative_value": 20.0,
            "extra_key": "should be ignored",
        }
        param = ScenarioParameter.from_dict(d)
        assert param.name == "Test"

    def test_result_none_confidence(self):
        result = ScenarioResult(
            status=ScenarioStatus.SIMULATED.value,
            description="No confidence",
        )
        d = result.to_dict()
        assert d["confidence"] is None


# ===================================================================
# 15. Forecast and model-estimate labeling
# ===================================================================

class TestForecastModelEstimateLabeling:
    """Scenario results correctly label model estimates."""

    def test_estimation_method_mentions_model(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Model label", description="Model labeling test",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(model_type="gradient boosting"),
        )
        assert "gradient boosting" in result.estimation_method.lower() or \
               "gradient boosting" in result.model_description.lower()

    def test_r_squared_in_explanation(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="R2", description="R-squared test",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(r_squared=0.92),
        )
        assert "0.92" in result.explanation

    def test_model_uncertainty_limitation(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Uncertainty", description="Model uncertainty",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        assert any(
            hasattr(lim, "category") and
            lim.category == LimitationCategory.MODEL_UNCERTAINTY.value
            for lim in result.limitations
        )


# ===================================================================
# 16. Scenario-versus-fact safeguards
# ===================================================================

class TestScenarioVersusFactSafeguards:
    """Scenario results are never presented as facts."""

    def test_explanation_says_not_observed(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Not fact", description="Scenario not fact",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        lower_expl = result.explanation.lower()
        assert ("projection" in lower_expl or "not" in lower_expl or
                "model" in lower_expl)

    def test_limitations_mention_not_observed(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Not observed", description="Test",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        lim_text = " ".join(
            lim.description for lim in result.limitations
            if hasattr(lim, "description")
        ).lower()
        assert "not observed" in lim_text or "projection" in lim_text

    def test_not_simulatable_explains_missing(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Explain missing", description="Missing model",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(defn, evidence_items=[ev])
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value
        assert len(result.missing_requirements) > 0


# ===================================================================
# 17. No guaranteed-outcome claims
# ===================================================================

class TestNoGuaranteedOutcomeClaims:
    """Results never claim guaranteed outcomes."""

    def test_no_guaranteed_in_description(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="No guarantee", description="Test",
            parameters=[_make_param()], evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(),
        )
        serialized = str(result.to_dict()).lower()
        assert "guaranteed savings" not in serialized
        assert "guaranteed benefit" not in serialized
        assert "will definitely" not in serialized

    def test_comparison_disclaims_guarantees(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=100.0, description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=200.0, description="R2",
        )
        comp = engine.compare_scenarios([r1, r2])
        assert any("not guaranteed" in n.lower() or "estimate" in n.lower()
                    for n in comp.comparison_notes)


# ===================================================================
# 18. No invented financial estimates
# ===================================================================

class TestNoInventedFinancialEstimates:
    """No invented currency, savings, revenue, or ROI."""

    def test_no_currency_in_result(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Throughput evidence")
        defn = engine.define_scenario(
            name="No currency", description="Generic scenario",
            parameters=[_make_param(name="throughput", unit="items/hour")],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"throughput": 1.5}, unit="items/hour"
            ),
        )
        serialized = str(result.to_dict()).lower()
        for term in ["$", "€", "£", "usd", "eur", "revenue", "savings", "roi"]:
            assert term not in serialized

    def test_no_fabricated_business_benefit(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="No benefits", description="Check no benefits invented",
            parameters=[_make_param()],
        )
        result = engine.evaluate_scenario(defn)
        serialized = str(result.to_dict()).lower()
        for term in ["cost saving", "revenue increase", "profit"]:
            assert term not in serialized


# ===================================================================
# 19. Generic non-marketing examples
# ===================================================================

class TestGenericNonMarketingExamples:
    """Engine works with non-marketing domains."""

    def test_healthcare_scenario(self):
        engine = ScenarioEngine(dataset_id="ds_hospital")
        ev = _make_evidence(
            description="Patient wait time data",
            source="hospital_analytics",
            dataset_id="ds_hospital",
        )
        defn = engine.define_scenario(
            name="Reduce wait times",
            description="What if nurse staffing increases by 15%?",
            parameters=[
                _make_param(
                    name="nurse_staff_count",
                    baseline_value=20, alternative_value=23,
                    unit="FTEs",
                ),
            ],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info={
                "coefficients": {"nurse_staff_count": -2.5},
                "intercept": 80.0,
                "model_type": "staffing model",
                "unit": "minutes",
            },
        )
        assert result.status == ScenarioStatus.SIMULATED.value
        assert result.unit == "minutes"

    def test_manufacturing_scenario(self):
        engine = ScenarioEngine(dataset_id="ds_factory")
        ev = _make_evidence(
            description="Production line data",
            dataset_id="ds_factory",
        )
        defn = engine.define_scenario(
            name="Machine speed",
            description="What if machine speed increases?",
            parameters=[
                _make_param(
                    name="machine_rpm",
                    baseline_value=1000, alternative_value=1200,
                    unit="RPM",
                ),
            ],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info={
                "coefficients": {"machine_rpm": 0.05},
                "intercept": 100.0,
                "model_type": "throughput model",
                "unit": "units/hour",
            },
        )
        assert result.status == ScenarioStatus.SIMULATED.value
        assert result.unit == "units/hour"

    def test_education_scenario(self):
        engine = ScenarioEngine(dataset_id="ds_school")
        ev = _make_evidence(
            description="Student performance data",
            dataset_id="ds_school",
        )
        defn = engine.define_scenario(
            name="Class size",
            description="What if class size is reduced?",
            parameters=[
                _make_param(
                    name="class_size",
                    baseline_value=30, alternative_value=20,
                    unit="students",
                ),
            ],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info={
                "coefficients": {"class_size": -0.5},
                "intercept": 85.0,
                "model_type": "performance model",
                "unit": "average_score",
            },
        )
        assert result.status == ScenarioStatus.SIMULATED.value

    def test_logistics_scenario(self):
        engine = ScenarioEngine(dataset_id="ds_logistics")
        ev = _make_evidence(
            description="Delivery route data",
            dataset_id="ds_logistics",
        )
        defn = engine.define_scenario(
            name="Fleet size",
            description="What if fleet is expanded?",
            parameters=[
                _make_param(
                    name="vehicle_count",
                    baseline_value=10, alternative_value=15,
                    unit="vehicles",
                ),
            ],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info={
                "coefficients": {"vehicle_count": -3.0},
                "intercept": 120.0,
                "model_type": "delivery time model",
                "unit": "avg_delivery_minutes",
            },
        )
        assert result.status == ScenarioStatus.SIMULATED.value

    def test_iot_scenario_no_model(self):
        engine = ScenarioEngine(dataset_id="ds_iot")
        ev = _make_evidence(
            description="Sensor reading data",
            dataset_id="ds_iot",
        )
        defn = engine.define_scenario(
            name="Sensor threshold",
            description="What if sensor threshold changes?",
            parameters=[
                _make_param(
                    name="temperature_threshold",
                    baseline_value=75.0, alternative_value=80.0,
                    unit="celsius",
                ),
            ],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(defn, evidence_items=[ev])
        assert result.status == ScenarioStatus.NOT_SIMULATABLE.value
        assert len(result.missing_requirements) > 0


# ===================================================================
# Partial simulation
# ===================================================================

class TestPartialSimulation:
    """Partial evaluation when some params have coefficients and some don't."""

    def test_partial_simulation_status(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        params = [
            _make_param(name="known", baseline_value=10, alternative_value=20),
            _make_param(name="unknown", baseline_value=5, alternative_value=10),
        ]
        defn = engine.define_scenario(
            name="Partial", description="Partial simulation",
            parameters=params, evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"known": 3.0},  # "unknown" not in coefficients
            ),
        )
        assert result.status == ScenarioStatus.PARTIALLY_SIMULATED.value
        assert "unknown" in result.explanation


# ===================================================================
# Define scenario evidence filtering
# ===================================================================

class TestDefineScenarioFiltering:
    """define_scenario filters evidence and captures notes."""

    def test_filtering_note_added(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev_same = _make_evidence(description="Same", dataset_id=DS_ID)
        ev_other = _make_evidence(description="Other", dataset_id="ds_x")
        engine.define_scenario(
            name="Filter",
            description="Filter test",
            parameters=[_make_param()],
            evidence_items=[ev_same, ev_other],
        )
        assert any("Filtered" in n for n in engine.processing_notes)

    def test_unsupported_param_creates_limitation(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        param = _make_param(supported=False, unsupported_reason="No data")
        defn = engine.define_scenario(
            name="Unsupported",
            description="Unsupported param",
            parameters=[param],
        )
        assert any(
            "Unsupported" in lim.description
            for lim in defn.limitations
            if hasattr(lim, "description")
        )


# ===================================================================
# Comparison helpers
# ===================================================================

class TestComparisonHelpers:
    """ScenarioComparison helper properties."""

    def test_best_outcome_returns_highest(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=50.0, description="Low",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            estimated_outcome=150.0, description="High",
        )
        comp = ScenarioComparison(
            dataset_id=DS_ID, results=[r1, r2],
        )
        assert comp.best_outcome.estimated_outcome == 150.0

    def test_best_outcome_none_when_no_numeric(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.NOT_SIMULATABLE.value,
            description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.NOT_SIMULATABLE.value,
            description="R2",
        )
        comp = ScenarioComparison(
            dataset_id=DS_ID, results=[r1, r2],
        )
        assert comp.best_outcome is None

    def test_all_simulated_true(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            description="R2",
        )
        comp = ScenarioComparison(dataset_id=DS_ID, results=[r1, r2])
        assert comp.all_simulated is True

    def test_all_simulated_false_with_mixed(self):
        r1 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.SIMULATED.value,
            description="R1",
        )
        r2 = ScenarioResult(
            dataset_id=DS_ID, status=ScenarioStatus.NOT_SIMULATABLE.value,
            description="R2",
        )
        comp = ScenarioComparison(dataset_id=DS_ID, results=[r1, r2])
        assert comp.all_simulated is False


# ===================================================================
# Zero baseline and negative values
# ===================================================================

class TestZeroAndNegativeValues:
    """Edge cases with zero baselines and negative values."""

    def test_zero_baseline_no_divide_by_zero(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Zero base",
            description="Zero baseline",
            parameters=[_make_param(baseline_value=0.0, alternative_value=10.0)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 1.0}, intercept=0.0,
            ),
        )
        # baseline_outcome = 0, delta_pct should be None (divide by zero)
        # Actually intercept + coeff * 0 = 0, intercept + coeff * 10 = 10
        # delta_pct = 10/0 → should not crash
        assert result.status == ScenarioStatus.SIMULATED.value

    def test_negative_values_allowed(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        ev = _make_evidence(description="Evidence")
        defn = engine.define_scenario(
            name="Negative",
            description="Negative values",
            parameters=[_make_param(baseline_value=-5, alternative_value=-10)],
            evidence_items=[ev],
        )
        result = engine.evaluate_scenario(
            defn, evidence_items=[ev],
            model_info=_make_model_info(
                coefficients={"param_a": 2.0}, intercept=0.0,
            ),
        )
        assert result.status == ScenarioStatus.SIMULATED.value
        assert result.baseline_outcome == -10.0
        assert result.estimated_outcome == -20.0

    def test_param_change_pct_zero_baseline(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        defn = engine.define_scenario(
            name="Zero pct", description="Zero baseline pct",
            parameters=[_make_param(baseline_value=0.0, alternative_value=10.0)],
        )
        assert defn.parameters[0].change_pct is None  # Division by zero → None


# ===================================================================
# Dict-based parameter handling
# ===================================================================

class TestDictBasedParameters:
    """Parameters can be provided as dicts."""

    def test_dict_parameter_validated(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        param_dict = {
            "name": "dict_param",
            "baseline_value": 10.0,
            "alternative_value": 20.0,
            "unit": "units",
        }
        defn = engine.define_scenario(
            name="Dict param", description="Dict parameter",
            parameters=[param_dict],
        )
        assert defn.parameters[0].name == "dict_param"
        assert defn.parameters[0].change_pct is not None


# ===================================================================
# Processing notes
# ===================================================================

class TestProcessingNotes:
    """Processing notes are properly collected."""

    def test_processing_notes_immutable_copy(self):
        engine = ScenarioEngine(dataset_id=DS_ID)
        notes = engine.processing_notes
        notes.append("should not affect engine")
        assert "should not affect engine" not in engine.processing_notes
